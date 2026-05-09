#!/usr/bin/env python3
"""
download_sharp_cea.py — only download missing, resume-safe, fast

- For each chunk/day we create one JSOC export and read its URL list.
- We only download URLs whose .npz is not present locally.
- If a .fits exists but .npz is missing, we just convert it (no re-download).
- Whole-day skip when counts already match.
- Robust FITS→NPZ (find first data HDU; bitmap→uint8 mask; float32 data).
- Works well with long spans and restarts.

Requires: drms, pandas, tqdm, tenacity, astropy, numpy
"""

import argparse, os, sys, time, math, socket, random, logging, io, zipfile, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import drms
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock, local as _thread_local

# ------------------ constants ------------------
SERIES = "hmi.sharp_cea_720s"
DEFAULT_SEGMENTS = ["Br", "Bt", "Bp", "bitmap"]
DEFAULT_STEP = "1h"

# ------------------ args ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Download HMI SHARP CEA data; only fetch missing files.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    p.add_argument("--end",   required=True, help="YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--out",   default="./sharp_download", help="Output directory")
    p.add_argument("--segments", nargs="+", default=DEFAULT_SEGMENTS)
    p.add_argument("--chunk", choices=["day", "month"], default="day")
    p.add_argument("--step", default=DEFAULT_STEP, help="Cadence for T_REC (e.g. 1h, 12m).")
    p.add_argument("--harpnums", nargs="*", default=None, help="HARPNUMs to include (space/comma separated)")
    p.add_argument("--estimate", action="store_true", help="Only estimate record counts; no downloads.")
    p.add_argument("--probe-mins", type=float, default=0.0, help="Stop estimate after ~N minutes (best effort).")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="Parallel network workers")
    p.add_argument("--proc-workers", type=int, default=None, help="CPU workers for FITS→NPZ; 0 disables pool")
    p.add_argument("--delete-after", action="store_true", help="Delete FITS after successful convert")
    p.add_argument("--retry-forever", action="store_true")
    p.add_argument("--net-poll", type=int, default=30, help="Seconds between connectivity checks.")
    p.add_argument("--socket-timeout", type=int, default=90, help="Global socket timeout (s)")
    return p.parse_args()

# ------------------ env/log ------------------
def jsoc_email_or_die() -> str:
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        print("ERROR: set JSOC_EMAIL (e.g. $env:JSOC_EMAIL='you@example.com')", file=sys.stderr)
        sys.exit(2)
    return email

def setup_logging(verbose: bool):
    lvl = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

log = logging.getLogger("sharp")
_THREAD = _thread_local()

# ------------------ utils ------------------
def have_network(host="171.64.103.244", port=80, timeout=5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def daterange_chunks(start: datetime, end: datetime, mode: str) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    if mode == "day":
        step = timedelta(days=1)
        while cur < end:
            nxt = min(cur + step, end)
            yield cur, nxt
            cur = nxt
    else:
        def add_month(d: datetime) -> datetime:
            y, m = d.year, d.month
            return datetime(y + (m // 12), (m % 12) + 1, 1)
        while cur < end:
            nxt = add_month(cur)
            yield cur, min(nxt, end)
            cur = nxt

def segment_brace(segments: List[str]) -> str:
    return "{" + ",".join(segments) + "}"

def csv_append(path: Path, df: pd.DataFrame):
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

def looks_like_day_dir(p: Path) -> bool:
    n = p.name
    return len(n) == 10 and n[4] == '-' and n[7] == '-'

# ------------------ recordset ------------------
def make_recordset(start_dt: datetime, end_dt: datetime,
                   harpnums: Optional[List[int]], step: Optional[str]) -> List[str]:
    t0 = start_dt.strftime("%Y.%m.%d_%H:%M:%S_UTC")
    dur_days = (end_dt - start_dt).days + (end_dt - start_dt).seconds / 86400.0
    ndays = max(1, math.ceil(dur_days))
    step_part = f"@{step}" if step else ""
    if not harpnums:
        return [f"{SERIES}[][{t0}/{ndays}d{step_part}]"]
    return [f"{SERIES}[{int(h)}][{t0}/{ndays}d{step_part}]" for h in harpnums]

# ------------------ NPZ writer ------------------
def _np_to_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()

def save_npz_fast(out_path: Path, arrays: dict, compresslevel: int = 3):
    tmp = out_path.with_suffix(".npz.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for name, arr in arrays.items():
            if isinstance(arr, dict):
                obj = np.empty(1, dtype=object)
                obj[0] = arr
                b = io.BytesIO()
                np.save(b, obj, allow_pickle=True)
                zf.writestr(name + ".npy", b.getvalue())
            else:
                zf.writestr(name + ".npy", _np_to_bytes(arr))
    tmp.replace(out_path)

def postprocess_one_file(path: Path) -> Path:
    path = Path(path)
    out_path = path.with_suffix(".npz")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with fits.open(path, memmap=False) as hdul:
        # find first data-bearing HDU (robust to primary-only/extension files)
        hdu = None
        for h in hdul:
            if getattr(h, "data", None) is not None:
                hdu = h
                break
        if hdu is None:
            raise ValueError("No data-containing HDU found")
        hdr = hdu.header
        arr = np.array(hdu.data)
        seg = (hdr.get("SEGMENT") or hdr.get("EXTNAME") or hdr.get("CONTENT") or "")
        if not seg:
            fn = path.name.lower()
            if ".bitmap.fits" in fn: seg = "bitmap"
            elif ".br.fits" in fn:   seg = "Br"
            elif ".bt.fits" in fn:   seg = "Bt"
            elif ".bp.fits" in fn:   seg = "Bp"
        seg_l = str(seg).lower()

        def _s(x):
            if isinstance(x, bytes):
                try: return x.decode("ascii", "ignore")
                except Exception: return str(x)
            return x

        meta = {
            "segment": _s(seg),
            "FILENAME": path.name,
            "T_REC": _s(hdr.get("T_REC") or hdr.get("DATE-OBS")),
            "HARPNUM": int(hdr.get("HARPNUM") or hdr.get("HARP_NUM") or -1),
            "CDELT1": hdr.get("CDELT1"), "CDELT2": hdr.get("CDELT2"),
            "CRVAL1": hdr.get("CRVAL1"), "CRVAL2": hdr.get("CRVAL2"),
            "CRPIX1": hdr.get("CRPIX1"), "CRPIX2": hdr.get("CRPIX2"),
            "RSUN_OBS": hdr.get("RSUN_OBS"),
        }

        if "bitmap" in seg_l:
            mask = (arr > 0).astype(np.uint8, copy=False)
            save_npz_fast(out_path, {"mask": mask, "meta": meta})
        else:
            data = arr.astype(np.float32, copy=False)
            save_npz_fast(out_path, {"data": data, "meta": meta})
    return out_path

# ------------------ http ------------------
def http_download(url: str, dest: Path, timeout: int = 90):
    tmp = dest.with_suffix(dest.suffix + ".part")
    ensure_dir(dest.parent)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)
    tmp.replace(dest)

# ------------------ drms client ------------------
def get_client(email: str) -> drms.Client:
    c = getattr(_THREAD, "client", None)
    if c is None:
        try: c = drms.Client(email=email, verbose=False)
        except TypeError: c = drms.Client(email=email)
        _THREAD.client = c
    return c

# For SHARP CEA, stick to url+fits (as-is can yield "record" URLs)
@retry(reraise=True, stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=60),
       retry=retry_if_exception_type((drms.DrmsError, TimeoutError)))
def submit_export(client: drms.Client, rec: str, segments: List[str]) -> drms.ExportRequest:
    method = "url"
    protocol = "fits"
    return client.export(f"{rec}{segment_brace(segments)}", method=method, protocol=protocol)

# ------------------ completeness ------------------
def day_is_complete(rec: str, subdir: Path, segments: List[str], email: str, verbose: bool) -> bool:
    if not looks_like_day_dir(subdir):  # only for day chunk dirs
        return False
    try:
        df = get_client(email).query(rec, key="HARPNUM,T_REC")
        expected_records = 0 if df is None else int(len(df))
    except Exception as e:
        if verbose: log.info(f"Skip-check query failed for {subdir.name}: {e}")
        return False
    expected_npz = expected_records * max(1, len(segments))
    if expected_npz == 0:
        if verbose: log.info(f"{subdir.name}: nothing expected; skipping.")
        return True
    have_npz = sum(1 for _ in subdir.glob("*.npz"))
    if verbose: log.info(f"{subdir.name}: expected {expected_npz} .npz; have {have_npz}")
    return have_npz >= expected_npz

# ------------------ worker ------------------
def _download_task(args_tuple):
    (rec, subdir, segments, sleep_s, verbose,
     delete_after, max_retries, retry_forever, net_poll, proc_pool, email) = args_tuple

    # Skip check removed for performance - local file checks are faster

    rows = []
    attempt = 0
    while True:
        if not have_network():
            if verbose: log.info(f"No network for {rec}. Rechecking in {net_poll}s…")
            time.sleep(net_poll); continue
        try:
            exp = submit_export(get_client(email), rec, segments)
            urls_obj = getattr(exp, "urls", None)
            url_list = [str(u) for u in urls_obj] if urls_obj is not None else []
            break
        except Exception as e:
            attempt += 1
            if not retry_forever and attempt >= max_retries:
                return [], f"{rec} :: export failed: {e}"
            back = min(60, 2 ** min(attempt, 6)) * (0.5 + random.random())
            if verbose: log.warning(f"{rec} export failed (attempt {attempt}): {e} -> retry in {back:.1f}s")
            time.sleep(back)

    # If ANY url is not http(s), use drms' downloader for this chunk
    def _is_http(u: str) -> bool:
        return u.lower().startswith("http://") or u.lower().startswith("https://")

    if not url_list or any(not _is_http(u) for u in url_list):
        # Fallback: let drms download everything for the chunk
        try:
            dl_df = exp.download(str(subdir))
        except Exception as e:
            return [], f"{rec} :: drms.download failed: {e}"

        if dl_df is not None and "download" in dl_df.columns:
            for _, row in dl_df.iterrows():
                local = row.get("download"); url = row.get("url")
                if not local: continue
                path = Path(local)
                if not path.exists(): continue
                out_npz = path.with_suffix(".npz")

                # Only convert if npz missing; else just delete the newly downloaded FITS if requested
                ok = True
                if not (out_npz.exists() and out_npz.stat().st_size > 0):
                    try:
                        if proc_pool is None:
                            out_npz = postprocess_one_file(path)
                        else:
                            out_npz = proc_pool.submit(postprocess_one_file, path).result()
                    except Exception as e:
                        ok = False
                        if verbose: log.warning(f"postprocess failed: {path.name}: {e}")

                rows.append({
                    "recordset": rec, "url": url,
                    "fits_path": str(path.resolve()),
                    "npz_path": str(out_npz),
                    "npz_exists": Path(out_npz).exists(),
                    "postprocessed": ok
                })
                # If npz already existed OR conversion succeeded, optionally delete FITS
                if delete_after and (ok or out_npz.exists()):
                    try: path.unlink(missing_ok=True)
                    except Exception: pass

        if sleep_s > 0: time.sleep(sleep_s)
        return rows, None

    # Otherwise, we have proper HTTP URLs → only fetch truly missing
    wanted = []
    for url in url_list:
        fname = url.split("/")[-1].split("?")[0]
        fits_path = subdir / fname
        npz_path  = fits_path.with_suffix(".npz")
        if npz_path.exists() and npz_path.stat().st_size > 0:
            continue  # already have npz
        wanted.append((url, fits_path, npz_path))

    if not wanted:
        if sleep_s > 0: time.sleep(sleep_s)
        return rows, None

    # Prefer converting existing FITS; otherwise download
    to_convert = []
    for url, fits_path, npz_path in wanted:
        if fits_path.exists() and fits_path.stat().st_size > 0:
            to_convert.append((url, fits_path))
            continue

        dl_attempt = 0
        while True:
            try:
                http_download(url, fits_path, timeout=90)
                break
            except Exception as e:
                dl_attempt += 1
                if not retry_forever and dl_attempt >= max_retries:
                    rows.append({
                        "recordset": rec, "url": url,
                        "fits_path": str(fits_path.resolve()),
                        "npz_path": str(npz_path),
                        "npz_exists": npz_path.exists(),
                        "postprocessed": False
                    })
                    try: fits_path.unlink(missing_ok=True)
                    except Exception: pass
                    url = None
                    break
                back = min(60, 2 ** min(dl_attempt, 6)) * (0.5 + random.random())
                if verbose: log.warning(f"{fits_path.name} download failed (attempt {dl_attempt}): {e} -> retry in {back:.1f}s")
                time.sleep(back)
        if url:
            to_convert.append((url, fits_path))

    # Convert (pool or inline)
    if proc_pool is None:
        for url, fits_path in to_convert:
            ok = True
            try:
                out_npz = postprocess_one_file(fits_path)
            except Exception as e:
                ok = False
                if verbose: log.warning(f"postprocess failed: {fits_path.name}: {e}")
                out_npz = fits_path.with_suffix(".npz")
            rows.append({
                "recordset": rec, "url": url,
                "fits_path": str(fits_path.resolve()),
                "npz_path": str(out_npz),
                "npz_exists": Path(out_npz).exists(),
                "postprocessed": ok
            })
            if delete_after and ok:
                try: fits_path.unlink(missing_ok=True)
                except Exception: pass
    else:
        futs = [(url, fits_path, proc_pool.submit(postprocess_one_file, fits_path)) for (url, fits_path) in to_convert]
        for url, fits_path, fut in futs:
            ok = True
            out_npz = fits_path.with_suffix(".npz")
            try:
                out_npz = fut.result()
            except Exception as e:
                ok = False
                if verbose: log.warning(f"postprocess failed: {fits_path.name}: {e}")
            rows.append({
                "recordset": rec, "url": url,
                "fits_path": str(fits_path.resolve()),
                "npz_path": str(out_npz),
                "npz_exists": Path(out_npz).exists(),
                "postprocessed": ok
            })
            if delete_after and ok:
                try: fits_path.unlink(missing_ok=True)
                except Exception: pass

    if sleep_s > 0: time.sleep(sleep_s)
    return rows, None

def main():
    args = parse_args()
    setup_logging(args.verbose)
    email = jsoc_email_or_die()

    socket.setdefaulttimeout(max(1, int(args.socket_timeout)))

    out_root = Path(args.out).expanduser().resolve()
    ensure_dir(out_root)

    start_dt = datetime.fromisoformat(args.start)
    end_dt   = datetime.fromisoformat(args.end)
    if end_dt <= start_dt:
        print("ERROR: end must be after start.", file=sys.stderr)
        sys.exit(2)

    harpnums = None
    if args.harpnums:
        joined = " ".join(args.harpnums).replace(",", " ").split()
        harpnums = [int(s) for s in joined if s.strip().isdigit()]
        if args.verbose: log.info(f"Filtering HARPNUMs: {harpnums}")

    manifest_path = out_root / "manifest.csv"

    # -------- Estimate mode --------
    if args.estimate:
        rows = []
        client = get_client(email)
        time_limit = time.time() + args.probe_mins * 60.0 if args.probe_mins > 0 else None
        chunks = list(daterange_chunks(start_dt, end_dt, args.chunk))
        total_chunks = len(chunks); done_chunks = 0; t0 = time.time()

        for c_start, c_end in tqdm(chunks, desc="Estimating", unit=args.chunk):
            subtotal = 0
            for rec in make_recordset(c_start, c_end, harpnums, args.step):
                try:
                    df = client.query(rec, key="HARPNUM,T_REC")
                    n = 0 if df is None else len(df)
                    subtotal += int(n) * max(1, len(args.segments))
                except Exception as e:
                    if args.verbose: log.warning(f"Estimate query failed for {rec}: {e}")
            rows.append({"chunk_start": c_start.isoformat(),
                         "chunk_end": c_end.isoformat(),
                         "expected_npz": subtotal})
            done_chunks += 1
            if args.sleep > 0: time.sleep(args.sleep)
            if time_limit is not None and time.time() > time_limit: break

        est = pd.DataFrame(rows)
        est.to_csv(out_root / "estimate.csv", index=False)
        partial = int(est["expected_npz"].clip(lower=0).sum()) if not est.empty else 0
        if done_chunks and done_chunks < total_chunks:
            extrap = int(round(partial * (total_chunks / done_chunks)))
            mins = (time.time() - t0)/60.0
            print(f"\nEstimate (partial {done_chunks}/{total_chunks} chunks in {mins:.1f} min):")
            print(f"  Count so far: {partial} .npz")
            print(f"  Extrapolated total: ~{extrap} .npz (rough)")
        else:
            print(f"\nEstimated total .npz files: {partial}")
        print(f"Wrote per-chunk details to: {out_root / 'estimate.csv'}")
        return

    # -------- Build tasks --------
    tasks = []
    chunk_info = []
    for c_start, c_end in daterange_chunks(start_dt, end_dt, args.chunk):
        subdir = out_root / f"{c_start:%Y}" / f"{c_start:%m}" / (f"{c_start:%Y-%m-%d}" if args.chunk == "day" else "")
        ensure_dir(subdir)
        for rec in make_recordset(c_start, c_end, harpnums, args.step):
            tasks.append((rec, subdir, args.segments,
                          args.sleep, args.verbose, args.delete_after,
                          args.max_retries, args.retry_forever, args.net_poll,
                          None, email))  # proc_pool placeholder
            chunk_info.append((rec, c_start.isoformat(), c_end.isoformat()))

    if not tasks:
        print("No tasks to run.")
        return

    rec_to_bounds = {rec: (start, end) for rec, start, end in chunk_info}
    manifest_lock = Lock()

    # Respect --proc-workers 0
    proc_workers = 0 if args.proc_workers == 0 else (args.proc_workers or os.cpu_count() or 2)

    try:
        if proc_workers == 0:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                tasks_np = []
                for t in tasks:
                    t = list(t); t[9] = None  # no pool
                    tasks_np.append(tuple(t))
                futures = [ex.submit(_download_task, t) for t in tasks_np]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="task"):
                    rows, err = fut.result()
                    if err: print(f"[WARN] {err}", file=sys.stderr)
                    if rows:
                        for r in rows:
                            se = rec_to_bounds.get(r["recordset"])
                            if se: r["start"], r["end"] = se
                        with manifest_lock: csv_append(manifest_path, pd.DataFrame(rows))
        else:
            with ProcessPoolExecutor(max_workers=proc_workers) as proc_pool, \
                 ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                tasks_with_pool = []
                for t in tasks:
                    t = list(t); t[9] = proc_pool
                    tasks_with_pool.append(tuple(t))
                futures = [ex.submit(_download_task, t) for t in tasks_with_pool]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="task"):
                    rows, err = fut.result()
                    if err: print(f"[WARN] {err}", file=sys.stderr)
                    if rows:
                        for r in rows:
                            se = rec_to_bounds.get(r["recordset"])
                            if se: r["start"], r["end"] = se
                        with manifest_lock: csv_append(manifest_path, pd.DataFrame(rows))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Any completed items were appended to the manifest.")

    print(f"\nDone. Manifest at: {manifest_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    main()
