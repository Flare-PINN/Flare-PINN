# src/data/fetch_flares.py
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import pandas as pd
from pydantic import BaseModel, Field

# SunPy HEK/Fido
from sunpy.net import attrs as a, Fido
from sunpy.net import hek as hek_mod

from src.utils.common import load_cfg, ensure_dirs

LETTER_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


# ---------------- Pydantic Models ----------------

class FlareColumnMapping(BaseModel):
    """Strongly typed column mapping for HEK flare data."""
    event_starttime: str = Field(default="start", description="Start time column")
    event_peaktime: str = Field(default="peak", description="Peak time column") 
    event_endtime: str = Field(default="end", description="End time column")
    fl_goescls: str = Field(default="class", description="GOES class column")
    ar_noaanum: str = Field(default="noaa_ar", description="NOAA AR number column")
    
    def get_mapping_dict(self) -> Dict[str, str]:
        """Convert to dictionary for pandas rename operation."""
        return {
            "event_starttime": self.event_starttime,
            "event_peaktime": self.event_peaktime,
            "event_endtime": self.event_endtime,
            "fl_goescls": self.fl_goescls,
            "ar_noaanum": self.ar_noaanum,
        }


class FlareQueryConfig(BaseModel):
    """Configuration for flare query parameters."""
    provider: str = Field(default="auto", description="HEK provider (auto/fido/client)")
    timeout_s: float = Field(default=90.0, description="Query timeout in seconds")
    retries: int = Field(default=2, description="Number of retry attempts")
    min_days: int = Field(default=2, description="Minimum days for range splitting")
    sleep_base: float = Field(default=5.0, description="Base sleep time for retries")
    debug: bool = Field(default=False, description="Enable debug output")


# ---------------- GOES helpers ----------------

def parse_goes_class(s: str) -> Tuple[Optional[float], Optional[str]]:
    """'M1.2' -> (1.2e-5, 'M'); returns (None, None) if unparseable."""
    if not isinstance(s, str):
        return None, None
    m = re.fullmatch(r"\s*([ABCMX])\s*([0-9]+(?:\.[0-9]*)?)?\s*", s.upper())
    if not m:
        return None, None
    letter = m.group(1)
    mag = float(m.group(2) or 1.0)
    return mag * LETTER_FLUX[letter], letter


def cutoff_string_for_class(letter: str) -> str:
    """
    Server-side class cutoff: we want >= {letter}1.0 .
    SunPy supports > comparator, so use > '{letter}0.9' to include 1.0.
    """
    return f"{letter}0.9"


# ---------------- child process runner (NO POOLS) ----------------

def _to_temp_parquet(df: pd.DataFrame) -> str:
    """Convert DataFrame to parquet with proper type handling for PyArrow compatibility."""
    tmpdir = Path(tempfile.gettempdir()) / "hek_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / f"hek_{uuid.uuid4().hex}.parquet"
    
    # Clean DataFrame for PyArrow compatibility
    df_clean = _clean_dataframe_for_parquet(df)
    df_clean.to_parquet(p, index=False, engine='pyarrow')
    return str(p)


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to ensure PyArrow compatibility."""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Handle each column type properly
    for col in df_clean.columns:
        # Handle datetime columns first
        if 'datetime' in str(df_clean[col].dtype):
            # Ensure datetime columns are timezone-naive
            if hasattr(df_clean[col].dtype, 'tz') and df_clean[col].dtype.tz is not None:
                df_clean[col] = df_clean[col].dt.tz_localize(None)
        
        # Handle numeric columns
        elif df_clean[col].dtype.name in ['int64', 'int32', 'float64', 'float32']:
            # Keep numeric columns as-is, but ensure no inf values
            df_clean[col] = df_clean[col].replace([float('inf'), float('-inf')], None)
        
        # Handle object columns (strings, mixed types)
        elif df_clean[col].dtype == 'object':
            # Check if column contains mixed types or just strings
            sample_values = df_clean[col].dropna().head(10)
            if len(sample_values) > 0:
                # If all non-null values are strings, convert to string type
                if all(isinstance(val, str) for val in sample_values):
                    df_clean[col] = df_clean[col].astype('string')
                else:
                    # For mixed types, convert everything to string to avoid PyArrow issues
                    df_clean[col] = df_clean[col].astype(str)
                    # Replace 'nan' strings with None
                    df_clean[col] = df_clean[col].replace('nan', None)
        
        # Handle complex data types that PyArrow can't handle
        elif df_clean[col].dtype.name in ['complex', 'complex64', 'complex128']:
            df_clean[col] = df_clean[col].astype(str)
    
    return df_clean


def _child_run(conn: mp.connection.Connection, which: str, start: str, end: str, class_letter: str) -> None:
    try:
        if which == "fido":
            df = _query_fido_raw(start, end, class_letter)
        elif which == "client":
            df = _query_client_raw(start, end, class_letter)
        else:
            raise ValueError(f"unknown provider {which}")
        path = _to_temp_parquet(df)
        conn.send(("ok_path", path))
    except Exception as e:
        conn.send(("err", (type(e).__name__, str(e), traceback.format_exc())))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_with_timeout(provider: str, start: str, end: str, class_letter: str, timeout_s: float) -> pd.DataFrame:
    """
    Run a HEK query in a separate process with a hard timeout.
    If the child hangs, terminate it and raise TimeoutError (caller will retry/split).
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_run, args=(child, provider, start, end, class_letter))
    p.daemon = True
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(5)
        raise TimeoutError(f"operation exceeded {timeout_s}s")
    if not parent.poll(2):
        raise RuntimeError("child ended without result (likely crash)")
    status, payload = parent.recv()
    if status == "ok_path":
        path = payload
        try:
            return pd.read_parquet(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
    # error path
    exc_name, msg, tb = payload
    raise RuntimeError(f"{exc_name}: {msg}\n{tb}")


# ---------------- chunking & resume ----------------

def month_chunks(start: str, end: str, months_per_chunk: int = 1):
    """Split [start,end) into calendar-aligned month chunks."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    cur = s.replace(day=1)
    if s.day != 1:
        out.append((s.strftime("%Y-%m-%d"),
                    min(_add_months(cur, months_per_chunk), e).strftime("%Y-%m-%d")))
        cur = _add_months(cur, months_per_chunk)
    while cur < e:
        nxt = _add_months(cur, months_per_chunk)
        out.append((cur.strftime("%Y-%m-%d"), min(nxt, e).strftime("%Y-%m-%d")))
        cur = nxt
    return out


def _add_months(dt: datetime, n: int) -> datetime:
    y = dt.year + (dt.month - 1 + n) // 12
    m = (dt.month - 1 + n) % 12 + 1
    return dt.replace(year=y, month=m, day=1)


def dedupe_hek_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer a stable HEK id if present; else return df unchanged (dedupe later)."""
    for k in ("kb_archivid", "hek_id", "event_id", "kb_archividnum"):
        if k in df.columns:
            return df.drop_duplicates(subset=[k])
    return df


def should_skip_chunk(cfile: Path, resume: bool, force_months: Set[str]) -> Tuple[bool, str]:
    """
    Resume behavior ONLY: skip if parquet exists (unless month is forced).
    .error files NEVER trigger skip; they mean 'retry'.
    """
    if any(m in cfile.name for m in force_months):
        return False, "forced"
    if resume and cfile.exists():
        try:
            n = len(pd.read_parquet(cfile))
            return True, f"exists (rows={n})"
        except Exception:
            return False, "corrupted file, will refetch"
    # if there's an .error, we will retry (no skip)
    return False, "not processed yet"


# ---------------- HEK queries (server-side class filter) ----------------

def _query_fido_raw(start: str, end: str, class_letter: str) -> pd.DataFrame:
    # Filter to GOES + class cutoff on the server side
    cutoff = cutoff_string_for_class(class_letter)  # e.g., 'M0.9'
    resp = Fido.search(
        a.Time(start, end),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES",
        a.hek.FL.GOESCls > cutoff,
    )
    try:
        hek_tbl = resp["hek"] if "hek" in getattr(resp, "_data", {}) else resp[0]
    except Exception:
        return pd.DataFrame()
    if len(hek_tbl) == 0:
        return pd.DataFrame()
    # Filter out problematic columns that can't be serialized to parquet
    problematic_cols = {"event_coord"}  # SkyCoord objects can't be serialized
    cols = [c for c in hek_tbl.colnames if hek_tbl[c].ndim <= 1 and c not in problematic_cols]
    return hek_tbl[cols].to_pandas()


def _query_client_raw(start: str, end: str, class_letter: str) -> pd.DataFrame:
    cutoff = cutoff_string_for_class(class_letter)
    client = hek_mod.HEKClient()
    res = client.search(
        a.Time(start, end),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES",
        a.hek.FL.GOESCls > cutoff,
    )
    if len(res) == 0:
        return pd.DataFrame()
    # Filter out problematic columns that can't be serialized to parquet
    problematic_cols = {"event_coord"}  # SkyCoord objects can't be serialized
    cols = [c for c in res.colnames if res[c].ndim <= 1 and c not in problematic_cols]
    return res[cols].to_pandas()


def _query_once(provider: str, start: str, end: str, class_letter: str, timeout_s: float) -> pd.DataFrame:
    if provider == "fido":
        return run_with_timeout("fido", start, end, class_letter, timeout_s)
    if provider == "client":
        return run_with_timeout("client", start, end, class_letter, timeout_s)
    # auto: try fido then client
    try:
        return run_with_timeout("fido", start, end, class_letter, timeout_s)
    except Exception as e1:
        print(f"   note: Fido failed -> {type(e1).__name__}: {e1}", flush=True)
        return run_with_timeout("client", start, end, class_letter, timeout_s)


def query_with_retries(provider: str, start: str, end: str, class_letter: str,
                       timeout_s: float, retries: int, sleep_base: float,
                       debug: bool) -> pd.DataFrame:
    attempt = 0
    t0 = time.time()
    while True:
        try:
            print(f"   attempt {attempt+1}/{retries+1}: {start}→{end} (timeout={timeout_s}s)", flush=True)
            a0 = time.time()
            df = _query_once(provider, start, end, class_letter, timeout_s)
            print(f"   ✓ query done in {time.time()-a0:.1f}s (rows={len(df)})", flush=True)
            return df
        except Exception as e:
            attempt += 1
            if attempt > retries:
                print(f"   failed after {retries+1} attempts ({time.time()-t0:.1f}s): {e}", flush=True)
                if debug:
                    print(traceback.format_exc(), flush=True)
                raise
            delay = sleep_base * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
            print(f"   warn: {type(e).__name__}: {e} → retry {attempt}/{retries} in {delay:.1f}s", flush=True)
            time.sleep(delay)


def fetch_range_recursive(provider: str, start: str, end: str, class_letter: str,
                          timeout_s: float, retries: int, min_days: int,
                          sleep_base: float, debug: bool) -> pd.DataFrame:
    """
    Try whole range; on failure, split in half recursively until ≤ min_days.
    If one half succeeds and the other fails, return the successful half.
    If both fail, re-raise to signal the chunk should NOT be written.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if s >= e:
        return pd.DataFrame()

    try:
        print(f"   trying: {start}→{end}", flush=True)
        return query_with_retries(provider, start, end, class_letter, timeout_s, retries, sleep_base, debug)
    except Exception as e_whole:
        print(f"   whole range failed: {type(e_whole).__name__}: {e_whole}", flush=True)

    days = (e - s).days
    if days <= min_days:
        # do NOT pretend it's empty; bubble up so caller knows it failed
        raise TimeoutError(f"range {start}→{end} failed and ≤ min_days={min_days}")

    mid = (s + (e - s) / 2).strftime("%Y-%m-%d")
    print(f"   split: {start}→{mid} and {mid}→{end}", flush=True)
    time.sleep(0.4 + random.random() * 0.6)

    left_df = right_df = None
    left_err = right_err = None

    try:
        left_df = fetch_range_recursive(provider, start, mid, class_letter, timeout_s, retries, min_days, sleep_base, debug)
    except Exception as e_left:
        left_err = e_left
        print(f"   left failed {start}→{mid}: {e_left}", flush=True)

    try:
        right_df = fetch_range_recursive(provider, mid, end, class_letter, timeout_s, retries, min_days, sleep_base, debug)
    except Exception as e_right:
        right_err = e_right
        print(f"   right failed {mid}→{end}: {e_right}", flush=True)

    if left_df is None and right_df is None:
        # both sides failed → propagate (caller will not write an empty file)
        raise left_err or right_err or RuntimeError("both halves failed")

    if left_df is None or left_df.empty:
        return right_df if right_df is not None else pd.DataFrame()
    if right_df is None or right_df.empty:
        return left_df if left_df is not None else pd.DataFrame()
    return pd.concat([left_df, right_df], ignore_index=True)


def crosscheck_empty_with_alt_provider(provider: str, start: str, end: str, class_letter: str,
                                       timeout_s: float) -> bool:
    """
    If first provider returned empty, confirm with the alternate provider
    so we don't write false zeros due to a provider glitch.
    Returns True if ALSO empty with the alternate, else False.
    """
    alt = {"auto": "client", "client": "fido", "fido": "client"}[provider]
    print(f"   empty result; cross-checking with {alt}...", flush=True)
    try:
        df_alt = _query_once(alt, start, end, class_letter, timeout_s)
        print(f"   alt provider rows={len(df_alt)}", flush=True)
        return df_alt.empty
    except Exception as e:
        print(f"   alt provider failed ({alt}): {e}; assuming NOT confirmed-empty", flush=True)
        return False


# ---------------- CLI ----------------

@click.command()
@click.option("--cfg", default="configs/data_train.yaml", show_default=True)
@click.option("--class_min", default="M", type=click.Choice(["C", "M", "X"]), show_default=True)
@click.option("--chunk-months", default=1, show_default=True)
@click.option("--timeout", default=90, show_default=True, help="Per-attempt hard timeout (seconds).")
@click.option("--retries", default=2, show_default=True)
@click.option("--min-days", default=2, show_default=True, help="Stop splitting when range ≤ this many days.")
@click.option("--sleep-base", default=5.0, show_default=True)
@click.option("--between-chunks-sleep", default=8.0, show_default=True)
@click.option("--resume/--force", default=True, show_default=True, help="Resume skips finished chunk files; force re-fetches.")
@click.option("--max-chunk-time", default=0, show_default=True, help="Soft budget per chunk; 0 disables.")
@click.option("--debug", is_flag=True)
@click.option("--provider", default="auto", type=click.Choice(["auto", "client", "fido"]), show_default=True,
              help="Backend for HEK queries.")
@click.option("--force-months", default="", show_default=True,
              help="Comma-separated YYYY-MM list to re-fetch even if parquet exists.")
@click.option("--skip-problematic", is_flag=True, help="No-op (kept for compatibility).")
def main(cfg, class_min, chunk_months, timeout, retries, min_days, sleep_base,
          between_chunks_sleep, resume, max_chunk_time, debug, provider, force_months, skip_problematic):

    # Windows-friendly MP
    try:
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    data_config = load_cfg(cfg)
    ensure_dirs(data_config)

    start, end = data_config.span.start, data_config.span.end
    base_dir = Path(data_config.paths.interim_dir)
    chunks_dir = base_dir / "hek_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "flares_hek.parquet"

    class_letter = class_min.upper()
    thr = {"C": 1e-6, "M": 1e-5, "X": 1e-4}[class_letter]

    # parse force-months
    fm: Set[str] = set()
    for tok in [t.strip() for t in force_months.split(",") if t.strip()]:
        if re.fullmatch(r"\d{4}-\d{2}", tok):
            fm.add(tok)

    print(f"[fetch_flares] span={start}→{end} class_min={class_min} provider={provider} (resume={resume})", flush=True)
    print(f" chunks_dir = {chunks_dir}", flush=True)
    print(f" out_path   = {out_path}", flush=True)
    if fm:
        print(f" force-months = {sorted(fm)}", flush=True)

    chunks = month_chunks(start, end, months_per_chunk=chunk_months)
    print(f" top-level chunks: {len(chunks)}", flush=True)

    successful = failed = skipped = 0

    for i, (c0, c1) in enumerate(chunks, 1):
        tag = f"{class_letter}_{c0}_{c1}".replace(":", "-")
        cfile = chunks_dir / f"{tag}.parquet"

        print(f"\n[{i}/{len(chunks)}] {c0}→{c1}", flush=True)

        do_skip, reason = should_skip_chunk(cfile, resume, fm)
        if do_skip:
            print(f" ✓ skip ({reason}): {cfile.name}", flush=True)
            skipped += 1
            time.sleep(0.25)
            continue

        tchunk = time.time()
        try:
            print(f" fetching (server-filtered) {class_letter}-class flares...", flush=True)
            df_raw = fetch_range_recursive(provider, c0, c1, class_letter, timeout, retries, min_days, sleep_base, debug)

            # If the selected provider returned empty, cross-check with alternate provider
            # to avoid false empties. Only accept empty if both agree.
            if df_raw.empty:
                if crosscheck_empty_with_alt_provider(provider, c0, c1, class_letter, timeout):
                    print(" · confirmed empty month (both providers)", flush=True)
                    df_to_write = pd.DataFrame(columns=["start", "peak", "end", "class", "noaa_ar", "hpc_x", "hpc_y", "hgs_lon", "hgs_lat"])
                else:
                    # Treat as transient failure – do not write a 0-row file
                    raise RuntimeError("empty not confirmed by alternate provider; treating as failure")

            else:
                # dedupe & normalize columns
                df_raw = dedupe_hek_df(df_raw)
                keep_map = {
                    "event_starttime": "start",
                    "event_peaktime": "peak",
                    "event_endtime": "end",
                    "fl_goescls": "class",
                    "ar_noaanum": "noaa_ar",
                    "hpc_x": "hpc_x",
                    "hpc_y": "hpc_y",
                    "hgs_x": "hgs_lon",  # Heliographic Stonyhurst longitude
                    "hgs_y": "hgs_lat",  # Heliographic Stonyhurst latitude
                }
                keep_src = [k for k in keep_map if k in df_raw.columns]
                df = df_raw.rename(columns=keep_map).filter([keep_map[k] for k in keep_src], axis=1)

                # parse GOES class locally and post-filter (belt & suspenders)
                flux_letter = [parse_goes_class(s) for s in df.get("class", pd.Series([None] * len(df)))]
                df["flux_Wm2"] = [fl for fl, _ in flux_letter]
                df["letter"] = [lt for _, lt in flux_letter]
                df = df.dropna(subset=["flux_Wm2", "letter"]).copy()

                # types
                if "start" in df: df["start"] = pd.to_datetime(df["start"], errors="coerce")
                if "peak"  in df: df["peak"]  = pd.to_datetime(df["peak"],  errors="coerce")
                if "end"   in df: df["end"]   = pd.to_datetime(df["end"],   errors="coerce")
                if "noaa_ar" in df: df["noaa_ar"] = pd.to_numeric(df["noaa_ar"], errors="coerce")

                df = (
                    df[(df["letter"] == class_letter) & (df["flux_Wm2"] >= thr)]
                    .drop(columns=["flux_Wm2", "letter"], errors="ignore")
                    .drop_duplicates()
                    .sort_values("start")
                    .reset_index(drop=True)
                )

                df_to_write = df

            # write parquet (either confirmed-empty or real rows)
            df_clean = _clean_dataframe_for_parquet(df_to_write)
            
            # Write to temporary file first, then move atomically
            temp_file = cfile.with_suffix('.tmp')
            try:
                df_clean.to_parquet(temp_file, index=False, engine='pyarrow')
                # Verify the file was written correctly
                test_df = pd.read_parquet(temp_file)
                if len(test_df) != len(df_to_write):
                    raise ValueError(f"Data integrity check failed: expected {len(df_to_write)} rows, got {len(test_df)}")
                # Atomic move
                temp_file.replace(cfile)
                print(f" ✓ wrote {cfile.name} (rows={len(df_to_write)})", flush=True)
            except Exception as e:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise e
            successful += 1

            # clear any prior error marker
            errf = cfile.with_suffix(".error")
            if errf.exists():
                errf.unlink()

        except KeyboardInterrupt:
            print(" !! interrupted; will resume next run", flush=True)
            break
        except Exception as e:
            print(f" ✗ error: {type(e).__name__}: {e} (chunk NOT saved; will retry next run)", flush=True)
            if debug:
                print(traceback.format_exc(), flush=True)
            failed += 1
            with open(cfile.with_suffix(".error"), "w", encoding="utf-8") as f:
                f.write(f"Error {c0}→{c1}: {type(e).__name__}: {e}\n")
                f.write(f"{datetime.now().isoformat()}\n")

        # soft budget warning only
        if max_chunk_time and (time.time() - tchunk) > max_chunk_time:
            print(f" ⚠ chunk exceeded soft budget: {(time.time()-tchunk):.1f}s > {max_chunk_time}s", flush=True)

        # be polite to HEK
        time.sleep(between_chunks_sleep)

    # summary
    print(f"\n[summary] processed {len(chunks)} chunks:", flush=True)
    print(f" successful: {successful}", flush=True)
    print(f" failed:     {failed}", flush=True)
    print(f" skipped:    {skipped}", flush=True)

    # combine
    files = sorted(chunks_dir.glob(f"{class_letter}_*.parquet"))
    out_cols = ["start", "peak", "end", "class", "noaa_ar", "hpc_x", "hpc_y", "hgs_lon", "hgs_lat"]
    if not files:
        print("WARN: no chunk files found; writing empty final parquet", flush=True)
        empty_df = pd.DataFrame(columns=out_cols)
        empty_clean = _clean_dataframe_for_parquet(empty_df)
        empty_clean.to_parquet(out_path, index=False, engine='pyarrow')
        print(f"Wrote: {out_path} (rows=0)", flush=True)
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f" skip broken file {f.name}: {e}", flush=True)

    if not dfs:
        print("WARN: all chunk files unreadable; writing empty final parquet", flush=True)
        empty_df = pd.DataFrame(columns=out_cols)
        empty_clean = _clean_dataframe_for_parquet(empty_df)
        empty_clean.to_parquet(out_path, index=False, engine='pyarrow')
        print(f"Wrote: {out_path} (rows=0)", flush=True)
        return

    full = pd.concat(dfs, ignore_index=True).drop_duplicates().sort_values("start").reset_index(drop=True)
    full_clean = _clean_dataframe_for_parquet(full)
    
    # Write final file atomically
    temp_out = out_path.with_suffix('.tmp')
    try:
        full_clean.to_parquet(temp_out, index=False, engine='pyarrow')
        # Verify the file was written correctly
        test_df = pd.read_parquet(temp_out)
        if len(test_df) != len(full):
            raise ValueError(f"Final file integrity check failed: expected {len(full)} rows, got {len(test_df)}")
        # Atomic move
        temp_out.replace(out_path)
        print(f"\n[done] combined → {out_path} (rows={len(full)})", flush=True)
    except Exception as e:
        # Clean up temp file on error
        if temp_out.exists():
            temp_out.unlink()
        raise e


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
