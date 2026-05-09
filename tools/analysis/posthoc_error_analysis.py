#!/usr/bin/env python3
"""
Post-hoc error analysis for flare forecasts (no GPU).

Uses validate_checkpoint-style NPZ (probs, labels, thresholds, horizons) aligned row-for-row
with a windows Parquet (same sort as validate_checkpoint: by t0 stable).

1) Spatial: if you provide frame metadata with cmd_deg, compares FP/FN/TP/TN vs |CMD|
   (e.g. fraction with |CMD| > 60° for limb-related inversion uncertainty).

2) False positives: probability stats, obs_coverage / masking columns, and optionally
   "near-miss" flares (C-class in horizon) if flare catalog + harp→NOAA mapping are given.

Usage
-----
  PYTHONPATH=.:src python tools/analysis/posthoc_error_analysis.py \\
    --npz outputs/checkpoints/PINN_Final_46k/validation_results/best_model_test.npz \\
    --windows data/windows_test_15.parquet \\
    --horizon-hours 24

  # With CMD (columns: harpnum, date_obs, cmd_deg):
  ... --frames-metadata path/to/frames_meta.parquet

  # With flare profiling (flares: start, class, noaa_ar | noaa_num; mapping: harpnum, noaa_ar):
  ... --flare-catalog data/flares_hek.parquet --harp-noaa data/harp_noaa.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy.time import Time as AstroTime
from sunpy.coordinates.sun import L0 as _sun_L0
try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None  # type: ignore[misc, assignment]

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
for _p in (_REPO, _SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _goes_class_letter(cls_str: str) -> str:
    s = str(cls_str).strip().upper()
    if not s:
        return ""
    return s[0] if s[0] in "ABCMX" else ""


def _class_rank(letter: str) -> int:
    order = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    return order.get(letter.upper(), 0)


def _carrington_lon_to_cmd(carr_lon: np.ndarray, times: pd.Series) -> np.ndarray:
    """Convert Carrington longitude to Central Meridian Distance (degrees, signed).

    CMD = Carrington_lon - L0(t), wrapped to [-180, 180].
    Positive = west of central meridian.
    """
    unique_times = times.drop_duplicates().sort_values()
    l0_map: dict[pd.Timestamp, float] = {}
    for t in unique_times:
        l0_map[t] = float(_sun_L0(AstroTime(t.to_pydatetime())).deg)
    l0_vals = np.array([l0_map[t] for t in times])
    cmd = carr_lon - l0_vals
    cmd = ((cmd + 180.0) % 360.0) - 180.0
    return cmd


def _attach_cmd_per_window(windows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Nearest date_obs Carrington lon per (harpnum, t0), then convert to real CMD.

    Preserves original row order (NPZ alignment).
    """
    need = {"harpnum", "date_obs", "cmd_deg"}
    miss = need - set(meta.columns)
    if miss:
        raise ValueError(f"frames-metadata missing columns: {miss}")
    m = meta.copy()
    m["date_obs"] = pd.to_datetime(m["date_obs"], utc=True)
    m["harpnum"] = m["harpnum"].astype(np.int64)
    w = windows.copy()
    w["t0"] = pd.to_datetime(w["t0"], utc=True)
    w["harpnum"] = w["harpnum"].astype(np.int64)
    w["_row_id"] = np.arange(len(w), dtype=np.int64)
    w = w.sort_values("t0", kind="stable")
    m = m.sort_values("date_obs", kind="stable")
    right = m.rename(columns={"date_obs": "cmd_time"})[["harpnum", "cmd_time", "cmd_deg"]]
    merged = pd.merge_asof(
        w,
        right,
        by="harpnum",
        left_on="t0",
        right_on="cmd_time",
        direction="nearest",
    )
    merged = merged.sort_values("_row_id", kind="stable").drop(
        columns=["_row_id", "cmd_time"], errors="ignore"
    )
    carr = pd.to_numeric(merged["cmd_deg"], errors="coerce").values
    merged["cmd_deg"] = _carrington_lon_to_cmd(carr, merged["t0"])
    return merged


def _augment_flares(
    df: pd.DataFrame,
    flares: pd.DataFrame,
    harp_noaa: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """Add max_class_letter and near_miss_c for each row (requires NOAA via harp)."""
    f = flares.copy()
    if "start" not in f.columns:
        raise ValueError("flare-catalog needs column 'start'")
    cls_col = "class" if "class" in f.columns else ("CLASS" if "CLASS" in f.columns else None)
    if cls_col is None:
        raise ValueError("flare-catalog needs column 'class' or 'CLASS'")
    noaa_col = "noaa_ar" if "noaa_ar" in f.columns else ("noaa_num" if "noaa_num" in f.columns else None)
    if noaa_col is None:
        raise ValueError("flare-catalog needs noaa_ar or noaa_num")
    f["start"] = pd.to_datetime(f["start"], utc=True)
    f["_letter"] = f[cls_col].map(_goes_class_letter)

    hn = harp_noaa.copy()
    if "noaa_ar" not in hn.columns and "noaa_num" in hn.columns:
        hn = hn.rename(columns={"noaa_num": "noaa_ar"})
    if "harpnum" not in hn.columns or "noaa_ar" not in hn.columns:
        raise ValueError("harp-noaa needs columns harpnum, noaa_ar (or noaa_num)")

    harp_to_noaa = hn.drop_duplicates("harpnum").set_index("harpnum")["noaa_ar"].to_dict()
    noaa_ser = pd.to_numeric(f[noaa_col], errors="coerce").fillna(-1).astype(int)

    max_letters: list[str] = []
    near_miss: list[bool] = []
    for _, row in df.iterrows():
        harp = int(row["harpnum"])
        t0 = pd.Timestamp(row["t0"])
        t1 = t0 + pd.Timedelta(hours=int(horizon_hours))
        noaa = harp_to_noaa.get(harp)
        if noaa is None or (isinstance(noaa, float) and np.isnan(noaa)):
            max_letters.append("")
            near_miss.append(False)
            continue
        noaa_i = int(noaa)
        ev = f[(f["start"] >= t0) & (f["start"] < t1) & (noaa_ser == noaa_i)]
        if ev.empty:
            max_letters.append("")
            near_miss.append(False)
            continue
        letters = [x for x in ev["_letter"].tolist() if x]
        if not letters:
            max_letters.append("")
            near_miss.append(False)
            continue
        best = max(letters, key=_class_rank)
        max_letters.append(best)
        has_m = any(_class_rank(L) >= _class_rank("M") for L in letters)
        has_c = any(L == "C" for L in letters)
        near_miss.append(bool(has_c and not has_m))

    out = df.copy()
    out["flare_max_class_letter"] = max_letters
    out["flare_near_miss_c"] = near_miss
    return out


def _summarize_spatial(
    df: pd.DataFrame,
    kind: Literal["FP", "FN", "TP", "TN"],
    limb_deg: float,
) -> dict[str, float | int]:
    sub = df[df["error_kind"] == kind]
    n = len(sub)
    if n == 0 or "cmd_deg" not in sub.columns:
        return {"n": n, "mean_abs_cmd": float("nan"), "frac_abs_cmd_gt_limb": float("nan")}
    cmd = np.abs(pd.to_numeric(sub["cmd_deg"], errors="coerce"))
    finite = cmd[np.isfinite(cmd)]
    if len(finite) == 0:
        return {"n": n, "mean_abs_cmd": float("nan"), "frac_abs_cmd_gt_limb": float("nan")}
    return {
        "n": n,
        "mean_abs_cmd": float(finite.mean()),
        "frac_abs_cmd_gt_limb": float((finite > limb_deg).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc FP/FN analysis vs CMD and flares")
    ap.add_argument("--npz", type=str, required=True, help="test/val NPZ from validate_checkpoint.py")
    ap.add_argument("--windows", type=str, required=True, help="Windows parquet (5716 test rows)")
    ap.add_argument("--horizon-hours", type=int, default=24, choices=(6, 12, 24))
    ap.add_argument(
        "--frames-metadata",
        type=str,
        default="",
        help="Parquet with harpnum, date_obs, cmd_deg for CMD at window time",
    )
    ap.add_argument("--limb-deg", type=float, default=60.0, help="|CMD| threshold for 'near limb'")
    ap.add_argument("--flare-catalog", type=str, default="", help="HEK/GOES flares parquet")
    ap.add_argument("--harp-noaa", type=str, default="", help="Parquet harpnum → noaa_ar")
    ap.add_argument("--out-dir", type=str, default="final_results/methodology/posthoc_errors")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        raise SystemExit(f"NPZ not found: {npz_path}")

    data = np.load(npz_path)
    probs = data["probs"]
    labels = data["labels"]
    thresholds = data["thresholds"]
    horizons = data["horizons"]

    h_arr = np.asarray(horizons).ravel()
    hit = np.where(h_arr == int(args.horizon_hours))[0]
    if len(hit) != 1:
        raise SystemExit(f"horizon {args.horizon_hours} not in NPZ horizons {h_arr.tolist()}")
    hi = int(hit[0])

    win = pd.read_parquet(_REPO / args.windows if not Path(args.windows).is_absolute() else args.windows)
    win["t0"] = pd.to_datetime(win["t0"], utc=True)
    win = win.sort_values("t0", kind="stable").reset_index(drop=True)

    if len(win) != len(probs):
        raise SystemExit(f"Row mismatch: windows={len(win)} probs={len(probs)}")

    thr = float(thresholds[hi])
    y = labels[:, hi].astype(np.float64)
    p = probs[:, hi].astype(np.float64)
    pred = (p >= thr).astype(int)
    y_bin = (y > 0.5).astype(int)

    win["prob"] = p
    win["y_true"] = y_bin
    win["y_pred"] = pred

    tp = (y_bin == 1) & (pred == 1)
    fn = (y_bin == 1) & (pred == 0)
    fp = (y_bin == 0) & (pred == 1)
    tn = (y_bin == 0) & (pred == 0)
    kind = np.full(len(win), "TN", dtype=object)
    kind[tp] = "TP"
    kind[fn] = "FN"
    kind[fp] = "FP"
    win["error_kind"] = kind

    label_col = f"y_geq_M_{int(args.horizon_hours)}h"
    mask_col = f"is_masked_{int(args.horizon_hours)}h"
    cov_col = "obs_coverage"

    out_dir = _REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append(f"npz={npz_path}")
    lines.append(f"horizon_h={args.horizon_hours} idx={hi} threshold={thr:.4f}")
    lines.append(f"n_windows={len(win)}")
    lines.append(f"TP={int(tp.sum())} FN={int(fn.sum())} FP={int(fp.sum())} TN={int(tn.sum())}")
    if fp.sum() > 0:
        far = fp.sum() / max(1, (tp.sum() + fp.sum()))
        lines.append(f"FAR_fp_over_tp_fp={far:.4f}")

    fps = win[win["error_kind"] == "FP"]
    if len(fps) > 0:
        lines.append(
            f"FP prob mean={fps['prob'].mean():.4f} median={fps['prob'].median():.4f} "
            f"p90={fps['prob'].quantile(0.9):.4f}"
        )
        if cov_col in fps.columns:
            lines.append(f"FP obs_coverage mean={fps[cov_col].mean():.4f}")
        if mask_col in fps.columns:
            lines.append(f"FP frac {mask_col}={fps[mask_col].astype(bool).mean():.4f}")

    fns = win[win["error_kind"] == "FN"]
    if len(fns) > 0 and cov_col in fns.columns:
        lines.append(f"FN obs_coverage mean={fns[cov_col].mean():.4f}")

    meta_path = Path(args.frames_metadata) if args.frames_metadata else None
    if meta_path and meta_path.is_file():
        meta = pd.read_parquet(meta_path if meta_path.is_absolute() else _REPO / meta_path)
        win = _attach_cmd_per_window(win, meta)
        for ek in ("FP", "FN", "TP", "TN"):
            s = _summarize_spatial(win, ek, args.limb_deg)
            lines.append(
                f"{ek}: n={s['n']} mean|CMD|={s['mean_abs_cmd']:.2f} "
                f"frac|CMD|>{args.limb_deg}={s['frac_abs_cmd_gt_limb']:.4f}"
            )
        # Quick test: are FN more limb-heavy than TN?
        fn_m = win.loc[win["error_kind"] == "FN", "cmd_deg"]
        tn_m = win.loc[win["error_kind"] == "TN", "cmd_deg"]
        fn_abs = np.abs(pd.to_numeric(fn_m, errors="coerce"))
        tn_abs = np.abs(pd.to_numeric(tn_m, errors="coerce"))
        fn_abs = fn_abs[np.isfinite(fn_abs)]
        tn_abs = tn_abs[np.isfinite(tn_abs)]
        if mannwhitneyu is not None and len(fn_abs) > 10 and len(tn_abs) > 10:
            _, pval = mannwhitneyu(fn_abs, tn_abs, alternative="two-sided")
            lines.append(
                f"MannWhitney |CMD| FN vs TN: p={pval:.4e} "
                f"(FN mean={fn_abs.mean():.2f} TN mean={tn_abs.mean():.2f})"
            )
    else:
        lines.append("CMD: skipped (no --frames-metadata)")

    fc_path = Path(args.flare_catalog) if args.flare_catalog else None
    hn_path = Path(args.harp_noaa) if args.harp_noaa else None
    if fc_path and hn_path and fc_path.is_file() and hn_path.is_file():
        fl = pd.read_parquet(fc_path if fc_path.is_absolute() else _REPO / fc_path)
        hn = pd.read_parquet(hn_path if hn_path.is_absolute() else _REPO / hn_path)
        win = _augment_flares(win, fl, hn, int(args.horizon_hours))
        fp2 = win[win["error_kind"] == "FP"]
        if len(fp2) > 0 and "flare_near_miss_c" in fp2.columns:
            lines.append(
                f"FP frac with C-class (no M+) in horizon={fp2['flare_near_miss_c'].mean():.4f}"
            )
            let = fp2["flare_max_class_letter"].replace("", np.nan).dropna()
            if len(let) > 0:
                lines.append(f"FP flare max-class value_counts top:\n{let.value_counts().head(6).to_string()}")
    elif args.flare_catalog or args.harp_noaa:
        lines.append("Flare augmentation skipped (need both --flare-catalog and --harp-noaa)")

    summary_path = out_dir / f"posthoc_h{args.horizon_hours}h.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = out_dir / f"errors_detail_h{args.horizon_hours}h.csv"
    win.to_csv(csv_path, index=False)

    print("\n".join(lines))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
