#!/usr/bin/env python3
"""
Compute real per-event lead times for every model in Table 3.

For each true-positive prediction (label=1 AND model probability >= D2C
threshold), look up the actual M/X-class flare(s) within [t0, t0+horizon]
from the SC24 GOES flare catalog and report:

    lead_time_hours = (first matching flare start time) - (window t0)

Aggregates median, mean, and IQR across all true positives, per model and
per horizon. Writes one CSV row per (model, horizon).

Required inputs:
  - data/windows_test_15.parquet                        (test window metadata)
  - data/harp_noaa_mapping.parquet                      (HARP -> NOAA AR mapping)
  - SC24 M/X flare catalog with `start`, `letter`, `noaa_ar` columns
  - One test NPZ per model with `probs`, `labels`, `thresholds` arrays

Usage:
    python tools/analysis/compute_real_lead_times.py \\
        --catalog /path/to/flares_hek_mx.parquet \\
        --out final_results/methodology/real_lead_times_all_models.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Models evaluated in Table 3 (model name -> path to test NPZ).
DEFAULT_MODELS = {
    "Flare-PINN (Weak)":   "outputs/checkpoints/weak_form/final/validation_results/checkpoint_step_0044000_test.npz",
    "Strong-Form":         "outputs/checkpoints/strong_form/final/validation_results/checkpoint_step_0044000_test.npz",
    "Benchmark (CNN)":     "outputs/checkpoints/benchmark_classifier/validation_results/checkpoint_step_0040000_test.npz",
    "Logistic Regression": "outputs/classical_baselines/logistic_regression_test.npz",
    "XGBoost":             "outputs/classical_baselines/xgboost_test.npz",
    "SVM":                 "outputs/classical_baselines/svm_test.npz",
}
HORIZONS = [6, 12, 24]


def load_shared_assets(catalog_path: Path):
    win = pd.read_parquet(PROJECT_ROOT / "data/windows_test_15.parquet")
    win["t0"] = pd.to_datetime(win["t0"], utc=True)
    win = win.sort_values("t0", kind="stable").reset_index(drop=True)

    cat = pd.read_parquet(catalog_path)
    cat["start"] = pd.to_datetime(cat["start"], utc=True)
    cat = cat[cat["letter"].isin(["M", "X"])].reset_index(drop=True)
    cat["noaa_ar"] = pd.to_numeric(cat["noaa_ar"], errors="coerce").astype("Int64")
    cat_by_noaa = {
        int(noaa): grp.sort_values("start")
        for noaa, grp in cat.dropna(subset=["noaa_ar"]).groupby("noaa_ar")
    }

    hn = pd.read_parquet(PROJECT_ROOT / "data/harp_noaa_mapping.parquet")
    hn["harpnum"] = hn["harpnum"].astype(int)
    hn["noaa_ar"] = pd.to_numeric(hn["noaa_ar"], errors="coerce").astype("Int64")
    harp_to_noaa = (
        hn.dropna()
        .groupby("harpnum")["noaa_ar"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )
    return win, cat_by_noaa, harp_to_noaa


def lead_times_for_model(probs, labels, thresholds, horizon_idx, horizon, win, cat_by_noaa, harp_to_noaa):
    thr = float(thresholds[horizon_idx])
    y_true = labels[:, horizon_idx].astype(int)
    y_pred = (probs[:, horizon_idx] >= thr).astype(int)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_rows = win[tp_mask]

    leads = []
    for _, r in tp_rows.iterrows():
        harp = int(r["harpnum"])
        t0 = r["t0"]
        t_end = t0 + pd.Timedelta(hours=int(horizon))
        for noaa in harp_to_noaa.get(harp, set()):
            grp = cat_by_noaa.get(int(noaa))
            if grp is None:
                continue
            hits = grp[(grp["start"] >= t0) & (grp["start"] <= t_end)]
            if len(hits) > 0:
                leads.append((hits["start"].min() - t0).total_seconds() / 3600.0)
                break
    return np.array(leads), int(tp_mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        type=str,
        default="/Users/akshgarg/Downloads/flare-pinn-personal-keep-2026-04-24/data/defn/flares_hek_mx.parquet",
        help="Path to M/X SC24 flare catalog parquet (cols: start, letter, noaa_ar).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="final_results/methodology/real_lead_times_all_models.csv",
    )
    args = ap.parse_args()

    win, cat_by_noaa, harp_to_noaa = load_shared_assets(Path(args.catalog))
    print(f"Test windows: {len(win)},  flare catalog NOAA ARs: {len(cat_by_noaa)}")

    print(f"\n{'='*84}")
    print("Median lead time (hours) per model x horizon")
    print(f"{'='*84}")
    print(f"{'Model':>22s}  {'6h':>14s}  {'12h':>14s}  {'24h':>14s}")
    print("-" * 84)

    out_rows = []
    for name, p in DEFAULT_MODELS.items():
        full = PROJECT_ROOT / p
        if not full.exists():
            print(f"  WARN: missing {p} — skipping {name}")
            continue
        d = np.load(full)
        cells = []
        for hi, h in enumerate(HORIZONS):
            leads, n_tp = lead_times_for_model(
                d["probs"], d["labels"], d["thresholds"], hi, h,
                win, cat_by_noaa, harp_to_noaa,
            )
            med = float(np.median(leads)) if len(leads) > 0 else float("nan")
            cells.append(f"{med:5.2f}h (n={len(leads):>3d})")
            out_rows.append({
                "model": name,
                "horizon": h,
                "n_TP": n_tp,
                "n_matched": len(leads),
                "median_lead_h": med,
                "mean_lead_h": float(np.mean(leads)) if len(leads) > 0 else float("nan"),
                "p25_lead_h": float(np.percentile(leads, 25)) if len(leads) > 0 else float("nan"),
                "p75_lead_h": float(np.percentile(leads, 75)) if len(leads) > 0 else float("nan"),
            })
        print(f"{name:>22s}  " + "  ".join(f"{c:>14s}" for c in cells))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
