#!/usr/bin/env python3
"""
Compute L_data (magnetogram reconstruction error) for the weak-form paper-lock
model and the no-physics baseline, side-by-side, on the held-out test set.

Per test window, we query each model's reconstructed B at the LAST observed
frame timestep on the 128x128 spatial grid, and compare against the ground-truth
input frame. We report median, mean, and percentile statistics across windows.

This addresses the trivial-solution concern: a B≈0 collapsed model would have
small physics residual but ALSO terrible L_data. If the physics-trained model
has comparable L_data to the no-physics baseline AND much smaller physics
residual, the residual reduction is real (not from B-collapse).
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.models.pinn import PINNConfig, HybridPINNModel


def build_frames_tensor(cdir, harpnum, t0_iso, input_hours=48, target_px=128):
    data = np.load(cdir / f"H{harpnum}.npz", allow_pickle=True)
    fa, ts = data["frames"], data["timestamps"]
    tsm = {str(t): i for i, t in enumerate(ts)}
    t0 = pd.Timestamp(t0_iso)
    t_s = t0 - pd.Timedelta(hours=input_hours)
    T = input_hours + 1
    frames = torch.zeros(T, 3, target_px, target_px)
    raw_frames = torch.zeros(T, 3, target_px, target_px)
    observed = np.zeros(T, dtype=bool)
    last_idx = -1
    for ti in range(T):
        t = (t_s + pd.Timedelta(hours=ti)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        if t not in tsm:
            continue
        f = fa[tsm[t]].astype(np.float32)
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=0)
        f = np.nan_to_num(f, nan=0.0, posinf=3.0, neginf=-3.0)
        dr = np.abs(f).max()
        if dr > 10:
            f = f / 2000.0
        elif dr > 0:
            f = f / max(dr, 5.0)
        f = np.clip(f, -1.5, 1.5)
        frames[ti] = torch.from_numpy(f)
        raw_frames[ti] = torch.from_numpy(f)
        observed[ti] = True
        last_idx = ti
    if not observed.any():
        observed[0] = True
        last_idx = 0
    return frames, raw_frames, torch.from_numpy(observed), last_idx, T


def load_model(cfg, ckpt_path, device):
    model = HybridPINNModel(cfg, encoder_in_channels=None).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ema = None
    if "ema_state_dict" in ckpt:
        from src.utils.training_utils import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model, decay=cfg.train.ema_decay)
        try:
            ema.load_state_dict(ckpt["ema_state_dict"])
        except Exception:
            ema = None
    model.eval()
    return model, ema


def query_B(model, ema, frames, obs_mask, t_idx, T, grid=128, device=torch.device("cpu")):
    xs = torch.linspace(-1, 1, grid, device=device)
    ys = torch.linspace(-1, 1, grid, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    t_norm = 2.0 * t_idx / max(T - 1, 1) - 1.0
    coords = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.full((grid * grid,), t_norm, device=device)],
        dim=-1,
    )
    with torch.no_grad():
        ctx = ema.average_parameters() if ema else None
        if ctx:
            ctx.__enter__()
        try:
            L, g = model.encode_frames(frames.to(device), obs_mask.to(device))
            field = model.query_field(coords, L, g)
            B = field.B.detach().cpu().numpy()
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    if B.shape[-1] == 1:
        Bz = B[:, 0].reshape(grid, grid)
        return None, None, Bz
    Bx = B[:, 0].reshape(grid, grid)
    By = B[:, 1].reshape(grid, grid)
    Bz = B[:, 2].reshape(grid, grid)
    return Bx, By, Bz


def per_window_l_data(model, ema, frames, raw_frames, obs, last_idx, T, device):
    """Return L_data (MSE between predicted B and ground-truth at last observed frame)."""
    Bx_p, By_p, Bz_p = query_B(model, ema, frames, obs, last_idx, T, device=device)
    gt = raw_frames[last_idx].numpy()  # [3, 128, 128]
    Bx_gt, By_gt, Bz_gt = gt[0], gt[1], gt[2]
    if Bx_p is None:
        # 1-component model, only Bz
        mse = float(np.mean((Bz_p - Bz_gt) ** 2))
    else:
        mse = float(np.mean((Bx_p - Bx_gt) ** 2 + (By_p - By_gt) ** 2 + (Bz_p - Bz_gt) ** 2) / 3.0)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak-config", default=str(PROJECT_ROOT / "src/configs/flare_pinn_final.yaml"))
    ap.add_argument("--weak-ckpt",   default=str(PROJECT_ROOT / "outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt"))
    ap.add_argument("--baseline-config", default=str(PROJECT_ROOT / "src/configs/benchmark_classifier.yaml"))
    ap.add_argument("--baseline-ckpt",   default=str(PROJECT_ROOT / "outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt"))
    ap.add_argument("--windows", default="data/windows_test_15.parquet")
    ap.add_argument("--cdir",    default="~/flare_data/consolidated")
    ap.add_argument("--device",  default="mps")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out-csv", default="final_results/physics_residuals_full/l_data_comparison.csv")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    cdir = Path(args.cdir).expanduser()
    df = pd.read_parquet(PROJECT_ROOT / args.windows)
    if args.max_windows > 0:
        df = df.head(args.max_windows)
    print(f"N test windows: {len(df)}, device: {device}")

    print("\nLoading weak-form (paper-lock) ...")
    cfg_w = PINNConfig.from_yaml(args.weak_config)
    weak_model, weak_ema = load_model(cfg_w, Path(args.weak_ckpt), device)

    print("Loading no-physics baseline ...")
    cfg_b = PINNConfig.from_yaml(args.baseline_config)
    base_model, base_ema = load_model(cfg_b, Path(args.baseline_ckpt), device)

    rows = []
    skipped = 0
    for i, r in enumerate(df.itertuples(), 1):
        try:
            frames, raw, obs, last_idx, T = build_frames_tensor(cdir, int(r.harpnum), str(r.t0))
            l_w = per_window_l_data(weak_model, weak_ema, frames, raw, obs, last_idx, T, device)
            l_b = per_window_l_data(base_model, base_ema, frames, raw, obs, last_idx, T, device)
            rows.append({
                "harpnum": int(r.harpnum), "t0": str(r.t0),
                "obs_coverage": float(getattr(r, "obs_coverage", 0)),
                "l_data_weak": l_w, "l_data_baseline": l_b,
            })
            if i % 200 == 0 or i == len(df):
                arr_w = np.array([x["l_data_weak"] for x in rows])
                arr_b = np.array([x["l_data_baseline"] for x in rows])
                print(f"  [{i}/{len(df)}] running medians:  weak={np.median(arr_w):.5f}  baseline={np.median(arr_b):.5f}")
        except Exception as e:
            skipped += 1
            if skipped < 5:
                print(f"  SKIP HARP {r.harpnum} t0={r.t0}: {e}")

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["harpnum", "t0", "obs_coverage", "l_data_weak", "l_data_baseline"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out_path}  (n={len(rows)}, skipped={skipped})")

    arr_w = np.array([x["l_data_weak"] for x in rows])
    arr_b = np.array([x["l_data_baseline"] for x in rows])
    print(f"\n=== L_data summary (test set, n={len(rows)}) ===")
    print(f"{'Statistic':>12s}  {'Weak (PINN)':>16s}  {'Baseline (no phys)':>20s}  {'ratio (W/B)':>13s}")
    for stat, fn in [('mean', np.mean), ('median', np.median), ('p25', lambda a: np.percentile(a, 25)),
                     ('p75', lambda a: np.percentile(a, 75)), ('std', np.std)]:
        vw, vb = fn(arr_w), fn(arr_b)
        print(f"{stat:>12s}  {vw:>16.5e}  {vb:>20.5e}  {vw/vb if vb!=0 else float('inf'):>13.3f}")
    print(f"\nWeak L_data median / Baseline L_data median: {np.median(arr_w)/np.median(arr_b):.3f}")
    print(f"  (1.0 = comparable fit; >1.0 = weak-form fits worse; <1.0 = weak-form fits better)")

    del weak_model, base_model
    gc.collect()


if __name__ == "__main__":
    main()
