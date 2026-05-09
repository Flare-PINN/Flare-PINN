#!/usr/bin/env python3
"""
Latent field diagnostics for the inferred PINN transport surrogate u.

For each unique HARP in the test set, picks a representative window (highest
obs_coverage; for flaring HARPs prefers a M+ window when possible), queries the
seed-1234 paper-lock model for both B and u on a 128x128 grid at the final
window timestep, then computes:

  (1) Spatial autocorrelation of u (lag-1 normalized correlation of |u|)
  (2) Pearson r between |u| and |B_z| (flattened)
  (3) Mean |curl(u)| at the polarity inversion line (PIL = top-20% |grad B_z|)

Per-HARP results are written to a CSV. Group statistics (flaring HARPs vs
quiet HARPs) report mean +/- std and the Cohen's d / Mann-Whitney U test on
mean |curl(u)| at PIL between the two groups.
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
            f /= 2000.0
        elif dr > 0:
            f /= max(dr, 5.0)
        f = np.clip(f, -1.5, 1.5)
        frames[ti] = torch.from_numpy(f)
        observed[ti] = True
        last_idx = ti
    if not observed.any():
        observed[0] = True
        last_idx = 0
    return frames, torch.from_numpy(observed), last_idx, T


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


def query_field(model, ema, frames, obs, t_idx, T, grid=128, device=torch.device("cpu")):
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
            L, g = model.encode_frames(frames.to(device), obs.to(device))
            field = model.query_field(coords, L, g)
            B = field.B.detach().cpu().numpy()
            u = field.u.detach().cpu().numpy()
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    if B.shape[-1] == 1:
        Bz = B[:, 0].reshape(grid, grid)
    else:
        Bz = B[:, 2].reshape(grid, grid)
    ux = u[:, 0].reshape(grid, grid)
    uy = u[:, 1].reshape(grid, grid)
    return ux, uy, Bz


def lag1_spatial_autocorr(field):
    """Mean of lag-1 Pearson r in x and y (over the 2D field)."""
    f = field
    # x lag
    a = f[:, :-1].ravel(); b = f[:, 1:].ravel()
    r_x = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
    # y lag
    a = f[:-1, :].ravel(); b = f[1:, :].ravel()
    r_y = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
    return float(np.nanmean([r_x, r_y]))


def curl_z_2d(ux, uy):
    duy_dx = np.gradient(uy, axis=1)
    dux_dy = np.gradient(ux, axis=0)
    return duy_dx - dux_dy


def per_window_metrics(ux, uy, Bz):
    u_mag = np.sqrt(ux * ux + uy * uy)
    abs_Bz = np.abs(Bz)
    autocorr = lag1_spatial_autocorr(u_mag)

    if u_mag.std() > 0 and abs_Bz.std() > 0:
        r_uBz = float(np.corrcoef(u_mag.ravel(), abs_Bz.ravel())[0, 1])
    else:
        r_uBz = float("nan")

    gx = np.gradient(Bz, axis=1); gy = np.gradient(Bz, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    pil_thresh = np.percentile(grad_mag, 80)
    pil_mask = grad_mag >= pil_thresh

    curl_u = curl_z_2d(ux, uy)
    abs_curl = np.abs(curl_u)
    if pil_mask.sum() > 0:
        mean_curl_pil = float(abs_curl[pil_mask].mean())
        mean_curl_off = float(abs_curl[~pil_mask].mean()) if (~pil_mask).any() else float("nan")
    else:
        mean_curl_pil = mean_curl_off = float("nan")

    return dict(
        autocorr_u=autocorr,
        r_u_Bz=r_uBz,
        mean_curl_at_pil=mean_curl_pil,
        mean_curl_off_pil=mean_curl_off,
        pil_pixel_count=int(pil_mask.sum()),
        u_mag_mean=float(u_mag.mean()),
        u_mag_max=float(u_mag.max()),
    )


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def select_representative_windows(df, cdir):
    """One window per HARP. For flaring HARPs prefer a window that flags y_geq_M_24h.
    Among candidates, pick highest obs_coverage."""
    flaring_harps = set(df.loc[df["y_geq_M_24h"] == True, "harpnum"].unique())
    rows = []
    for harp in df["harpnum"].unique():
        sub = df[df["harpnum"] == harp].copy()
        if not (cdir / f"H{harp}.npz").exists():
            continue
        if harp in flaring_harps:
            cand = sub[sub["y_geq_M_24h"] == True]
            if len(cand) == 0:
                cand = sub
        else:
            cand = sub
        cand = cand.sort_values("obs_coverage", ascending=False)
        r = cand.iloc[0]
        rows.append(dict(
            harpnum=int(harp),
            t0=str(r["t0"]),
            label="flaring" if harp in flaring_harps else "quiet",
            obs_coverage=float(r["obs_coverage"]),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "src/configs/flare_pinn_final.yaml"))
    ap.add_argument("--checkpoint", default=str(
        PROJECT_ROOT
        / "outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt"
    ))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cdir", default="~/flare_data/consolidated")
    ap.add_argument("--windows", default="data/windows_test_15.parquet")
    ap.add_argument("--out-dir", default="final_results/methodology/latent_diagnostics")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="If >0, cap the number of windows processed (debug).")
    args = ap.parse_args()

    cdir = Path(args.cdir).expanduser()
    cfg = PINNConfig.from_yaml(args.config)
    device = torch.device(args.device)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROJECT_ROOT / args.windows)
    sel = select_representative_windows(df, cdir)
    print(f"Selected {len(sel)} HARPs ({(sel.label=='flaring').sum()} flaring, "
          f"{(sel.label=='quiet').sum()} quiet)")
    if args.max_windows > 0:
        sel = sel.head(args.max_windows)

    print(f"Loading model from {args.checkpoint} ...")
    model, ema = load_model(cfg, Path(args.checkpoint), device)

    rows = []
    for i, r in enumerate(sel.itertuples(), 1):
        try:
            frames, obs, last_idx, T = build_frames_tensor(cdir, r.harpnum, r.t0)
            ux, uy, Bz = query_field(model, ema, frames, obs, last_idx, T, device=device)
            m = per_window_metrics(ux, uy, Bz)
            m.update(harpnum=r.harpnum, t0=r.t0, label=r.label,
                     obs_coverage=r.obs_coverage)
            rows.append(m)
            if i % 25 == 0 or i == len(sel):
                print(f"  [{i}/{len(sel)}] HARP {r.harpnum} ({r.label}): "
                      f"autocorr={m['autocorr_u']:.3f}, r(|u|,|Bz|)={m['r_u_Bz']:.3f}, "
                      f"curl@PIL={m['mean_curl_at_pil']:.4f}")
        except Exception as e:
            print(f"  SKIP HARP {r.harpnum}: {e}")

    res = pd.DataFrame(rows)
    csv_path = out_dir / "latent_diagnostics_per_harp.csv"
    res.to_csv(csv_path, index=False)
    print(f"\nPer-HARP CSV: {csv_path} ({len(res)} rows)")

    flaring = res[res["label"] == "flaring"]
    quiet = res[res["label"] == "quiet"]

    summary_lines = []
    summary_lines.append(f"n_flaring={len(flaring)}, n_quiet={len(quiet)}, n_total={len(res)}")
    summary_lines.append("")

    def fmt(arr):
        a = np.asarray(arr); a = a[np.isfinite(a)]
        if a.size == 0:
            return "n=0"
        return f"mean={a.mean():.3f}  std={a.std():.3f}  median={np.median(a):.3f}  n={a.size}"

    for col, name in [
        ("autocorr_u", "Spatial autocorrelation of |u| (lag-1)"),
        ("r_u_Bz", "Pearson r between |u| and |B_z|"),
        ("mean_curl_at_pil", "Mean |curl(u)| at PIL"),
        ("mean_curl_off_pil", "Mean |curl(u)| off-PIL"),
    ]:
        summary_lines.append(f"## {name}")
        summary_lines.append(f"  All HARPs:    {fmt(res[col])}")
        summary_lines.append(f"  Flaring:      {fmt(flaring[col])}")
        summary_lines.append(f"  Quiet:        {fmt(quiet[col])}")

        try:
            from scipy.stats import mannwhitneyu, ttest_ind
            a = flaring[col].dropna().values
            b = quiet[col].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
                t_stat, t_p = ttest_ind(a, b, equal_var=False)
                d = cohens_d(a, b)
                summary_lines.append(
                    f"  Effect size flaring vs quiet:  Cohen's d = {d:.3f}  "
                    f"Mann-Whitney p = {u_p:.3e}  Welch's t p = {t_p:.3e}"
                )
        except Exception as e:
            summary_lines.append(f"  effect size error: {e}")
        summary_lines.append("")

    summary_path = out_dir / "latent_diagnostics_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print(f"\nSummary: {summary_path}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
