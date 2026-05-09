#!/usr/bin/env python3
"""
Fig 6: Physics-derived diagnostics from Flare-PINN's reconstructed magnetic field.

Loads the seed-1234 paper-lock checkpoint, queries the PINN's implicit neural
field for B at the final frame on a 128x128 spatial grid, computes physical
diagnostics (eta|J|^2, J_z, |J x B|), and plots a 3x2 panel comparing a
flaring active region vs a quiet one.

Defaults to HARP 3721 (flaring; 24h before first M-class+ event in window
2014-02-10T06:00) vs HARP 4683 (quiet; high-coverage window 2014-10-15T07:00).
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
    last_observed_idx = -1
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
        last_observed_idx = ti
    if not observed.any():
        observed[0] = True
        last_observed_idx = 0
    return frames, torch.from_numpy(observed), last_observed_idx, T


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


def query_B_field(model, ema, frames, obs_mask, t_idx, T, grid=128, device=torch.device("cpu")):
    """Return reconstructed Bx, By, Bz on a grid x grid spatial grid at t_idx."""
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
            B = field.B.detach().cpu().numpy()  # [N, C]
            u = field.u.detach().cpu().numpy()  # [N, 2]
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    if B.shape[-1] == 1:
        Bz = B[:, 0].reshape(grid, grid)
        Bx = By = np.zeros_like(Bz)
    else:
        Bx = B[:, 0].reshape(grid, grid)
        By = B[:, 1].reshape(grid, grid)
        Bz = B[:, 2].reshape(grid, grid)
    ux = u[:, 0].reshape(grid, grid)
    uy = u[:, 1].reshape(grid, grid)
    return Bx, By, Bz, ux, uy


def compute_physics(Bx, By, Bz, eta=0.01):
    Jx = np.gradient(Bz, axis=0)
    Jy = -np.gradient(Bz, axis=1)
    dBy_dx = np.gradient(By, axis=1)
    dBx_dy = np.gradient(Bx, axis=0)
    Jz = dBy_dx - dBx_dy
    J_mag2 = Jx * Jx + Jy * Jy + Jz * Jz
    Q = eta * J_mag2  # ohmic heating density
    Fx = Jy * Bz - Jz * By
    Fy = Jz * Bx - Jx * Bz
    Fz = Jx * By - Jy * Bx
    F_mag = np.sqrt(Fx * Fx + Fy * Fy + Fz * Fz)
    return dict(Jx=Jx, Jy=Jy, Jz=Jz, J_mag=np.sqrt(J_mag2), Q=Q, F_mag=F_mag)


def draw_quiver_original(ax, vx, vy, color):
    """Quiver overlay copied from the original synthetic 3-panel script."""
    H, W = vx.shape
    step = 8
    Y_plot = np.linspace(-1, 1, H)
    X_plot, Y_plot = np.meshgrid(np.linspace(-1, 1, W), Y_plot)
    x_vec = X_plot[::step, ::step]
    y_vec = Y_plot[::step, ::step]
    vx_plot = vx[::step, ::step]
    vy_plot = vy[::step, ::step]
    v_norm = np.sqrt(vx_plot ** 2 + vy_plot ** 2) + 1e-6
    ax.quiver(
        x_vec, y_vec, vx_plot / v_norm * 0.08, vy_plot / v_norm * 0.08,
        color=color, alpha=0.7, width=0.003,
        headwidth=5, headlength=6, scale=1, scale_units="xy",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "src/configs/flare_pinn_final.yaml"))
    ap.add_argument("--checkpoint", default=str(
        PROJECT_ROOT
        / "outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt"
    ))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cdir", default="~/flare_data/consolidated")
    ap.add_argument("--harp-flaring", type=int, default=4698)
    ap.add_argument("--t0-flaring", default="2014-10-19T10:00:00+00:00",
                    help="Window start; HARP 4698 has 38 M-class+ windows in 24h horizon.")
    ap.add_argument("--harp-quiet", type=int, default=321)
    ap.add_argument("--t0-quiet", default="2011-01-03T00:00:00+00:00")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "final_results/figures/physics_comparison_combined.png"
    ))
    ap.add_argument("--eta", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = PINNConfig.from_yaml(args.config)
    cdir = Path(args.cdir).expanduser()

    print(f"Loading model from {args.checkpoint} ...")
    model, ema = load_model(cfg, Path(args.checkpoint), device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        dict(label="Flaring Active Region",
             harp=args.harp_flaring, t0=args.t0_flaring),
        dict(label="Stable Active Region",
             harp=args.harp_quiet, t0=args.t0_quiet),
    ]

    fields = []
    for c in cases:
        print(f"\nHARP {c['harp']} ({c['label']}) at t0={c['t0']}")
        frames, obs, last_idx, T = build_frames_tensor(cdir, c["harp"], c["t0"])
        print(f"  observed frames: {int(obs.sum())}/{T}, last_obs_idx={last_idx}")
        Bx, By, Bz, ux, uy = query_B_field(model, ema, frames, obs, last_idx, T,
                                            grid=128, device=device)
        phys = compute_physics(Bx, By, Bz, eta=args.eta)
        fields.append((c, Bx, By, Bz, ux, uy, phys))
        print(f"  |B| range: [{np.abs(Bz).min():.3f}, {np.abs(Bz).max():.3f}]"
              f"  |J| max: {phys['J_mag'].max():.3f}"
              f"  Q max: {phys['Q'].max():.3e}"
              f"  |JxB| max: {phys['F_mag'].max():.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # Direct port of the original `generate_physics_comparisons_3panel.py`
    # styling, composed as 3 rows (ohmic / Jz / Lorentz) x 2 cols (HARP).
    # ─────────────────────────────────────────────────────────────────────

    Q_f, Q_n = fields[0][6]["Q"], fields[1][6]["Q"]
    Jz_f, Jz_n = fields[0][6]["Jz"], fields[1][6]["Jz"]
    F_f, F_n = fields[0][6]["F_mag"], fields[1][6]["F_mag"]
    Jx_f, Jy_f = fields[0][6]["Jx"], fields[0][6]["Jy"]
    Jx_n, Jy_n = fields[1][6]["Jx"], fields[1][6]["Jy"]

    # vmin/vmax exactly as in the synthetic-original code:
    #   non-symmetric → shared min/max across panels, then * scale_factor
    #   symmetric    → ±max(|.|), full range
    Q_vmin, Q_vmax = min(Q_f.min(), Q_n.min()), max(Q_f.max(), Q_n.max())
    Q_vmax = Q_vmin + (Q_vmax - Q_vmin) * 0.6   # scale_factor 0.6 (original)

    Jz_vmax = max(np.abs(Jz_f).max(), np.abs(Jz_n).max())
    Jz_vmin = -Jz_vmax

    F_vmin, F_vmax = min(F_f.min(), F_n.min()), max(F_f.max(), F_n.max())
    F_vmax = F_vmin + (F_vmax - F_vmin) * 1.0   # scale_factor 1.0 for lorentz (original)

    # 3 rows x 2 cols, with one shared colorbar per row (extra column).
    fig = plt.figure(figsize=(11, 14.5), dpi=300)
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.02, hspace=0.16,
        left=0.07, right=0.93, top=0.92, bottom=0.05,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])
    cbar_axes = [fig.add_subplot(gs[r, 2]) for r in range(3)]

    rows = [
        dict(name="ohmic", panels=[Q_f, Q_n], cmap="inferno",
             vmin=Q_vmin, vmax=Q_vmax,
             label="Heating Rate (η J²)",
             title="Ohmic Heating: Energy Dissipation",
             show_arrows=False, vec_color="cyan",
             vec=[(Jx_f, Jy_f), (Jx_n, Jy_n)]),
        dict(name="current", panels=[Jz_f, Jz_n], cmap="RdBu_r",
             vmin=Jz_vmin, vmax=Jz_vmax,
             label="Jz (vertical current)",
             title="Current Density (J = ∇×B)",
             show_arrows=True, vec_color="black",
             vec=[(Jx_f, Jy_f), (Jx_n, Jy_n)]),
        dict(name="lorentz", panels=[F_f, F_n], cmap="plasma",
             vmin=F_vmin, vmax=F_vmax,
             label="Force |J×B|",
             title="Lorentz Force (Magnetic Stress)",
             show_arrows=False, vec_color="lime",
             vec=[(Jx_f, Jy_f), (Jx_n, Jy_n)]),
    ]

    panel_tags = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"]]
    panel_subs = [
        f"Flaring Active Region\nHARP {fields[0][0]['harp']}",
        f"Stable Active Region\nHARP {fields[1][0]['harp']}",
    ]

    for r, row in enumerate(rows):
        last_im = None
        for col in range(2):
            ax = axes[r, col]
            field = row["panels"][col]
            im = ax.imshow(
                field, extent=[-1, 1, -1, 1], origin="lower",
                cmap=row["cmap"], vmin=row["vmin"], vmax=row["vmax"],
                interpolation="bilinear", aspect="equal",
            )
            last_im = im

            if row["show_arrows"]:
                vx, vy = row["vec"][col]
                draw_quiver_original(ax, vx, vy, row["vec_color"])

            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=8.5, length=3, width=0.6)
            for s in ax.spines.values():
                s.set_linewidth(0.8)

            # x label on bottom row only.
            if r == 2:
                ax.set_xlabel("x (normalized)", fontsize=10.5,
                              fontweight="bold", labelpad=2)
            else:
                ax.set_xticklabels([])

            # y label on left column only.
            if col == 0:
                ax.set_ylabel("y (normalized)", fontsize=10.5,
                              fontweight="bold", labelpad=2)
            else:
                ax.set_yticklabels([])

            ax.text(
                0.025, 0.03, panel_tags[r][col], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.85, edgecolor="none"),
            )

            if r == 0:
                ax.set_title(panel_subs[col], fontsize=11,
                             fontweight="bold", pad=4)

        # Single shared colorbar per row.
        cbar = fig.colorbar(last_im, cax=cbar_axes[r])
        cbar.set_label(row["label"], fontsize=10, rotation=270, labelpad=14)
        cbar.ax.tick_params(labelsize=8, length=2.5, width=0.5)
        cbar.outline.set_linewidth(0.6)

    fig.suptitle(
        "PINN-Computed Physics Comparison: Flaring vs Stable Active Regions",
        fontsize=14, fontweight="bold", y=0.97,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                 facecolor="white", pad_inches=0.05)
    print(f"\nSaved: {out_path}")
    plt.close()
    del model
    gc.collect()


if __name__ == "__main__":
    main()
