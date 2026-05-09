#!/usr/bin/env python3
"""
Standalone REV figure: three panels (6 h, 12 h, 24 h).

Each panel shows the Relative Economic Value vs Cost/Loss ratio for one
forecast horizon, with the positive-value region shaded and the event
base rate marked.

Defaults to the locked weak-form paper checkpoint, but accepts any test NPZ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import confusion_at_threshold

COLORS = ["#0D47A1", "#1976D2", "#64B5F6"]
FILL_COLORS = ["#0D47A1", "#1976D2", "#64B5F6"]


def compute_rev_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_ratios: int = 600,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    p = y_true.mean()
    tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, threshold)

    pod = tp / max(1, tp + fn)
    pofd = fp / max(1, fp + tn)

    cl_ratios = np.logspace(-4, 0, n_ratios)
    rev_values = np.zeros(n_ratios)

    for i, cl in enumerate(cl_ratios):
        cost_clim = min(cl, p)
        cost_perfect = cl * p
        cost_forecast = cl * pofd * (1.0 - p) + (1.0 - pod) * p
        denom = cost_clim - cost_perfect
        if abs(denom) < 1e-15:
            rev_values[i] = 0.0
        else:
            rev_values[i] = (cost_clim - cost_forecast) / denom

    max_rev = float(np.max(rev_values))
    optimal_cl = float(cl_ratios[np.argmax(rev_values)])
    return cl_ratios, rev_values, max_rev, optimal_cl, p


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate REV curves from a test NPZ.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "outputs"
        / "checkpoints"
        / "weak_form/final"
        / "validation_results"
        / "checkpoint_step_0044000_test.npz",
        help="Test-set NPZ produced by validate_checkpoint.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "final_results" / "figures" / "rev_curve.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    npz_path = args.input
    data = np.load(npz_path)
    probs = data["probs"]
    labels = data["labels"]
    thresholds = data["thresholds"]

    horizons = [6, 12, 24]
    horizon_labels = ["6-hour horizon", "12-hour horizon", "24-hour horizon"]
    panel_tags = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), facecolor="white",
        gridspec_kw={"wspace": 0.30},
    )

    for hi, (ax, horizon, hlabel, tag, color, fill) in enumerate(
        zip(axes, horizons, horizon_labels, panel_tags, COLORS, FILL_COLORS)
    ):
        y_true = labels[:, hi].astype(np.float64)
        y_prob = probs[:, hi].astype(np.float64)
        thr = float(thresholds[hi])

        valid = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true, y_prob = y_true[valid], np.clip(y_prob[valid], 0.0, 1.0)

        cl_ratios, rev_values, max_rev, optimal_cl, base_rate = compute_rev_curve(
            y_true, y_prob, thr
        )

        # Shade positive-value region
        pos_mask = rev_values > 0
        if pos_mask.any():
            ax.fill_between(
                cl_ratios, 0, rev_values,
                where=pos_mask,
                color=fill, alpha=0.12, zorder=1,
            )

        # REV curve
        ax.plot(
            cl_ratios, rev_values,
            color=color, linewidth=2.5, zorder=3,
        )

        # Peak marker + label
        ax.plot(
            optimal_cl, max_rev, "o",
            color=color, markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5,
        )
        ax.text(
            optimal_cl * 3.5, max_rev - 0.06,
            f"peak = {max_rev:.2f}",
            fontsize=9, color=color, fontweight="bold",
        )

        # Zero line
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        ax.set_xscale("log")
        ax.set_xlim([1e-4, 1.0])
        ax.set_ylim([-0.08, 1.0])
        ax.set_xlabel("Cost / Loss ratio  ($C/L$)", fontsize=10)
        if hi == 0:
            ax.set_ylabel("Relative Economic Value", fontsize=10)
        ax.set_title(f"{tag}  {hlabel}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.15, which="both")
        ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.90, wspace=0.28)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()

    import csv
    summary_rows = []
    for hi, (horizon, hlabel) in enumerate(zip(horizons, horizon_labels)):
        y_true = labels[:, hi].astype(np.float64)
        y_prob = probs[:, hi].astype(np.float64)
        thr = float(thresholds[hi])
        valid = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[valid]
        y_prob_v = y_prob[valid]
        cl_ratios, rev_values, max_rev, optimal_cl, base_rate = compute_rev_curve(y_true, y_prob_v, thr)

        rev_at_001 = float(np.interp(0.01, cl_ratios, rev_values))
        pos_mask = rev_values > 0
        if pos_mask.any():
            pos_cl = cl_ratios[pos_mask]
            pos_lower = float(pos_cl.min())
            pos_upper = float(pos_cl.max())
        else:
            pos_lower = float("nan")
            pos_upper = float("nan")
        print(
            f"  {hlabel}: peak={max_rev:.3f} @ C/L={optimal_cl:.4f}, "
            f"REV@0.01={rev_at_001:.3f}, positive_range=[{pos_lower:.4f},{pos_upper:.4f}], "
            f"base={base_rate:.5f}"
        )
        summary_rows.append(
            dict(
                horizon=f"{horizon}h",
                peak_rev=max_rev,
                optimal_cl=optimal_cl,
                rev_at_cl_0p01=rev_at_001,
                positive_cl_lower=pos_lower,
                positive_cl_upper=pos_upper,
                base_rate=base_rate,
                threshold=thr,
            )
        )

    csv_out = args.output.with_name(args.output.stem + "_summary.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved summary CSV: {csv_out}")


if __name__ == "__main__":
    main()
