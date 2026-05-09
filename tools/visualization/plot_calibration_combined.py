#!/usr/bin/env python3
"""
Generate combined reliability diagram comparing 3 models.
3×3 grid: Row 1 = PINN, Row 2 = Strong-Form, Row 3 = Benchmark; Columns = 6h/12h/24h
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import adaptive_ece


def compute_reliability_curve(y_true, y_prob, n_bins=25):
    """Compute reliability curve (calibration curve) using adaptive quantile bins."""
    # Use quantile-based bins like adaptive_ece
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(y_prob, qs)
    edges[0], edges[-1] = 0.0, 1.0
    
    bin_centers = []
    bin_accuracies = []
    
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + 1e-12
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 3:  # Require at least 3 samples per bin (reduced for smoother curves)
            bin_centers.append(y_prob[mask].mean())
            bin_accuracies.append(y_true[mask].mean())
    
    return np.array(bin_centers), np.array(bin_accuracies)


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def plot_reliability_diagram(ax, y_true, y_prob_raw, y_prob_cal, model_name, horizon, base_rate, is_bottom_row):
    """Plot reliability diagram for one model and horizon."""
    
    # Compute curves and ECE (use adaptive_ece to match calibrate_platt.py)
    centers_raw, acc_raw = compute_reliability_curve(y_true, y_prob_raw)
    centers_cal, acc_cal = compute_reliability_curve(y_true, y_prob_cal)
    ece_raw = adaptive_ece(y_true, y_prob_raw, n_bins=15)
    ece_cal = adaptive_ece(y_true, y_prob_cal, n_bins=15)
    
    # Compute Brier scores
    from sklearn.metrics import brier_score_loss
    brier_raw = brier_score_loss(y_true, y_prob_raw)
    brier_cal = brier_score_loss(y_true, y_prob_cal)
    climo_brier = base_rate * (1 - base_rate)
    bss_raw = 1 - brier_raw / climo_brier
    bss_cal = 1 - brier_cal / climo_brier
    
    # Fixed axis limits for all horizons
    x_max = 0.5
    y_max = 0.5
    
    # Plot perfect calibration line
    ax.plot([0, min(x_max, y_max)], [0, min(x_max, y_max)], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
    
    # Plot raw and calibrated
    if len(centers_raw) > 0:
        ax.plot(centers_raw, acc_raw, 'o-', color='#d62728', linewidth=2, 
                markersize=7, alpha=0.8, label=f'Raw (ECE={ece_raw:.3f})')
    
    if len(centers_cal) > 0:
        ax.plot(centers_cal, acc_cal, 's-', color='#2ca02c', linewidth=2, 
                markersize=7, alpha=0.8, label=f'Calibrated (ECE={ece_cal:.3f})')
    
    # Add base rate line
    ax.axhline(base_rate, color='gray', linestyle=':', linewidth=1, alpha=0.5, 
               label=f'Base rate={base_rate:.3f}')
    
    # Formatting
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{model_name} — {horizon}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Only add x-label on bottom row
    if is_bottom_row:
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    
    # Only add y-label on leftmost column
    if horizon == '6h':
        ax.set_ylabel('Observed Frequency', fontsize=10)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate combined calibration figure")
    parser.add_argument("--pinn-raw", type=Path, required=True)
    parser.add_argument("--pinn-cal", type=Path, required=True)
    parser.add_argument("--strong-raw", type=Path, required=True)
    parser.add_argument("--strong-cal", type=Path, required=True)
    parser.add_argument("--benchmark-raw", type=Path, required=True)
    parser.add_argument("--benchmark-cal", type=Path, required=True)
    parser.add_argument("--output", type=Path, 
                       default=Path("final_results/calibration/calibration_combined.png"))
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING COMBINED CALIBRATION FIGURE (3×3)")
    print("=" * 70)
    print()
    
    # Load data
    print("📦 Loading Flare-PINN results...")
    pinn_raw = np.load(args.pinn_raw)
    pinn_cal = np.load(args.pinn_cal)
    
    print("📦 Loading Strong-Form results...")
    strong_raw = np.load(args.strong_raw)
    strong_cal = np.load(args.strong_cal)
    
    print("📦 Loading Benchmark results...")
    bench_raw = np.load(args.benchmark_raw)
    bench_cal = np.load(args.benchmark_cal)
    
    horizons = ['6h', '12h', '24h']
    model_configs = [
        (pinn_raw, pinn_cal, 'Flare-PINN (Weak-Form)', 0),
        (strong_raw, strong_cal, 'Strong-Form', 1),
        (bench_raw, bench_cal, 'Benchmark (No Physics)', 2),
    ]
    
    # Create 3×3 figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for row, (raw_data, cal_data, model_name, row_idx) in enumerate(model_configs):
        for i, horizon in enumerate(horizons):
            y_true = raw_data['labels'][:, i]
            y_prob_raw = raw_data['probs'][:, i]
            y_prob_cal = cal_data['probs_calibrated'][:, i]
            
            valid = np.isfinite(y_true) & np.isfinite(y_prob_raw) & np.isfinite(y_prob_cal)
            base_rate = y_true[valid].mean()
            
            is_bottom_row = (row_idx == 2)
            
            plot_reliability_diagram(axes[row_idx, i], y_true[valid], y_prob_raw[valid], 
                                    y_prob_cal[valid], model_name, horizon, base_rate, 
                                    is_bottom_row)
    
    # Overall title
    fig.suptitle('Probability Calibration: Model Comparison Across Forecast Horizons',
                 fontsize=14, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved combined figure to: {args.output}")
    print("\n Done!")


if __name__ == "__main__":
    main()

