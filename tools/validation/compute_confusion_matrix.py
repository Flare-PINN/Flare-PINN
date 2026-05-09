#!/usr/bin/env python3
"""
Compute confusion matrix metrics from saved validation results.

This script loads the validation predictions and optimal thresholds
computed by validate_checkpoint.py and generates:
  1. Confusion matrix metrics (TSS, HSS, POD, FAR, CSI, Accuracy, TP/FP/TN/FN)
  2. Confusion matrix plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def tss_from_cm(tp: int, fn: int, fp: int, tn: int) -> float:
    """Compute TSS from confusion matrix."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity - (1.0 - specificity)


def hss_from_cm(tp: int, fn: int, fp: int, tn: int) -> float:
    """Compute Heidke Skill Score from confusion matrix."""
    n = tp + fn + fp + tn
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
    actual = tp + tn
    return (actual - expected) / (n - expected) if (n - expected) > 0 else 0.0


def compute_cm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all confusion matrix metrics."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    
    total = tp + fn + fp + tn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # POD
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # ROC false positive rate
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    tss = sensitivity - (1.0 - specificity)
    hss = hss_from_cm(tp, fn, fp, tn)
    
    return {
        'TSS': tss,
        'HSS': hss,
        'POD': sensitivity,
        'FPR': fpr,
        'FAR': far,
        'CSI': csi,
        'Accuracy': accuracy,
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn
    }


def plot_confusion_matrix(cm: np.ndarray, horizon: int, tss: float, threshold: float, 
                         output_path: Path, model_name: str = "Model"):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=['No Flare', 'Flare'],
                yticklabels=['No Flare', 'Flare'],
                ax=ax, cbar_kws={'label': 'Normalized Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{model_name} - {horizon}h Horizon\nTSS = {tss:.4f} @ threshold = {threshold:.2f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute confusion matrix from validation results")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to .npz file from validate_checkpoint.py",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        dest="input_legacy",
        help="Deprecated alias for --input (same meaning).",
    )
    parser.add_argument("--model-name", type=str, default="Model",
                       help="Model name for plots and CSV")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as input)")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write only the CSV metrics; do not emit PNG confusion matrices.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="CSV filename (default: cnn_baseline_* or pinn_* heuristic from --model-name)",
    )
    args = parser.parse_args()
    npz_path = args.input or args.input_legacy
    if not npz_path:
        parser.error("Pass --input PATH.npz (or deprecated --results).")
    
    # Load validation results
    input_path = Path(npz_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading validation results from: {input_path}")
    data = np.load(input_path)
    
    probs = data['probs']  # [N, 3]
    labels = data['labels']  # [N, 3]
    horizons = data['horizons']  # [3]
    thresholds = data['thresholds']  # [3]
    tss_values = data['tss_values']  # [3]
    
    print(f"Loaded {len(labels)} samples, {len(horizons)} horizons")
    print(f"Horizons: {horizons}")
    print(f"Optimal thresholds: {thresholds}")
    print(f"TSS values: {tss_values}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(exist_ok=True)
    
    # Compute metrics for each horizon
    all_metrics = []
    
    print(f"\n{'='*60}")
    print("Computing Confusion Matrix Metrics")
    print(f"{'='*60}\n")
    
    for i, horizon in enumerate(horizons):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        threshold = thresholds[i]
        tss_val = tss_values[i]
        
        # Apply threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Compute metrics
        metrics = compute_cm_metrics(y_true, y_pred)
        
        print(f"{int(horizon)}h Horizon (threshold={threshold:.2f}):")
        print(f"  TSS: {metrics['TSS']:.4f} (expected: {tss_val:.4f})")
        print(f"  HSS: {metrics['HSS']:.4f}")
        print(f"  POD: {metrics['POD']:.4f}")
        print(f"  FPR: {metrics['FPR']:.4f}")
        print(f"  FAR: {metrics['FAR']:.4f}")
        print(f"  CSI: {metrics['CSI']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  TP={metrics['TP']}, FN={metrics['FN']}, FP={metrics['FP']}, TN={metrics['TN']}")
        
        # Verify TSS matches
        tss_diff = abs(metrics['TSS'] - tss_val)
        if tss_diff > 0.001:
            print(f"    WARNING: TSS mismatch! Difference: {tss_diff:.6f}")
        else:
            print(f"   TSS matches expected value!")
        print()
        
        # Save for CSV
        all_metrics.append({
            'Model': args.model_name,
            'Horizon': f"{int(horizon)}h",
            'Threshold': threshold,
            'TSS': metrics['TSS'],
            'HSS': metrics['HSS'],
            'POD': metrics['POD'],
            'FPR': metrics['FPR'],
            'FAR': metrics['FAR'],
            'CSI': metrics['CSI'],
            'Accuracy': metrics['Accuracy'],
            'TP': metrics['TP'],
            'FN': metrics['FN'],
            'FP': metrics['FP'],
            'TN': metrics['TN']
        })
        
        # Create confusion matrix and plot
        cm = np.array([[metrics['TN'], metrics['FP']], 
                       [metrics['FN'], metrics['TP']]])
        
        # Determine model suffix for paper figure assembly.
        model_name_lower = args.model_name.lower()
        if "strong" in model_name_lower:
            model_suffix = "Strong"
        elif "baseline" in model_name_lower or "benchmark" in model_name_lower or "cnn" in model_name_lower:
            model_suffix = "Base"
        else:
            model_suffix = "PINN"
        if not args.skip_plots:
            plot_path = output_dir / f"{int(horizon)}h{model_suffix}.png"
            plot_confusion_matrix(cm, int(horizon), metrics['TSS'], threshold, 
                                plot_path, args.model_name)
    
    # Save metrics to CSV with correct naming
    df = pd.DataFrame(all_metrics)
    # Determine CSV prefix based on model name
    if args.csv_name:
        csv_path = output_dir / args.csv_name
    else:
        if "baseline" in args.model_name.lower() or "cnn" in args.model_name.lower():
            csv_prefix = "cnn_baseline"
        else:
            csv_prefix = "pinn_46k"
        csv_path = output_dir / f"{csv_prefix}_confusion_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f" Complete!")
    print(f"{'='*60}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
