#!/usr/bin/env python3
"""
Generate ROC/PR/TSS comparison with all available models.

Defaults to the locked weak-form paper checkpoint and the canonical strong/baseline
artifacts, but accepts explicit NPZ paths so paper figures do not depend on stale
hardcoded locations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import (
    confusion_at_threshold,
    tss_at_threshold,
    precision_recall_curve
)


def compute_roc_curve(y_true, y_prob, n=512):
    """Compute ROC curve (TPR vs FPR) with denser sampling."""
    # Add tiny jitter to break ties (common for SVM with many 0.0 predictions)
    y_prob_jitter = y_prob + np.random.RandomState(42).uniform(-1e-7, 1e-7, size=len(y_prob))
    y_prob_jitter = np.clip(y_prob_jitter, 0.0, 1.0)
    
    thresholds = np.linspace(0, 1, n)[::-1]
    tpr_list, fpr_list = [], []
    
    for thr in thresholds:
        tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob_jitter, thr)
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    
    # CRITICAL FIX: Ensure ROC starts at (0, 0) and ends at (1, 1)
    # Prepend (0, 0) if not already there
    if fpr_arr[0] > 0 or tpr_arr[0] > 0:
        fpr_arr = np.concatenate([[0.0], fpr_arr])
        tpr_arr = np.concatenate([[0.0], tpr_arr])
    
    # Append (1, 1) if not already there
    if fpr_arr[-1] < 1 or tpr_arr[-1] < 1:
        fpr_arr = np.concatenate([fpr_arr, [1.0]])
        tpr_arr = np.concatenate([tpr_arr, [1.0]])
    
    return fpr_arr, tpr_arr, thresholds


def compute_pr_curve(y_true, y_prob, n=256):
    """Compute Precision-Recall curve."""
    recall, precision, thresholds = precision_recall_curve(y_true, y_prob, n=n)
    return recall, precision, thresholds


def compute_tss_curve(y_true, y_prob, n=256):
    """Compute TSS vs threshold curve."""
    thresholds = np.linspace(0, 1, n)
    tss_list = []
    
    for thr in thresholds:
        tss = tss_at_threshold(y_true, y_prob, thr)
        tss_list.append(tss)
    
    return thresholds, np.array(tss_list)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate ROC/PR/TSS comparison figure.")
    parser.add_argument(
        "--pinn",
        type=Path,
        default=project_root
        / "outputs"
        / "checkpoints"
        / "weak_form/final"
        / "validation_results"
        / "checkpoint_step_0044000_test.npz",
    )
    parser.add_argument(
        "--strong",
        type=Path,
        default=project_root
        / "outputs"
        / "checkpoints"
        / "Strong Form Pinn Final 44k"
        / "validation_results"
        / "checkpoint_step_0044000_test.npz",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project_root
        / "outputs"
        / "checkpoints"
        / "benchmark_classifier"
        / "validation_results"
        / "checkpoint_step_0040000_test.npz",
    )
    parser.add_argument(
        "--logreg",
        type=Path,
        default=project_root / "outputs" / "classical_baselines" / "logistic_regression_test.npz",
    )
    parser.add_argument(
        "--svm",
        type=Path,
        default=project_root / "outputs" / "classical_baselines" / "svm_test.npz",
    )
    parser.add_argument(
        "--xgboost",
        type=Path,
        default=project_root / "outputs" / "classical_baselines" / "xgboost_test.npz",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Include XGBoost in the plot. Off by default to keep the legend readable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "final_results" / "roc_tss" / "roc_pr_tss_comparison.png",
    )
    args = parser.parse_args()

    print("Loading predictions...")
    
    pinn_path = args.pinn
    strong_path = args.strong
    baseline_path = args.baseline
    lr_path = args.logreg
    svm_path = args.svm
    xgb_path = args.xgboost
    
    # Check which files exist
    models = []
    
    if pinn_path.exists():
        models.append(("Flare-PINN", np.load(pinn_path), '#0D47A1', 3.5, '-'))
        print(f"  ✓ Loaded Flare-PINN")
    
    if strong_path.exists():
        models.append(("Strong-Form", np.load(strong_path), '#1976D2', 3.0, '-'))
        print(f"  ✓ Loaded Strong-Form")
    
    if baseline_path.exists():
        models.append(("CNN Baseline", np.load(baseline_path), '#64B5F6', 2.5, '-'))
        print(f"  ✓ Loaded CNN Baseline")
    
    if lr_path.exists():
        models.append(("Logistic Reg.", np.load(lr_path), '#C62828', 2.5, '-'))
        print(f"  ✓ Loaded Logistic Regression")
    
    if svm_path.exists():
        models.append(("SVM", np.load(svm_path), '#EF6C00', 2.5, '-'))
        print(f"  ✓ Loaded SVM")
    
    if args.include_xgboost and xgb_path.exists():
        models.append(("XGBoost", np.load(xgb_path), '#2E7D32', 2.5, '-'))
        print(f"  ✓ Loaded XGBoost")
    
    if len(models) < 2:
        print("\n Need at least 2 models to compare!")
        print("Make sure prediction files exist in outputs/test_predictions/ and outputs/classical_baselines/")
        return
    
    horizons = [6, 12, 24]
    horizon_names = ['6h', '12h', '24h']
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25,
                  left=0.08, right=0.98, top=0.92, bottom=0.06)
    
    row_titles = ['ROC Curves', 'Precision-Recall Curves', 'TSS vs Threshold']
    
    print("\nGenerating plots...")
    
    for col_idx, (horizon, horizon_name) in enumerate(zip(horizons, horizon_names)):
        print(f"\n{horizon_name} Horizon:")
        
        ax_roc = fig.add_subplot(gs[0, col_idx])
        ax_pr = fig.add_subplot(gs[1, col_idx])
        ax_tss = fig.add_subplot(gs[2, col_idx])
        
        base_rate = None
        
        for model_name, data, color, linewidth, linestyle in models:
            y_true = data['labels'][:, col_idx]
            y_prob = data['probs'][:, col_idx]
            
            # Remove NaN/Inf
            valid = np.isfinite(y_true) & np.isfinite(y_prob)
            y_true = y_true[valid]
            y_prob = np.clip(y_prob[valid], 0.0, 1.0)
            
            if base_rate is None:
                base_rate = y_true.mean()
            
            # Compute curves (use denser sampling for smoother curves)
            fpr, tpr, _ = compute_roc_curve(y_true, y_prob, n=512)
            recall, precision, _ = compute_pr_curve(y_true, y_prob, n=512)
            tss_thrs, tss_vals = compute_tss_curve(y_true, y_prob, n=512)
            
            # Compute AUC
            idx_roc = np.argsort(fpr)
            roc_auc = float(np.trapz(tpr[idx_roc], fpr[idx_roc]))
            
            # Compute PR-AUC
            idx_pr = np.argsort(recall)
            pr_auc = float(np.trapz(precision[idx_pr], recall[idx_pr]))
            
            # Compute max TSS
            max_tss = tss_vals.max()
            
            print(f"  {model_name}: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, Max TSS={max_tss:.3f}")
            
            # Plot ROC
            ax_roc.plot(fpr, tpr, label=f'{model_name} ({roc_auc:.3f})',
                       color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.9)
            
            # Plot PR
            ax_pr.plot(recall, precision, label=f'{model_name} ({pr_auc:.3f})',
                      color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.9)
            
            # Plot TSS vs Threshold
            ax_tss.plot(tss_thrs, tss_vals, label=f'{model_name} ({max_tss:.3f})',
                       color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.9)
        
        # ROC formatting
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax_roc.set_title(f'{horizon_name} Horizon', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.95, ncol=1)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        
        if col_idx == 0:
            ax_roc.text(-0.25, 0.5, row_titles[0], transform=ax_roc.transAxes,
                       fontsize=15, fontweight='bold', va='center', rotation=90)
        
        # PR formatting
        ax_pr.axhline(base_rate, color='gray', linestyle=':', linewidth=1.5,
                     alpha=0.5, label=f'Base ({base_rate:.3f})')
        ax_pr.set_xlabel('Recall (POD)', fontsize=12, fontweight='bold')
        ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax_pr.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=1)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.02])
        
        if col_idx == 0:
            ax_pr.text(-0.25, 0.5, row_titles[1], transform=ax_pr.transAxes,
                      fontsize=15, fontweight='bold', va='center', rotation=90)
        
        # TSS formatting
        ax_tss.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_tss.set_xlabel('Probability Threshold', fontsize=12, fontweight='bold')
        ax_tss.set_ylabel('True Skill Statistic', fontsize=12, fontweight='bold')
        ax_tss.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=1)
        ax_tss.grid(True, alpha=0.3)
        ax_tss.set_xlim([-0.02, 1.02])
        ax_tss.set_ylim([0.6, 1.0])
        
        if col_idx == 0:
            ax_tss.text(-0.25, 0.5, row_titles[2], transform=ax_tss.transAxes,
                       fontsize=15, fontweight='bold', va='center', rotation=90)
    
    # Overall title
    fig.suptitle('Model Performance Comparison',
                fontsize=16, fontweight='bold', y=0.96)
    
    # Save
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
