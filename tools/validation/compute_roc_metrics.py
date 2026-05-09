#!/usr/bin/env python3
"""
Compute ROC and PR metrics for all models and save to CSV.

Computes:
- ROC-AUC
- PR-AUC
- Max TSS
- TSS at D2C threshold
- Base rate

For all models (PINN, Strong-Form, Benchmark) across all horizons.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import (
    confusion_at_threshold,
    tss_at_threshold,
    precision_recall_curve
)


def compute_roc_curve(y_true, y_prob, n=512):
    """Compute ROC curve (TPR vs FPR)."""
    thresholds = np.linspace(0, 1, n)[::-1]
    tpr_list, fpr_list = [], []
    
    for thr in thresholds:
        tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, thr)
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_pr_curve(y_true, y_prob, n=512):
    """Compute Precision-Recall curve."""
    recall, precision, thresholds = precision_recall_curve(y_true, y_prob, n=n)
    return recall, precision, thresholds


def compute_tss_curve(y_true, y_prob, n=512):
    """Compute TSS vs threshold curve."""
    thresholds = np.linspace(0, 1, n)
    tss_list = []
    
    for thr in thresholds:
        tss = tss_at_threshold(y_true, y_prob, thr)
        tss_list.append(tss)
    
    return thresholds, np.array(tss_list)


def compute_metrics_for_model(data, model_name, horizons):
    """Compute all metrics for one model."""
    results = []
    
    for col_idx, horizon in enumerate(horizons):
        y_true = data['labels'][:, col_idx]
        y_prob = data['probs'][:, col_idx]
        threshold = data['thresholds'][col_idx]
        
        # Remove NaN/Inf
        valid = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[valid]
        y_prob = np.clip(y_prob[valid], 0.0, 1.0)
        
        # Compute curves
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob, n=512)
        recall, precision, _ = compute_pr_curve(y_true, y_prob, n=512)
        tss_thrs, tss_vals = compute_tss_curve(y_true, y_prob, n=512)
        
        # Compute AUC metrics
        idx_roc = np.argsort(fpr)
        roc_auc = float(np.trapz(tpr[idx_roc], fpr[idx_roc]))
        
        idx_pr = np.argsort(recall)
        pr_auc = float(np.trapz(precision[idx_pr], recall[idx_pr]))
        
        # TSS metrics
        max_tss = float(tss_vals.max())
        tss_at_d2c = float(tss_at_threshold(y_true, y_prob, threshold))
        
        # Base rate
        base_rate = float(y_true.mean())
        
        # Store results
        results.append({
            'Model': model_name,
            'Horizon': f'{horizon}h',
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'Max_TSS': max_tss,
            'TSS_at_D2C': tss_at_d2c,
            'D2C_Threshold': float(threshold),
            'Base_Rate': base_rate,
            'PR_AUC_vs_Random': pr_auc / base_rate if base_rate > 0 else 0,
        })
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute ROC/PR metrics for all models")
    parser.add_argument("--pinn", type=str, required=True, help="Path to Flare-PINN test NPZ")
    parser.add_argument("--strong", type=str, required=True, help="Path to Strong-Form test NPZ")
    parser.add_argument("--baseline", type=str, required=True, help="Path to Benchmark test NPZ")
    parser.add_argument("--logreg", type=str, default=None, help="Path to Logistic Regression test NPZ")
    parser.add_argument("--xgboost", type=str, default=None, help="Path to XGBoost test NPZ")
    parser.add_argument("--svm", type=str, default=None, help="Path to SVM test NPZ")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()
    
    print("="*80)
    print("COMPUTING ROC/PR METRICS FOR ALL MODELS")
    print("="*80)
    
    # Load data
    print("\nLoading predictions...")
    pinn_data = np.load(args.pinn)
    strong_data = np.load(args.strong)
    baseline_data = np.load(args.baseline)
    
    horizons = [6, 12, 24]
    
    # Compute metrics for each model
    all_results = []
    
    print("\nFlare-PINN (Weak-Form):")
    pinn_results = compute_metrics_for_model(pinn_data, "Flare-PINN (Weak-Form)", horizons)
    for r in pinn_results:
        print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
    all_results.extend(pinn_results)
    
    print("\nStrong-Form:")
    strong_results = compute_metrics_for_model(strong_data, "Strong-Form", horizons)
    for r in strong_results:
        print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
    all_results.extend(strong_results)
    
    print("\nBenchmark (No Physics):")
    baseline_results = compute_metrics_for_model(baseline_data, "Benchmark (No Physics)", horizons)
    for r in baseline_results:
        print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
    all_results.extend(baseline_results)
    
    # Add classical baselines if provided
    if args.logreg:
        print("\nLogistic Regression:")
        logreg_data = np.load(args.logreg)
        logreg_results = compute_metrics_for_model(logreg_data, "Logistic Regression", horizons)
        for r in logreg_results:
            print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
        all_results.extend(logreg_results)
    
    if args.xgboost:
        print("\nXGBoost:")
        xgboost_data = np.load(args.xgboost)
        xgboost_results = compute_metrics_for_model(xgboost_data, "XGBoost", horizons)
        for r in xgboost_results:
            print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
        all_results.extend(xgboost_results)
    
    if args.svm:
        print("\nSVM:")
        svm_data = np.load(args.svm)
        svm_results = compute_metrics_for_model(svm_data, "SVM", horizons)
        for r in svm_results:
            print(f"  {r['Horizon']}: ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, Max TSS={r['Max_TSS']:.4f}")
        all_results.extend(svm_results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print('='*80)
    print(df.to_string(index=False))
    
    print(f"\n Saved ROC/PR metrics to: {output_path}")


if __name__ == "__main__":
    main()

