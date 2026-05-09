#!/usr/bin/env python3
"""
Apply Platt scaling to calibrate model probabilities.

Fits on validation set, applies to test set, reports before/after metrics.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.eval.metrics import adaptive_ece


def to_logits(p, eps=1e-7):
    """Convert probabilities to logits."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(logits_np, y_np, iters=2000, lr=0.01, l2=1e-4):
    """
    Fit Platt scaling: p_cal = sigmoid(a * logit(p) + b)
    
    Args:
        logits_np: Logits from validation set
        y_np: True labels from validation set
        iters: Number of optimization iterations
        lr: Learning rate
        l2: L2 regularization weight
        
    Returns:
        a, b: Platt scaling parameters
    """
    logits = torch.tensor(logits_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    a = torch.nn.Parameter(torch.ones(()))
    b = torch.nn.Parameter(torch.zeros(()))

    opt = torch.optim.Adam([a, b], lr=lr)

    best_loss = float('inf')
    best_a, best_b = 1.0, 0.0
    
    for i in range(iters):
        opt.zero_grad()
        z = a * logits + b
        loss = F.binary_cross_entropy_with_logits(z, y)
        
        if l2 > 0:
            loss = loss + l2 * (a * a + b * b)
        
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_a = float(a.detach().cpu())
            best_b = float(b.detach().cpu())
    
    return best_a, best_b


def apply_platt(probs, a, b, eps=1e-7):
    """Apply Platt scaling to probabilities."""
    logits = to_logits(probs, eps)
    z = a * logits + b
    return 1 / (1 + np.exp(-z))


def compute_metrics(y_true, y_prob):
    """Compute Brier and ECE."""
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 1e-7, 1 - 1e-7)
    
    brier = brier_score_loss(y_true, y_prob)
    ece = adaptive_ece(y_true, y_prob, n_bins=15)
    
    base_rate = y_true.mean()
    mean_prob = y_prob.mean()
    
    return {
        'brier': brier,
        'ece': ece,
        'base_rate': base_rate,
        'mean_prob': mean_prob,
        'inflation': mean_prob / base_rate if base_rate > 0 else 0,
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum())
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply Platt scaling calibration")
    parser.add_argument("--val-results", type=Path, required=True,
                       help="Path to validation NPZ (e.g., checkpoint_step_0046000_validation_on_val.npz)")
    parser.add_argument("--test-results", type=Path, required=True,
                       help="Path to test NPZ (e.g., checkpoint_step_0046000_validation.npz)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Optional: Save calibrated test results")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PLATT SCALING CALIBRATION")
    print("=" * 70)
    print()
    
    # Load validation results
    print(f"📦 Loading validation results from {args.val_results.name}")
    val_npz = np.load(args.val_results, allow_pickle=True)
    val_probs = val_npz['probs']  # [N, 3]
    val_labels = val_npz['labels']  # [N, 3]
    
    # Load test results
    print(f"📦 Loading test results from {args.test_results.name}")
    test_npz = np.load(args.test_results, allow_pickle=True)
    test_probs = test_npz['probs']  # [N, 3]
    test_labels = test_npz['labels']  # [N, 3]
    print()
    
    horizons = ['6h', '12h', '24h']
    platt_params = {}
    calibrated_probs = np.zeros_like(test_probs)
    
    for i, horizon in enumerate(horizons):
        print("=" * 70)
        print(f"{horizon.upper()} HORIZON")
        print("=" * 70)
        
        # Get validation data
        val_p = val_probs[:, i]
        val_y = val_labels[:, i]
        valid_val = np.isfinite(val_p) & np.isfinite(val_y)
        val_p = val_p[valid_val]
        val_y = val_y[valid_val]
        
        # Get test data
        test_p = test_probs[:, i]
        test_y = test_labels[:, i]
        valid_test = np.isfinite(test_p) & np.isfinite(test_y)
        test_p_valid = test_p[valid_test]
        test_y_valid = test_y[valid_test]
        
        print(f"\nFitting Platt scaling on {len(val_y)} validation samples...")
        val_logits = to_logits(val_p)
        a, b = fit_platt(val_logits, val_y, iters=2000, lr=0.01, l2=1e-4)
        platt_params[horizon] = (a, b)
        print(f"  Platt parameters: a={a:.4f}, b={b:.4f}")
        
        # Apply to test set
        print(f"Applying to {len(test_y_valid)} test samples...")
        test_p_cal = apply_platt(test_p_valid, a, b)
        
        # Store calibrated probabilities
        calibrated_probs[valid_test, i] = test_p_cal
        calibrated_probs[~valid_test, i] = np.nan
        
        # Compute metrics
        print("\n--- BEFORE CALIBRATION ---")
        before = compute_metrics(test_y_valid, test_p_valid)
        print(f"  Brier:          {before['brier']:.4f}")
        print(f"  ECE:            {before['ece']:.4f}")
        print(f"  Mean prob:      {before['mean_prob']:.4f}")
        print(f"  Base rate:      {before['base_rate']:.4f}")
        print(f"  Inflation:      {before['inflation']:.1f}×")
        
        print("\n--- AFTER CALIBRATION ---")
        after = compute_metrics(test_y_valid, test_p_cal)
        print(f"  Brier:          {after['brier']:.4f}  (Δ = {after['brier']-before['brier']:+.4f})")
        print(f"  ECE:            {after['ece']:.4f}  (Δ = {after['ece']-before['ece']:+.4f})")
        print(f"  Mean prob:      {after['mean_prob']:.4f}  (Δ = {after['mean_prob']-before['mean_prob']:+.4f})")
        print(f"  Base rate:      {after['base_rate']:.4f}")
        print(f"  Inflation:      {after['inflation']:.1f}×")
        
        # Compute climatology baseline
        climo_brier = before['base_rate'] * (1 - before['base_rate'])
        bss_before = 1 - before['brier'] / climo_brier
        bss_after = 1 - after['brier'] / climo_brier
        
        print(f"\n  Climatology Brier: {climo_brier:.4f}")
        print(f"  BSS before:        {bss_before:+.4f}")
        print(f"  BSS after:         {bss_after:+.4f}  " if bss_after > 0 else f"  BSS after:         {bss_after:+.4f}")
        print()
    
    # Save calibrated results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            probs=test_probs,  # Original
            probs_calibrated=calibrated_probs,  # Calibrated
            labels=test_labels,
            horizons=test_npz.get('horizons', horizons),
            platt_params=platt_params
        )
        print(f"💾 Saved calibrated results to: {args.output}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nPlatt scaling parameters:")
    for horizon, (a, b) in platt_params.items():
        print(f"  {horizon}: a={a:.4f}, b={b:.4f}")


if __name__ == "__main__":
    main()

