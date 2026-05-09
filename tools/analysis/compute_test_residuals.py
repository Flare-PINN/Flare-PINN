#!/usr/bin/env python3
"""
Compute physics residuals on test set for PINN vs baseline.

This script evaluates the weak-form induction equation residual on held-out
test data to demonstrate that physics-trained models generalize the constraint.

Usage:
    python tools/compute_test_residuals.py \
        --pinn-config src/configs/flare_pinn_final.yaml \
        --pinn-checkpoint outputs/checkpoints/Final\ model\ PINN/checkpoint_step_0046000.pt \
        --baseline-config src/configs/benchmark_classifier.yaml \
        --baseline-checkpoint outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt \
        --test-data data/windows_test_15.parquet \
        --output final_results/physics_residuals/
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.pinn.config import PINNConfig
from models.pinn.hybrid_model import HybridPINNModel
from models.pinn.physics import VectorInduction2p5D
from data.consolidated_dataset import ConsolidatedWindowsDataset


def load_model(config_path: Path, checkpoint_path: Path, device: str = "cpu"):
    """Load model from checkpoint."""
    print(f"Loading config: {config_path}")
    cfg = PINNConfig.from_yaml(config_path)
    
    print(f"Creating model...")
    model = HybridPINNModel(cfg, encoder_in_channels=cfg.data.n_components)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle EMA weights if present
    if "ema_state_dict" in ckpt:
        print("Using EMA weights")
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, cfg


def compute_sample_residual(
    model: HybridPINNModel,
    coords: torch.Tensor,
    frames: torch.Tensor,
    gt_bz: torch.Tensor,
    observed_mask: torch.Tensor,
    pil_mask: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """
    Compute physics residual for a single sample.
    
    Returns:
        dict with:
            - residual_mean: mean residual magnitude
            - residual_std: std of residual
            - residual_max: max residual
            - residual_median: median residual
    """
    # Move to device
    coords = coords.to(device)
    frames = frames.to(device)
    gt_bz = gt_bz.to(device)
    observed_mask = observed_mask.to(device)
    if pil_mask is not None:
        pil_mask = pil_mask.to(device)
    
    # Encode frames (no grad needed here)
    with torch.no_grad():
        L, g = model.encode_frames(frames, observed_mask)
        L = L.detach()
        g = g.detach()
    
    # Flatten coords for field query (NEED GRADIENTS for physics)
    T, P = coords.shape[0], coords.shape[1]
    coords_flat = coords.reshape(-1, 3).contiguous()
    coords_flat = coords_flat.detach().clone().requires_grad_(True)
    
    # Get physics module
    physics = model.physics
    
    # Model wrapper for physics evaluation
    def model_wrapper(c):
        c = c.clamp(-1.0, 1.0)
        out = model.backbone(c, L, g, use_nearest=False)
        
        B = out["B"].clamp(-10.0, 10.0)
        u = out["u"].clamp(-5.0, 5.0)
        
        if model.n_components == 1:
            return {
                "B_z": B,
                "u_x": u[..., 0:1],
                "u_y": u[..., 1:2],
                "eta_raw": out["eta_raw"]
            }
        else:
            return {
                "B": B,
                "u": u,
                "B_x": B[..., 0:1],
                "B_y": B[..., 1:2],
                "B_z": B[..., 2:3],
                "u_x": u[..., 0:1],
                "u_y": u[..., 1:2],
                "eta_raw": out["eta_raw"]
            }
    
    # Compute importance weights (uniform for residual evaluation)
    N = coords_flat.shape[0]
    imp_weights = torch.ones((N, 1), device=device)
    
    # Compute physics residual
    try:
        eta_mode = "field" if model.cfg.model.learn_eta else "scalar"
        residual, _ = physics(
            model_wrapper,
            coords_flat,
            imp_weights,
            eta_mode=eta_mode,
            eta_scalar=model.cfg.model.eta_scalar
        )
        
        # If residual is scalar, convert to tensor
        if isinstance(residual, float):
            residual = torch.tensor(residual, device=device)
        
        # Detach and move to CPU for statistics
        residual = residual.detach().cpu()
        
        # Get absolute value for statistics
        residual_abs = residual.abs().item() if residual.numel() == 1 else residual.abs()
        
        # Compute statistics
        if isinstance(residual_abs, torch.Tensor) and residual_abs.numel() > 1:
            stats = {
                "residual_mean": float(residual_abs.mean()),
                "residual_std": float(residual_abs.std()),
                "residual_max": float(residual_abs.max()),
                "residual_median": float(residual_abs.median()),
            }
        else:
            val = float(residual_abs) if isinstance(residual_abs, torch.Tensor) else residual_abs
            stats = {
                "residual_mean": val,
                "residual_std": 0.0,
                "residual_max": val,
                "residual_median": val,
            }
    except Exception as e:
        print(f"Warning: Failed to compute residual: {e}")
        stats = {
            "residual_mean": float('nan'),
            "residual_std": float('nan'),
            "residual_max": float('nan'),
            "residual_median": float('nan'),
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute test-set physics residuals")
    parser.add_argument("--pinn-config", type=str, required=True)
    parser.add_argument("--pinn-checkpoint", type=str, required=True)
    parser.add_argument("--baseline-config", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output", type=str, default="final_results/physics_residuals/")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    print("\n[1/2] Loading PINN model...")
    pinn_model, pinn_cfg = load_model(
        Path(args.pinn_config),
        Path(args.pinn_checkpoint),
        args.device
    )
    
    print("\n[2/2] Loading Baseline model...")
    baseline_model, baseline_cfg = load_model(
        Path(args.baseline_config),
        Path(args.baseline_checkpoint),
        args.device
    )
    
    # Load test dataset
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    # Load windows DataFrame
    windows_df = pd.read_parquet(args.test_data)
    print(f"Loaded {len(windows_df)} test windows from {args.test_data}")
    
    test_dataset = ConsolidatedWindowsDataset(
        windows_df=windows_df,
        consolidated_dir=Path(pinn_cfg.data.consolidated_dir).expanduser(),
        target_px=pinn_cfg.data.target_size,
        input_hours=pinn_cfg.data.input_hours,
        horizons=pinn_cfg.classifier.horizons,
        P_per_t=pinn_cfg.data.P_per_t,
        pil_top_pct=pinn_cfg.data.pil_top_pct,
        training=False,  # Evaluation mode
        augment=False,   # No augmentation
        use_pil_evolution=getattr(pinn_cfg.data, 'use_pil_evolution', True),
        use_temporal_statistics=getattr(pinn_cfg.data, 'use_temporal_statistics', True),
        fast_mode=False,  # Full features for proper evaluation
    )
    
    n_samples = len(test_dataset) if args.n_samples is None else min(args.n_samples, len(test_dataset))
    print(f"Processing {n_samples} test samples with {args.num_workers} workers...")
    
    # Create subset dataset if needed
    if args.n_samples is not None:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, range(n_samples))
    
    # Create DataLoader (num_workers=0 due to unpickleable dataset objects)
    def collate_fn(batch):
        """Custom collate to handle variable-size tensors."""
        return batch  # Return list of samples, don't stack
    
    # Note: ConsolidatedWindowsDataset has RLock objects that can't be pickled
    # So we use num_workers=0 but process in batches for efficiency
    effective_workers = 0  # Force 0 due to pickle issues
    if args.num_workers > 0:
        print(f"Note: Using num_workers=0 (dataset has unpickleable objects)")
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        collate_fn=collate_fn,
    )
    
    # Compute residuals for both models
    print("\n" + "="*80)
    print("COMPUTING RESIDUALS")
    print("="*80)
    
    pinn_residuals = []
    baseline_residuals = []
    sample_info = []
    sample_idx = 0
    
    for batch in tqdm(dataloader, desc="Computing residuals", total=len(dataloader)):
        for sample in batch:
            try:
                # Extract data
                coords = sample["coords"]
                frames = sample["frames"]
                gt_bz = sample["gt_bz"]
                observed_mask = sample["observed_mask"]
                pil_mask = sample.get("pil_mask", None)
                labels = sample["labels"]
                
                # Compute PINN residual
                pinn_stats = compute_sample_residual(
                    pinn_model, coords, frames, gt_bz, observed_mask, pil_mask, args.device
                )
                
                # Compute baseline residual
                baseline_stats = compute_sample_residual(
                    baseline_model, coords, frames, gt_bz, observed_mask, pil_mask, args.device
                )
                
                # Store results
                pinn_residuals.append(pinn_stats)
                baseline_residuals.append(baseline_stats)
                
                # Store sample info
                sample_info.append({
                    "sample_idx": sample_idx,
                    "has_flare_6h": bool(labels[0]),
                    "has_flare_12h": bool(labels[1]),
                    "has_flare_24h": bool(labels[2]),
                })
                sample_idx += 1
                
            except Exception as e:
                print(f"\nWarning: Failed to process sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                sample_idx += 1
                continue
        
        # Memory cleanup after each batch
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
    
    # Convert to DataFrame
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    df_pinn = pd.DataFrame(pinn_residuals)
    df_baseline = pd.DataFrame(baseline_residuals)
    df_info = pd.DataFrame(sample_info)
    
    # Combine
    df_results = pd.concat([
        df_info,
        df_pinn.add_prefix("pinn_"),
        df_baseline.add_prefix("baseline_"),
    ], axis=1)
    
    # Save results
    csv_path = output_dir / "test_residuals.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved residuals to: {csv_path}")
    
    # Save summary statistics
    summary_path = output_dir / "residual_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("TEST SET PHYSICS RESIDUAL COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("PINN Model (Physics-Trained):\n")
        f.write("-" * 40 + "\n")
        for col in ["residual_mean", "residual_median", "residual_std", "residual_max"]:
            pinn_col = f"pinn_{col}"
            vals = df_results[pinn_col].dropna()
            f.write(f"{col:20s}: median={vals.median():.6f}, mean={vals.mean():.6f}, std={vals.std():.6f}\n")
        
        f.write("\nBaseline Model (No Physics):\n")
        f.write("-" * 40 + "\n")
        for col in ["residual_mean", "residual_median", "residual_std", "residual_max"]:
            baseline_col = f"baseline_{col}"
            vals = df_results[baseline_col].dropna()
            f.write(f"{col:20s}: median={vals.median():.6f}, mean={vals.mean():.6f}, std={vals.std():.6f}\n")
        
        f.write("\nRelative Improvement:\n")
        f.write("-" * 40 + "\n")
        for col in ["residual_mean", "residual_median"]:
            pinn_vals = df_results[f"pinn_{col}"].dropna()
            baseline_vals = df_results[f"baseline_{col}"].dropna()
            
            pinn_med = pinn_vals.median()
            baseline_med = baseline_vals.median()
            
            improvement = (baseline_med - pinn_med) / baseline_med * 100
            f.write(f"{col:20s}: {improvement:.1f}% reduction\n")
    
    print(f"Saved summary to: {summary_path}")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot 1: Bar chart with error bars (cleaner for paper)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute statistics
    pinn_median = df_results["pinn_residual_median"].median()
    baseline_median = df_results["baseline_residual_median"].median()
    pinn_iqr = df_results["pinn_residual_median"].quantile(0.75) - df_results["pinn_residual_median"].quantile(0.25)
    baseline_iqr = df_results["baseline_residual_median"].quantile(0.75) - df_results["baseline_residual_median"].quantile(0.25)
    
    improvement_pct = (baseline_median - pinn_median) / baseline_median * 100
    
    # Left: Bar chart
    x = np.arange(2)
    medians = [pinn_median, baseline_median]
    iqrs = [pinn_iqr, baseline_iqr]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = axes[0].bar(x, medians, yerr=iqrs, capsize=10, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['PINN\n(Physics)', 'Baseline\n(No Physics)'], fontsize=12)
    axes[0].set_ylabel('Median Physics Residual', fontsize=12)
    axes[0].set_title('Test-Set Induction Equation Residual\n(N=5,716 samples)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, medians)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + iqrs[i] + height*0.05,
                    f'{val:.2e}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    axes[0].text(0.5, max(medians) * 0.5, f'{improvement_pct:.1f}%\nreduction',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Right: Histogram comparison
    bins = np.linspace(
        min(df_results["pinn_residual_median"].min(), df_results["baseline_residual_median"].min()),
        max(df_results["pinn_residual_median"].max(), df_results["baseline_residual_median"].max()),
        50
    )
    
    axes[1].hist(df_results["baseline_residual_median"], bins=bins, alpha=0.6, 
                 label='Baseline', color='#e74c3c', edgecolor='black')
    axes[1].hist(df_results["pinn_residual_median"], bins=bins, alpha=0.6,
                 label='PINN', color='#2ecc71', edgecolor='black')
    axes[1].axvline(baseline_median, color='#e74c3c', linestyle='--', linewidth=2, label='Baseline median')
    axes[1].axvline(pinn_median, color='#2ecc71', linestyle='--', linewidth=2, label='PINN median')
    axes[1].set_xlabel('Physics Residual Magnitude', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "residual_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
    plt.close()
    
    # Plot 2: Grouped bar chart by flare status
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Compute statistics by flare status
    flare_mask = df_results["has_flare_24h"] == True
    noflare_mask = df_results["has_flare_24h"] == False
    
    # Flare samples
    pinn_flare_med = df_results[flare_mask]["pinn_residual_median"].median()
    baseline_flare_med = df_results[flare_mask]["baseline_residual_median"].median()
    pinn_flare_iqr = (df_results[flare_mask]["pinn_residual_median"].quantile(0.75) - 
                      df_results[flare_mask]["pinn_residual_median"].quantile(0.25))
    baseline_flare_iqr = (df_results[flare_mask]["baseline_residual_median"].quantile(0.75) - 
                          df_results[flare_mask]["baseline_residual_median"].quantile(0.25))
    
    # No-flare samples
    pinn_noflare_med = df_results[noflare_mask]["pinn_residual_median"].median()
    baseline_noflare_med = df_results[noflare_mask]["baseline_residual_median"].median()
    pinn_noflare_iqr = (df_results[noflare_mask]["pinn_residual_median"].quantile(0.75) - 
                        df_results[noflare_mask]["pinn_residual_median"].quantile(0.25))
    baseline_noflare_iqr = (df_results[noflare_mask]["baseline_residual_median"].quantile(0.75) - 
                            df_results[noflare_mask]["baseline_residual_median"].quantile(0.25))
    
    # Grouped bar chart
    x = np.arange(2)
    width = 0.35
    
    pinn_vals = [pinn_flare_med, pinn_noflare_med]
    baseline_vals = [baseline_flare_med, baseline_noflare_med]
    pinn_errs = [pinn_flare_iqr, pinn_noflare_iqr]
    baseline_errs = [baseline_flare_iqr, baseline_noflare_iqr]
    
    bars1 = ax.bar(x - width/2, pinn_vals, width, yerr=pinn_errs, label='PINN (Physics)',
                   color='#2ecc71', alpha=0.7, capsize=8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, baseline_vals, width, yerr=baseline_errs, label='Baseline (No Physics)',
                   color='#e74c3c', alpha=0.7, capsize=8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Sample Type', fontsize=13)
    ax.set_ylabel('Median Physics Residual', fontsize=13)
    ax.set_title('Physics Residual by Flare Status (24h horizon)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Flare\nSamples\n(N=' + str(flare_mask.sum()) + ')', 
                        'Non-Flare\nSamples\n(N=' + str(noflare_mask.sum()) + ')'], fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + bar.get_height()*0.1,
                   f'{height:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_path_flare = output_dir / "residual_by_flare.png"
    plt.savefig(plot_path_flare, dpi=300, bbox_inches="tight")
    print(f"Saved flare comparison to: {plot_path_flare}")
    plt.close()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - residual_comparison.png")
    print(f"  - residual_by_flare.png")
    
    # Print quick summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    pinn_median = df_results["pinn_residual_mean"].dropna().median()
    baseline_median = df_results["baseline_residual_mean"].dropna().median()
    
    # Handle zero baseline case
    if baseline_median > 1e-10:
        improvement = (baseline_median - pinn_median) / baseline_median * 100
        print(f"PINN median residual:     {pinn_median:.6f}")
        print(f"Baseline median residual: {baseline_median:.6f}")
        print(f"Improvement:              {improvement:.1f}% reduction")
    else:
        print(f"PINN median residual:     {pinn_median:.6f}")
        print(f"Baseline median residual: {baseline_median:.6f}")
        print(f"Note: Baseline residual too small for reliable comparison")


if __name__ == "__main__":
    main()

