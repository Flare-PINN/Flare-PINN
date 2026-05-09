#!/usr/bin/env python3
"""
Measure inference latency and throughput for PINN model.

Useful for reporting model efficiency in the paper.
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.pinn import PINNConfig, HybridPINNModel
from data.consolidated_dataset import ConsolidatedWindowsDataset
from utils.masked_training import load_windows_with_mask


def measure_latency(model, sample, device, n_warmup=5, n_runs=50):
    """Measure inference latency."""
    
    # Prepare inputs
    frames = sample["frames"].to(device)
    scalars = sample["scalars"].unsqueeze(0).to(device)
    observed_mask = sample["observed_mask"].to(device)
    
    T, C, H, W = frames.shape
    t_idx = T - 1
    
    # Dense spatial grid
    n_points = H * W
    x = torch.linspace(-1, 1, H, device=device)
    y = torch.linspace(-1, 1, W, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    t_norm = 2 * t_idx / (T - 1) - 1
    tt = torch.full_like(xx, t_norm)
    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3).unsqueeze(0)
    
    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(coords=coords, frames=frames, scalars=scalars, observed_mask=observed_mask)
    
    # Benchmark
    print(f"  Benchmarking ({n_runs} runs)...")
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            
            start = time.time()
            out = model(coords=coords, frames=frames, scalars=scalars, observed_mask=observed_mask)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'n_points': n_points,
    }


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def estimate_memory(model, sample, device):
    """Estimate GPU memory usage."""
    if device.type not in ["cuda", "mps"]:
        return {'memory_mb': 0}
    
    frames = sample["frames"].to(device)
    scalars = sample["scalars"].unsqueeze(0).to(device)
    observed_mask = sample["observed_mask"].to(device)
    
    T, C, H, W = frames.shape
    x = torch.linspace(-1, 1, H, device=device)
    y = torch.linspace(-1, 1, W, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    t_norm = 1.0
    tt = torch.full_like(xx, t_norm)
    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3).unsqueeze(0)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        
        with torch.no_grad():
            _ = model(coords=coords, frames=frames, scalars=scalars, observed_mask=observed_mask)
        
        torch.cuda.synchronize()
        memory_bytes = torch.cuda.max_memory_allocated(device)
        memory_mb = memory_bytes / (1024 ** 2)
    else:
        # MPS doesn't have detailed memory tracking
        memory_mb = 0
    
    return {'memory_mb': memory_mb}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure PINN inference latency")
    parser.add_argument("--config", type=Path,
                       default=project_root / "src/configs/flare_pinn_final.yaml")
    parser.add_argument("--checkpoint", type=Path,
                       default=project_root / "outputs/checkpoints/Final model PINN/checkpoint_step_0046000.pt")
    parser.add_argument("--data", type=Path,
                       default=project_root / "data/windows_test_15.parquet")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PINN INFERENCE LATENCY BENCHMARK")
    print("=" * 70)
    
    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n📱 Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    cfg = PINNConfig.from_yaml(args.config)
    model = HybridPINNModel(cfg)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
        model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
        print("   Using EMA weights")
    else:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print("   Using model weights")
    
    model.to(device)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    print(f"\n Model size:")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Size: ~{params['total'] * 4 / (1024**2):.1f} MB (FP32)")
    
    # Load data
    print(f"\n📂 Loading data...")
    windows_df, _ = load_windows_with_mask(args.data)
    dataset = ConsolidatedWindowsDataset(
        windows_df=windows_df,
        consolidated_dir=str(cfg.data.consolidated_dir),
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=512,
        training=False,
        augment=False,
        max_cached_harps=100,
    )
    
    # Sample windows
    indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
    
    all_results = []
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{len(indices)}")
        print(f"{'='*50}")
        
        # Measure latency
        result = measure_latency(model, sample, device, n_warmup=5, n_runs=args.n_runs)
        all_results.append(result)
        
        print(f"  Mean: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        print(f"  Median: {result['median_ms']:.2f} ms")
        print(f"  P95: {result['p95_ms']:.2f} ms")
        print(f"  Min/Max: {result['min_ms']:.2f} / {result['max_ms']:.2f} ms")
        print(f"  Spatial points: {result['n_points']:,}")
    
    # Aggregate statistics
    mean_times = [r['mean_ms'] for r in all_results]
    median_times = [r['median_ms'] for r in all_results]
    p95_times = [r['p95_ms'] for r in all_results]
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nAcross {len(all_results)} samples:")
    print(f"  Mean latency: {np.mean(mean_times):.2f} ± {np.std(mean_times):.2f} ms")
    print(f"  Median latency: {np.median(median_times):.2f} ms")
    print(f"  P95 latency: {np.median(p95_times):.2f} ms")
    print(f"  Throughput: ~{1000 / np.mean(mean_times):.1f} inferences/sec")
    
    # Memory estimate
    print(f"\n💾 Estimating memory...")
    mem = estimate_memory(model, dataset[indices[0]], device)
    if mem['memory_mb'] > 0:
        print(f"   Peak GPU memory: {mem['memory_mb']:.1f} MB")
    else:
        print(f"   Memory tracking not available for {device.type}")
    
    # Paper-ready summary
    print("\n" + "=" * 70)
    print(" FOR PAPER:")
    print("=" * 70)
    print(f"""
Model Efficiency:
- Parameters: {params['total'] / 1e6:.1f}M ({params['trainable'] / 1e6:.1f}M trainable)
- Inference time: {np.mean(mean_times):.1f} ± {np.std(mean_times):.1f} ms per window
- Throughput: ~{1000 / np.mean(mean_times):.0f} inferences/sec on {device.type.upper()}
- Spatial resolution: {int(np.sqrt(all_results[0]['n_points']))} × {int(np.sqrt(all_results[0]['n_points']))} grid

This enables real-time operational forecasting (< 100ms per active region).
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

