#!/usr/bin/env python3
"""
Consolidate per-timestamp NPZ files into per-HARP bundles for faster I/O.

This reduces 151GB of individual NPZ files to ~40-50GB of consolidated files:
- Bundles all frames for each HARP into a single .npz file
- Stores as float16 (50% size reduction)
- Pre-processes and normalizes (no runtime cost)
- 50x fewer file opens during training

Usage:
    python tools/consolidate_frames.py \
        --frames-meta /Volumes/Lexar/flare_forecasting/frames_meta.parquet \
        --npz-root /Volumes/Lexar/flare_forecasting/frames \
        --output-dir ~/flare_data/consolidated \
        --target-size 64 \
        --workers 8
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def process_harp(args: tuple) -> dict | None:
    """Process a single HARP - runs in parallel."""
    harpnum, group_data, npz_root, target_size, output_dir = args
    
    # Skip if already processed
    output_file = Path(output_dir) / f"H{harpnum}.npz"
    if output_file.exists():
        # Return existing info without reprocessing
        try:
            with np.load(output_file, allow_pickle=True) as f:
                n_frames = f["frames"].shape[0]
                timestamps = f["timestamps"]
                return {
                    "harpnum": harpnum,
                    "file": str(output_file),
                    "n_frames": n_frames,
                    "start": str(timestamps[0]),
                    "end": str(timestamps[-1]),
                    "skipped": True
                }
        except Exception:
            pass  # File corrupted, reprocess
    
    harp_data = {}
    timestamps = []
    
    for row in group_data:
        frame_path = row["frame_path"]
        date_obs = row["date_obs"]
        
        # Normalize path
        frame_path = frame_path.replace("\\", "/")
        if frame_path.startswith("frames/"):
            frame_path = frame_path[7:]
        
        full_path = Path(npz_root) / frame_path
        if not full_path.exists():
            continue
        
        try:
            npz = np.load(full_path, allow_pickle=False)
            
            # Load all 3 components
            if "Bx" in npz and "By" in npz and "Bz" in npz:
                bx = npz["Bx"].astype(np.float32)
                by = npz["By"].astype(np.float32)
                bz = npz["Bz"].astype(np.float32)
            elif "Br" in npz: # Fallback for raw SHARP if needed, though convert_and_preprocess makes Bx/By/Bz
                 # If we only have Br, we can't really get vector data. 
                 # But assuming we run on convert_and_preprocess_sharp output:
                 continue
            else:
                continue
            
            # Resize if needed (center crop/pad)
            if bz.shape != (target_size, target_size):
                H, W = bz.shape
                def resize_crop(img):
                    out = np.zeros((target_size, target_size), np.float32)
                    y0 = max(0, (target_size - H) // 2)
                    x0 = max(0, (target_size - W) // 2)
                    y1 = y0 + min(H, target_size)
                    x1 = x0 + min(W, target_size)
                    sy0 = max(0, (H - target_size) // 2)
                    sx0 = max(0, (W - target_size) // 2)
                    sy1 = sy0 + (y1 - y0)
                    sx1 = sx0 + (x1 - x0)
                    out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
                    return out
                
                bx = resize_crop(bx)
                by = resize_crop(by)
                bz = resize_crop(bz)
            
            # Normalize (same as cached_dataset.py)
            # Note: Bx/By might have different ranges, but typical field strength normalization 
            # of dividing by ~2000G-3000G is standard. 
            # We apply the same normalization to all components to preserve vector direction.
            for img in [bx, by, bz]:
                if not np.isfinite(img).all():
                    img[~np.isfinite(img)] = 0.0
            
            # Use max of absolute value across all components for scaling if dynamic
            # But consistent scaling is better. Let's stick to the logic but applied per channel or globally?
            # Ideally we normalize vectors together. For now, applying existing logic per channel 
            # is safe enough for raw magnitude, but might distort direction slightly if ranges differ wildly.
            # Standard approach: Divide all by constant (e.g. 2000 G).
            # The existing logic:
            # if data_range > 10: img = img / 2000.0
            
            stack = np.stack([bx, by, bz], axis=0) # [3, H, W]
            
            # Normalize
            for c in range(3):
                img = stack[c]
                data_range = np.abs(img).max()
                if data_range > 10:
                    img = img / 2000.0
                elif data_range > 0:
                    img = img / max(data_range, 5.0)
                stack[c] = np.clip(img, -1.5, 1.5)
            
            ts = pd.Timestamp(date_obs).floor('s')  # Round to seconds to match lookup format
            ts_key = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")  # Consistent format
            harp_data[ts_key] = stack
            timestamps.append(ts)
            
        except Exception:
            continue
    
    if len(harp_data) == 0:
        return None
    
    # Save consolidated file
    output_file = Path(output_dir) / f"H{harpnum}.npz"
    timestamps = sorted(timestamps)
    # Use consistent timestamp format that matches the lookup
    frames = np.stack([harp_data[t.strftime("%Y-%m-%dT%H:%M:%S+00:00")] for t in timestamps], axis=0)
    ts_strings = [t.strftime("%Y-%m-%dT%H:%M:%S+00:00") for t in timestamps]
    
    # Save as compressed float16 (half the size!)
    np.savez_compressed(
        output_file,
        frames=frames.astype(np.float16),
        timestamps=np.array(ts_strings, dtype=object)
    )
    
    return {
        "harpnum": harpnum,
        "file": str(output_file),
        "n_frames": len(timestamps),
        "start": timestamps[0].isoformat(),
        "end": timestamps[-1].isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Consolidate NPZ frames into per-HARP bundles")
    parser.add_argument("--frames-meta", required=True, help="Path to frames_meta.parquet")
    parser.add_argument("--npz-root", required=True, help="Root directory of NPZ files")
    parser.add_argument("--output-dir", required=True, help="Output directory for consolidated files")
    parser.add_argument("--target-size", type=int, default=64, help="Target frame size (default: 64)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: CPU count)")
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = min(8, cpu_count())
    
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading frames metadata from {args.frames_meta}...")
    meta = pd.read_parquet(args.frames_meta)
    meta["date_obs"] = pd.to_datetime(meta["date_obs"], utc=True)
    meta = meta.sort_values(["harpnum", "date_obs"]).reset_index(drop=True)
    
    print(f"  Total frames: {len(meta):,}")
    print(f"  Unique HARPs: {meta['harpnum'].nunique():,}")
    
    # Group by HARP and convert to list of dicts for multiprocessing
    print("Preparing data for parallel processing...")
    tasks = []
    for harpnum, group in meta.groupby("harpnum"):
        group_data = group[["frame_path", "date_obs"]].to_dict("records")
        tasks.append((harpnum, group_data, args.npz_root, args.target_size, str(output_dir)))
    
    print(f"Processing {len(tasks)} HARPs with {args.workers} workers...")
    print(f"Output: {output_dir}")
    print()
    
    manifest = {}
    
    if args.workers == 1:
        # Single-threaded (most reliable for external drives)
        for task in tqdm(tasks, desc="Consolidating"):
            try:
                r = process_harp(task)
                if r is not None:
                    manifest[r["harpnum"]] = r
            except Exception as e:
                print(f"Warning: Failed to process HARP {task[0]}: {e}")
    else:
        # Multi-threaded with chunked processing (more stable than imap)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_harp, task): task[0] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Consolidating"):
                harpnum = futures[future]
                try:
                    r = future.result(timeout=120)  # 2 min timeout per HARP
                    if r is not None:
                        manifest[r["harpnum"]] = r
                except Exception as e:
                    print(f"Warning: Failed HARP {harpnum}: {e}")
    
    # Save manifest (needed for the consolidated dataset loader)
    manifest_file = output_dir / "manifest.pkl"
    with open(manifest_file, "wb") as f:
        pickle.dump(manifest, f)
    
    # Calculate stats
    total_frames = sum(m["n_frames"] for m in manifest.values())
    total_size_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npz"))
    total_size_gb = total_size_bytes / 1e9
    
    print()
    print("=" * 60)
    print("CONSOLIDATION COMPLETE!")
    print("=" * 60)
    print(f"HARPs processed: {len(manifest):,}")
    print(f"Total frames: {total_frames:,}")
    print(f"Output size: {total_size_gb:.2f} GB")
    print(f"Output directory: {output_dir}")
    print(f"Manifest file: {manifest_file}")
    print()
    print("Next steps:")
    print(f"  1. Copy {output_dir} to your local SSD")
    print("  2. Update your config to use ConsolidatedWindowsDataset")
    print("=" * 60)


if __name__ == "__main__":
    main()


