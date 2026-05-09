# src/data/cached_dataset.py
"""
Cached Windows Dataset - loads frames with shared memory cache for fast access.
Works correctly with multiprocessing DataLoader workers.

First epoch: loads from disk (slow), subsequent: from shared cache (fast).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pil_mask import (
    pil_mask_from_bz,
    compute_r_value,
    compute_pil_gradient_weighted_length,
)


# =============================================================================
# Typed Output Models (replacing Dict[str, Any])
# =============================================================================

@dataclass
class WindowSample:
    """Strongly-typed batch sample from the dataset."""
    coords: torch.Tensor        # [T, P, 3] collocation coordinates
    gt_bz: torch.Tensor         # [T, P, 1] ground truth Bz at coords
    frames: torch.Tensor        # [T, H, W] full Bz frames
    observed_mask: torch.Tensor # [T] boolean mask
    labels: torch.Tensor        # [n_horizons] classification targets
    pil_mask: torch.Tensor      # [H, W] PIL mask as tensor (standardized type)
    scalars: torch.Tensor       # [4] scalar features: R-value, GWPIL, obs_coverage, frame_count


# =============================================================================
# Shared Frame Cache (works with multiprocessing)
# =============================================================================

class SharedFrameCache:
    """
    Thread-safe frame cache that works with DataLoader multiprocessing.
    
    Uses a simple dict in main process. For multiprocessing, frames are
    preloaded before DataLoader workers are spawned, so each worker
    gets a copy of the already-populated cache.
    """
    def __init__(self, max_size: int = 250000):
        self._cache: dict[str, np.ndarray] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get frame from cache, returns copy to avoid mutation."""
        frame = self._cache.get(key)
        if frame is not None:
            return frame.copy()
        return None
    
    def put(self, key: str, frame: np.ndarray) -> None:
        """Store frame in cache."""
        if len(self._cache) < self._max_size:
            self._cache[key] = frame.copy()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def clear(self) -> None:
        self._cache.clear()


# Global cache instance - populated before forking workers
_FRAME_CACHE = SharedFrameCache()


def _load_and_preprocess_frame(
    npz_path: str, 
    target_px: int
) -> Optional[np.ndarray]:
    """Load frame from disk and preprocess (resize, normalize, clamp)."""
    try:
        npz = np.load(npz_path, allow_pickle=False)
        bz = npz["Bz"] if "Bz" in npz else npz.get("Br")
        if bz is None:
            return None
        bz = bz.astype(np.float32)
        
        # Resize if needed (center crop/pad)
        if bz.shape != (target_px, target_px):
            H, W = bz.shape
            out = np.zeros((target_px, target_px), np.float32)
            y0 = max(0, (target_px - H) // 2)
            x0 = max(0, (target_px - W) // 2)
            y1 = y0 + min(H, target_px)
            x1 = x0 + min(W, target_px)
            sy0 = max(0, (H - target_px) // 2)
            sx0 = max(0, (W - target_px) // 2)
            sy1 = sy0 + (y1 - y0)
            sx1 = sx0 + (x1 - x0)
            out[y0:y1, x0:x1] = bz[sy0:sy1, sx0:sx1]
            bz = out
        
        # Handle non-finite values
        if not np.isfinite(bz).all():
            bz[~np.isfinite(bz)] = 0.0
        
        # Normalize based on data range
        data_range = np.abs(bz).max()
        if data_range > 10:
            bz = bz / 2000.0
        elif data_range > 0:
            bz = bz / max(data_range, 5.0)
        bz = np.clip(bz, -1.5, 1.5)
        
        return bz
    except Exception:
        return None


# =============================================================================
# Efficient Bilinear Sampling (stays on GPU)
# =============================================================================

def _bilinear_sample_bz_tensor(
    bz: torch.Tensor,  # [H, W] already on device
    xy_norm: torch.Tensor  # [P, 2] normalized coords in [-1, 1]
) -> torch.Tensor:
    """
    Bilinear interpolation of Bz field at given coordinates.
    Keeps everything on the same device - no CPU-GPU ping-pong.
    
    Args:
        bz: [H, W] Bz field tensor on device
        xy_norm: [P, 2] normalized (x, y) coordinates in [-1, 1]
    
    Returns:
        values: [P, 1] interpolated Bz values
    """
    H, W = bz.shape
    device = bz.device
    
    # Convert from [-1, 1] to pixel coordinates
    x = (xy_norm[:, 0] + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[:, 1] + 1.0) * 0.5 * (H - 1)
    
    # Get integer coordinates
    x0 = x.floor().long().clamp(0, W - 2)
    y0 = y.floor().long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Compute interpolation weights
    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)
    
    # Gather values using advanced indexing (stays on device)
    v00 = bz[y0, x0].unsqueeze(-1)
    v01 = bz[y1, x0].unsqueeze(-1)
    v10 = bz[y0, x1].unsqueeze(-1)
    v11 = bz[y1, x1].unsqueeze(-1)
    
    # Bilinear interpolation
    v0 = v00 * (1 - wy) + v01 * wy
    v1 = v10 * (1 - wy) + v11 * wy
    return v0 * (1 - wx) + v1 * wx


def _resample_gt_from_frames(
    frames: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """
    Recompute ground-truth Bz samples after geometric augmentations.
    
    Args:
        frames: [T, H, W] Bz frames
        coords: [T, P, 3] normalized coordinates
        
    Returns:
        gt_bz: [T, P, 1] bilinearly sampled values
    """
    T, P = coords.shape[0], coords.shape[1]
    device = frames.device
    gt = torch.zeros(T, P, 1, device=device, dtype=frames.dtype)
    for t in range(T):
        gt[t] = _bilinear_sample_bz_tensor(frames[t], coords[t, :, 0:2])
    return gt


# =============================================================================
# Helper Functions
# =============================================================================

def _linspace_times(t0_utc: pd.Timestamp, hours: int) -> list[pd.Timestamp]:
    """Generate hourly timestamps from (t0 - hours) to t0."""
    start = t0_utc - pd.Timedelta(hours=hours)
    return list(pd.date_range(start, t0_utc, freq="1h", inclusive="both"))


def _coords_grid(T: int, P: int, device: torch.device, sample_seed: int = None) -> torch.Tensor:
    """Generate random collocation coordinates with deterministic per-sample seeding."""
    if sample_seed is not None:
        # Save current RNG state
        rng_state = torch.get_rng_state()
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state()
        elif device.type == "mps":
            mps_rng_state = torch.mps.get_rng_state() if hasattr(torch.mps, 'get_rng_state') else None
        
        # Use sample-specific seed for reproducible coordinates
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(sample_seed)
        elif device.type == "mps" and hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(sample_seed)
    
    xy = torch.rand(T, P, 2, device=device) * 2.0 - 1.0
    t = torch.linspace(-1.0, 1.0, T, device=device)[:, None, None].expand(T, P, 1)
    coords = torch.cat([xy, t], dim=-1)
    
    if sample_seed is not None:
        # Restore RNG state so training is unaffected
        torch.set_rng_state(rng_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state)
        elif device.type == "mps" and mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)
    
    return coords


# =============================================================================
# Main Dataset Class
# =============================================================================

class CachedWindowsDataset(Dataset):
    """
    Windows dataset with shared frame caching.
    
    Key features:
    - Preload option warms cache before DataLoader workers spawn
    - All tensors stay on CPU until collated, then move to device
    - Bilinear sampling uses pure tensor ops (no CPU-GPU ping-pong)
    - Returns strongly-typed WindowSample instead of Dict
    
    Usage:
        dataset = CachedWindowsDataset(..., preload=True)
        # Cache is now warm, workers will have copies
        loader = DataLoader(dataset, num_workers=4, ...)
    """
    
    def __init__(
        self,
        windows_df: pd.DataFrame,
        frames_meta_path: str,
        npz_root: str,
        target_px: int = 64,
        input_hours: int = 48,
        horizons: list[int] | None = None,
        P_per_t: int = 512,
        pil_top_pct: float = 0.15,
        training: bool = True,
        augment: bool = True,
        noise_std: float = 0.02,
        preload: bool = False,
    ):
        import time as _time
        import sys
        
        if horizons is None:
            horizons = [6, 12, 24]
        
        t0 = _time.time()
        self.df = windows_df.reset_index(drop=True)
        print(f"  [DEBUG] Reset index: {_time.time() - t0:.2f}s", flush=True)
        
        t0 = _time.time()
        self.meta = pd.read_parquet(frames_meta_path)
        print(f"  [DEBUG] Read parquet ({len(self.meta)} rows): {_time.time() - t0:.2f}s", flush=True)
        
        self.root = Path(npz_root)
        self.target_px = int(target_px)
        self.input_hours = int(input_hours)
        self.horizons = list(horizons)
        self.P_per_t = int(P_per_t)
        self.pil_top_pct = float(pil_top_pct)
        self.training = training
        self.augment = augment and training
        self.noise_std = noise_std

        # Prepare metadata index for fast lookup
        t0 = _time.time()
        self.meta["date_obs"] = pd.to_datetime(self.meta["date_obs"], utc=True)
        self.meta = self.meta.sort_values(["harpnum", "date_obs"]).reset_index(drop=True)
        print(f"  [DEBUG] Sort metadata: {_time.time() - t0:.2f}s", flush=True)
        
        # Build path lookup FAST using vectorized operations (not iterrows!)
        t0 = _time.time()
        self._path_index: dict[tuple[int, str], str] = {}
        
        # Vectorized path processing
        harps = self.meta["harpnum"].astype(int).values
        timestamps = self.meta["date_obs"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00").values
        paths = self.meta["frame_path"].str.replace("\\", "/", regex=False)
        paths = paths.str.replace("^frames/", "", regex=True).values
        print(f"  [DEBUG] Vectorize columns: {_time.time() - t0:.2f}s", flush=True)
        
        # Build dict in one pass with zip (much faster than iterrows)
        t0 = _time.time()
        root_str = str(self.root)
        for harp, ts, path in zip(harps, timestamps, paths):
            self._path_index[(harp, ts)] = f"{root_str}/{path}"
        print(f"  [DEBUG] Build path index: {_time.time() - t0:.2f}s", flush=True)
        
        print(f"Dataset ready: {len(self.df)} windows, {len(self._path_index)} frame paths indexed")
        
        # Preload cache if requested (MUST happen before DataLoader workers spawn)
        if preload:
            self._preload_cache()
    
    def _preload_cache(self) -> None:
        """Warm the frame cache by loading all referenced frames."""
        print(f"Preloading frame cache...")
        
        # Collect all unique frame paths needed
        all_paths: set[str] = set()
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            harp = int(row["harpnum"])
            t0 = pd.to_datetime(row["t0"], utc=True)
            times = _linspace_times(t0, self.input_hours)
            
            for ts in times:
                key = (harp, ts.isoformat())
                path = self._path_index.get(key)
                if path:
                    all_paths.add(path)
        
        # Load all frames into cache
        loaded = 0
        for path in all_paths:
            cache_key = f"{path}:{self.target_px}"
            if _FRAME_CACHE.get(cache_key) is None:
                frame = _load_and_preprocess_frame(path, self.target_px)
                if frame is not None:
                    _FRAME_CACHE.put(cache_key, frame)
                    loaded += 1
        
        print(f"Preloaded {loaded} frames into cache (total: {len(_FRAME_CACHE)})")

    def _get_frame(self, harp: int, ts: pd.Timestamp) -> Optional[np.ndarray]:
        """Get frame from cache or load from disk."""
        key = (harp, ts.isoformat())
        path = self._path_index.get(key)
        if path is None:
            return None
        
        cache_key = f"{path}:{self.target_px}"
        
        # Check cache first
        cached = _FRAME_CACHE.get(cache_key)
        if cached is not None:
            return cached
        
        # Load from disk and cache
        frame = _load_and_preprocess_frame(path, self.target_px)
        if frame is not None:
            _FRAME_CACHE.put(cache_key, frame)
        return frame

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single window sample.
        
        Returns dict for DataLoader compatibility, but all values are
        properly typed tensors.
        """
        row = self.df.iloc[idx]
        harp = int(row["harpnum"])
        t0 = pd.to_datetime(row["t0"], utc=True)
        times = _linspace_times(t0, self.input_hours)
        T = len(times)

        device = torch.device("cpu")
        #  Deterministic per-sample seed for reproducible coordinates
        sample_seed = harp * 1000000 + int(t0.timestamp() % 1000000)
        coords = _coords_grid(T, self.P_per_t, device, sample_seed=sample_seed)

        observed = np.zeros((T,), np.bool_)
        gt_bz = torch.zeros(T, self.P_per_t, 1, device=device)
        frames = torch.zeros(T, self.target_px, self.target_px, device=device)
        last_obs_bz: Optional[torch.Tensor] = None
        
        for ti, ts in enumerate(times):
            bz_np = self._get_frame(harp, ts)
            if bz_np is None:
                continue
            
            observed[ti] = True
            
            # Convert to tensor once, keep on device
            bz_tensor = torch.from_numpy(bz_np).to(device)
            if not torch.isfinite(bz_tensor).all():
                bz_tensor = torch.nan_to_num(bz_tensor, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Sample Bz at collocation points (pure tensor ops, no CPU-GPU bounce)
            xy = coords[ti, :, 0:2]
            gt = _bilinear_sample_bz_tensor(bz_tensor, xy)
            if not torch.isfinite(gt).all():
                gt = torch.nan_to_num(gt, nan=0.0, posinf=3.0, neginf=-3.0)
            gt_bz[ti] = gt
            
            frames[ti] = bz_tensor
            last_obs_bz = bz_tensor

        observed_mask = torch.from_numpy(observed)
        
        # Ensure at least one frame is marked observed
        if not observed.any():
            observed[0] = True
            observed_mask = torch.from_numpy(observed)

        # Classification labels
        labels = []
        for H in self.horizons:
            col = f"y_geq_M_{H}h"
            labels.append(float(bool(row[col])) if col in row.index else 0.0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # PIL mask and physics-based scalar features
        if last_obs_bz is not None:
            bz_np = last_obs_bz.numpy()
            pil_mask_np = pil_mask_from_bz(bz_np, top_percent=self.pil_top_pct)
            
            # Compute R-value and GWPIL (key flare predictors from literature)
            r_value = compute_r_value(bz_np, pil_mask_np)
            gwpil = compute_pil_gradient_weighted_length(bz_np, pil_mask_np)
        else:
            pil_mask_np = np.zeros((self.target_px, self.target_px), dtype=np.uint8)
            r_value = 0.0
            gwpil = 0.0
        pil_mask_tensor = torch.from_numpy(pil_mask_np.astype(np.float32))
        
        # Compute observation coverage
        obs_coverage = float(observed.sum()) / T
        
        # Build scalar features tensor
        scalars = torch.tensor([
            r_value,
            gwpil,
            obs_coverage,
            float(observed.sum()),  # frame_count_in_window
        ], dtype=torch.float32)

        # Data augmentation
        if self.augment:
            geom_changed = False
            # Horizontal flip
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-1])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-1])
                coords = coords.clone()
                coords[..., 0] = -coords[..., 0]
                geom_changed = True
            
            # Vertical flip
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-2])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-2])
                coords = coords.clone()
                coords[..., 1] = -coords[..., 1]
                geom_changed = True
            
            # Random rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                frames = torch.rot90(frames, k, dims=[-2, -1])
                pil_mask_tensor = torch.rot90(pil_mask_tensor, k, dims=[-2, -1])
                coords = coords.clone()
                for _ in range(k):
                    old_x, old_y = coords[..., 0].clone(), coords[..., 1].clone()
                    coords[..., 0] = -old_y
                    coords[..., 1] = old_x
                geom_changed = True
            
            if geom_changed:
                gt_bz = _resample_gt_from_frames(frames, coords)
            
            # Add noise
            if self.noise_std > 0:
                frames = frames + torch.randn_like(frames) * self.noise_std
                gt_bz = gt_bz + torch.randn_like(gt_bz) * (self.noise_std * 0.5)
        
        return {
            "coords": coords,
            "gt_bz": gt_bz,
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels_tensor,
            "pil_mask": pil_mask_tensor,
            "scalars": scalars,
        }


def clear_frame_cache() -> None:
    """Clear the global frame cache."""
    _FRAME_CACHE.clear()


def get_cache_size() -> int:
    """Get number of frames in cache."""
    return len(_FRAME_CACHE)
