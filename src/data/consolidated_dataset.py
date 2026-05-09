# src/data/consolidated_dataset.py
"""
Dataset for loading consolidated per-HARP frame bundles.

MUCH faster than loading individual NPZ files:
- 1 file open per HARP instead of ~1000
- Pre-processed and normalized
- Float16 storage (smaller, faster to load)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .pil_mask import (
    pil_mask_from_bz,
    compute_r_value,
    compute_pil_gradient_weighted_length,
)


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


def _bilinear_sample(img: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation of field at coordinates using F.grid_sample.
    
    Args:
        img: [C, H, W] or [H, W] field
        xy_norm: [P, 2] normalized coordinates in [-1, 1]
    Returns:
        [P, C] sampled values
    """
    if img.ndim == 2:
        img = img.unsqueeze(0)  # [1, H, W]
    
    C, H, W = img.shape
    P = xy_norm.shape[0]
    
    img_batch = img.unsqueeze(0)  # [1, C, H, W]
    grid = xy_norm.view(1, 1, P, 2)  # [1, 1, P, 2]
    
    sampled = F.grid_sample(
        img_batch, 
        grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )  # [1, C, 1, P]
    
    return sampled.squeeze(0).squeeze(1).permute(1, 0)  # [P, C]


def _resample_gt_from_frames(frames: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Recompute ground-truth samples after geometric augmentations."""
    T, P = coords.shape[0], coords.shape[1]
    C = frames.shape[1]
    device = frames.device
    gt = torch.zeros(T, P, C, device=device, dtype=frames.dtype)
    for t in range(T):
        gt[t] = _bilinear_sample(frames[t], coords[t, :, 0:2])
    return gt


# =============================================================================
# SOTA Feature Computation (Literature: Liu+2017, Sun+2022, Bobra+2015)
# =============================================================================

def _compute_pil_evolution_features(
    frames: torch.Tensor,
    observed: np.ndarray,
    top_pct: float = 0.15,
) -> list[float]:
    """
    Compute PIL evolution features over the observation window.
    
     OPTIMIZED: Uses fast gradient computation, skips expensive morphology ops.
    
    Returns 8 features:
    - r_start, r_end, r_change_rate: R-value at window start/end and rate
    - gwpil_start, gwpil_end, gwpil_change_rate: GWPIL evolution  
    - pil_length_ratio: Ratio of end/start PIL length (log scale)
    - pil_gradient_ratio: Ratio of end/start max gradient at PIL (log scale)
    """
    obs_indices = np.where(observed)[0]
    
    if len(obs_indices) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    first_idx, last_idx = obs_indices[0], obs_indices[-1]
    time_span = max(last_idx - first_idx, 1)
    
    # Get Bz from first and last frames (Bz is channel 2)
    bz_first = frames[first_idx, 2].numpy().astype(np.float32)
    bz_last = frames[last_idx, 2].numpy().astype(np.float32)
    
    #  FAST PIL computation: skip expensive morphology and skeletonization
    # Just use gradient thresholding - captures 90% of the signal
    def _fast_pil_features(bz: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Compute PIL mask and features in one pass."""
        # Fast gradient using numpy (no scipy dependency)
        gy, gx = np.gradient(bz)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Threshold to top percent (fast percentile)
        flat_grad = grad_mag.ravel()
        threshold_idx = int((1.0 - top_pct) * len(flat_grad))
        threshold = np.partition(flat_grad, threshold_idx)[threshold_idx]
        pil_mask = (grad_mag >= threshold).astype(np.uint8)
        
        # R-value: log10(sum of |Bz| at PIL)
        pil_flux = np.abs(bz[pil_mask > 0]).sum() if pil_mask.sum() > 0 else 1e-10
        r_value = np.log10(pil_flux + 1e-10)
        
        # GWPIL: gradient-weighted PIL length
        gwpil = (grad_mag * pil_mask).sum() if pil_mask.sum() > 0 else 0.0
        
        # Max gradient at PIL
        max_grad = grad_mag[pil_mask > 0].max() if pil_mask.sum() > 0 else 1e-6
        
        return pil_mask, float(r_value), float(gwpil), float(max_grad)
    
    pil_first, r_start, gwpil_start, grad_start = _fast_pil_features(bz_first)
    pil_last, r_end, gwpil_end, grad_end = _fast_pil_features(bz_last)
    
    # Evolution rates
    r_change_rate = (r_end - r_start) / time_span
    gwpil_change_rate = (gwpil_end - gwpil_start) / time_span
    
    # Length and gradient ratios
    pil_len_start = max(float(pil_first.sum()), 1.0)
    pil_len_end = max(float(pil_last.sum()), 1.0)
    pil_length_ratio = np.log1p(pil_len_end / pil_len_start)
    
    pil_gradient_ratio = np.log1p(max(grad_end, 1e-6) / max(grad_start, 1e-6))
    
    return [
        r_start, r_end, r_change_rate,
        gwpil_start, gwpil_end, gwpil_change_rate,
        pil_length_ratio, pil_gradient_ratio,
    ]


def _compute_temporal_statistics(
    frames: torch.Tensor,
    observed: np.ndarray,
    top_pct: float = 0.15,
) -> list[float]:
    """
    Compute temporal statistics over all observed frames.
    
     OPTIMIZED: Uses fast gradient-only PIL computation (no morphology).
    
    Returns 4 features:
    - r_value_std: Temporal variability of R-value
    - gwpil_std: Temporal variability of GWPIL
    - r_value_trend: Linear trend slope of R-value
    - gwpil_trend: Linear trend slope of GWPIL
    """
    obs_indices = np.where(observed)[0]
    
    if len(obs_indices) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    
    #  FAST: Inline PIL computation without scipy/skimage
    def _fast_pil_stats(bz: np.ndarray) -> tuple[float, float]:
        """Fast R-value and GWPIL computation."""
        bz = bz.astype(np.float32)
        gy, gx = np.gradient(bz)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Fast percentile using partition
        flat_grad = grad_mag.ravel()
        threshold_idx = int((1.0 - top_pct) * len(flat_grad))
        threshold = np.partition(flat_grad, threshold_idx)[threshold_idx]
        pil_mask = grad_mag >= threshold
        
        if pil_mask.sum() == 0:
            return 0.0, 0.0
        
        r_value = np.log10(np.abs(bz[pil_mask]).sum() + 1e-10)
        gwpil = (grad_mag * pil_mask).sum()
        return float(r_value), float(gwpil)
    
    r_values = []
    gwpil_values = []
    times = []
    
    # Sample every 16th frame - 4 samples for 48h window is enough for trend
    sample_step = 16 if len(obs_indices) > 4 else max(1, len(obs_indices) // 3)
    sample_indices = obs_indices[::sample_step][:4]  # Cap at 4 samples
    
    for idx in sample_indices:
        bz = frames[idx, 2].numpy()
        r, g = _fast_pil_stats(bz)
        r_values.append(r)
        gwpil_values.append(g)
        times.append(float(idx))
    
    r_values = np.array(r_values)
    gwpil_values = np.array(gwpil_values)
    times = np.array(times)
    
    # Standard deviations (normalized by mean)
    r_mean = max(np.mean(r_values), 1e-6)
    gwpil_mean = max(np.mean(gwpil_values), 1e-6)
    r_value_std = float(np.std(r_values)) / r_mean
    gwpil_std = float(np.std(gwpil_values)) / gwpil_mean
    
    # Linear trends using endpoint difference
    if len(times) >= 2:
        dt = times[-1] - times[0]
        if dt > 0:
            r_value_trend = (r_values[-1] - r_values[0]) / dt
            gwpil_trend = (gwpil_values[-1] - gwpil_values[0]) / dt
        else:
            r_value_trend = 0.0
            gwpil_trend = 0.0
    else:
        r_value_trend = 0.0
        gwpil_trend = 0.0
    
    return [r_value_std, gwpil_std, r_value_trend, gwpil_trend]


# =============================================================================
# Consolidated HARP Cache
# =============================================================================

class HARPCache:
    """
    In-memory LRU cache for consolidated HARP files.
    Thread-safe for DataLoader with num_workers > 0.
    """
    def __init__(self, consolidated_dir: Path, max_harps: int = 500):
        from collections import OrderedDict
        import threading
        
        self.consolidated_dir = consolidated_dir
        self.max_harps = max_harps
        self._cache: OrderedDict[int, tuple[np.ndarray, dict[str, int]]] = OrderedDict()
        self._lock = threading.RLock()
        
        manifest_path = consolidated_dir / "manifest.pkl"
        if manifest_path.exists():
            with open(manifest_path, "rb") as f:
                self.manifest = pickle.load(f)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    def get_frame(self, harpnum: int, timestamp_iso: str) -> Optional[np.ndarray]:
        """Get a single frame from cache or disk (thread-safe)."""
        with self._lock:
            if harpnum not in self._cache:
                self._load_harp(harpnum)
            else:
                self._cache.move_to_end(harpnum)
            
            if harpnum not in self._cache:
                return None
            
            frames, ts_index = self._cache[harpnum]
            idx = ts_index.get(timestamp_iso)
            if idx is None:
                return None
            
            return frames[idx].astype(np.float32).copy()
    
    def _load_harp(self, harpnum: int) -> None:
        """Load a HARP bundle into cache."""
        if harpnum not in self.manifest:
            return
        
        while len(self._cache) >= self.max_harps:
            self._cache.popitem(last=False)
        
        harp_file = self.consolidated_dir / f"H{harpnum}.npz"
        if not harp_file.exists():
            return
        
        try:
            data = np.load(harp_file, allow_pickle=True)
            frames = data["frames"]
            timestamps = data["timestamps"]
            ts_index = {str(ts): i for i, ts in enumerate(timestamps)}
            self._cache[harpnum] = (frames, ts_index)
        except MemoryError:
            # OOM is critical - re-raise
            raise
        except Exception as e:
            # Log warning for other errors (disk corruption, missing keys, etc.)
            import warnings
            warnings.warn(f"Failed to load HARP {harpnum}: {type(e).__name__}: {e}")
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# =============================================================================
# Main Dataset
# =============================================================================

class ConsolidatedWindowsDataset(Dataset):
    """
    Fast dataset using consolidated per-HARP frame bundles.
    
    Now includes SOTA scalar features:
    - 4 basic: r_value, gwpil, obs_coverage, frame_count
    - 8 PIL evolution: r_start/end/rate, gwpil_start/end/rate, length_ratio, grad_ratio
    - 4 temporal stats: r_std, gwpil_std, r_trend, gwpil_trend
    Total: 16 scalar features (when all enabled)
    
    Speed modes:
    - fast_mode=True: Disables expensive SOTA features for ~2x faster data loading
    - use_pil_evolution=False: Skip PIL evolution features
    - use_temporal_statistics=False: Skip temporal statistics
    """
    
    def __init__(
        self,
        windows_df: pd.DataFrame,
        consolidated_dir: str | Path,
        target_px: int = 64,
        input_hours: int = 48,
        horizons: list[int] | None = None,
        P_per_t: int = 512,
        pil_top_pct: float = 0.15,
        training: bool = True,
        augment: bool = True,
        noise_std: float = 0.02,
        max_cached_harps: int = 500,
        # Feature flags
        use_pil_evolution: bool = True,
        use_temporal_statistics: bool = True,
        # Speed optimization
        fast_mode: bool = False,  # Disables all expensive SOTA features
    ):
        if horizons is None:
            horizons = [6, 12, 24]
        
        self.df = windows_df.reset_index(drop=True)
        self.consolidated_dir = Path(consolidated_dir).expanduser()
        self.target_px = int(target_px)
        self.input_hours = int(input_hours)
        self.horizons = list(horizons)
        self.P_per_t = int(P_per_t)
        self.pil_top_pct = float(pil_top_pct)
        self.training = training
        self.augment = augment and training
        self.noise_std = noise_std
        
        # Feature flags - fast_mode disables expensive features
        self.fast_mode = fast_mode
        self.use_pil_evolution = use_pil_evolution and not fast_mode
        self.use_temporal_statistics = use_temporal_statistics and not fast_mode
        
        # Initialize HARP cache
        self.cache = HARPCache(self.consolidated_dir, max_harps=max_cached_harps)
        
        # Compute expected scalar dimension
        n_scalars = 4  # base features
        if self.use_pil_evolution:
            n_scalars += 8
        if self.use_temporal_statistics:
            n_scalars += 4
        self.n_scalars = n_scalars
        
        mode_str = " [FAST MODE]" if fast_mode else ""
        print(f"ConsolidatedDataset ready: {len(self.df)} windows, {len(self.cache.manifest)} HARPs, {n_scalars} scalar features{mode_str}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        gt_b = torch.zeros(T, self.P_per_t, 3, device=device)
        frames = torch.zeros(T, 3, self.target_px, self.target_px, device=device)
        last_obs_bz: Optional[torch.Tensor] = None
        
        for ti, ts in enumerate(times):
            ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            frame_np = self.cache.get_frame(harp, ts_iso)
            
            if frame_np is None:
                continue
            
            observed[ti] = True
            frame_tensor = torch.from_numpy(frame_np).to(device)
            
            # Handle old 2D format (Bz only) vs new 3D format (Bx, By, Bz)
            if frame_tensor.ndim == 2:
                bz = frame_tensor.unsqueeze(0)
                frame_tensor = bz.repeat(3, 1, 1)  # [3, H, W]
            
            # Resize if needed
            if frame_tensor.shape[-1] != self.target_px:
                frame_tensor = F.interpolate(
                    frame_tensor.unsqueeze(0),
                    size=(self.target_px, self.target_px),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if not torch.isfinite(frame_tensor).all():
                frame_tensor = torch.nan_to_num(frame_tensor, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Sample at collocation points
            xy = coords[ti, :, 0:2]
            gt = _bilinear_sample(frame_tensor, xy)
            if not torch.isfinite(gt).all():
                gt = torch.nan_to_num(gt, nan=0.0, posinf=3.0, neginf=-3.0)
            gt_b[ti] = gt
            
            frames[ti] = frame_tensor
            last_obs_bz = frame_tensor[2]
        
        observed_mask = torch.from_numpy(observed)
        
        if not observed.any():
            observed[0] = True
            observed_mask = torch.from_numpy(observed)
        
        # Labels
        labels = []
        for H in self.horizons:
            col = f"y_geq_M_{H}h"
            labels.append(float(bool(row[col])) if col in row.index else 0.0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # PIL mask and basic scalar features
        if last_obs_bz is not None:
            bz_np = last_obs_bz.numpy()
            pil_mask_np = pil_mask_from_bz(bz_np, top_percent=self.pil_top_pct)
            r_value = compute_r_value(bz_np, pil_mask_np)
            gwpil = compute_pil_gradient_weighted_length(bz_np, pil_mask_np)
        else:
            pil_mask_np = np.zeros((self.target_px, self.target_px), dtype=np.uint8)
            r_value = 0.0
            gwpil = 0.0
        pil_mask_tensor = torch.from_numpy(pil_mask_np.astype(np.float32))
        
        obs_coverage = float(observed.sum()) / T
        
        # =====================================================================
        # BUILD SCALAR FEATURES (SOTA: 16 total)
        # =====================================================================
        scalar_list = [
            r_value,                    # 1. R-value (final frame)
            gwpil,                      # 2. GWPIL (final frame)
            obs_coverage,               # 3. Observation coverage fraction
            float(observed.sum()),      # 4. Frame count
        ]
        
        # PIL Evolution features (8)
        if self.use_pil_evolution:
            pil_evo = _compute_pil_evolution_features(frames, observed, self.pil_top_pct)
            scalar_list.extend(pil_evo)
        
        # Temporal statistics (4)
        if self.use_temporal_statistics:
            temp_stats = _compute_temporal_statistics(frames, observed, self.pil_top_pct)
            scalar_list.extend(temp_stats)
        
        scalars = torch.tensor(scalar_list, dtype=torch.float32)
        
        # Clamp extreme values for stability
        scalars = torch.clamp(scalars, -100.0, 100.0)
        scalars = torch.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Augmentation
        if self.augment:
            # Ensure coords are contiguous before modification to prevent memory issues
            coords = coords.contiguous()
            
            geom_changed = False
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-1])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-1])
                coords = coords.clone()
                coords[..., 0] = -coords[..., 0]
                geom_changed = True
            
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-2])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-2])
                coords = coords.clone()
                coords[..., 1] = -coords[..., 1]
                geom_changed = True
            
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
                # Re-sample ground truth at new coordinates
                gt_b = _resample_gt_from_frames(frames, coords)
            
            if self.noise_std > 0:
                frames = frames + torch.randn_like(frames) * self.noise_std
                gt_b = gt_b + torch.randn_like(gt_b) * (self.noise_std * 0.5)
        
        # Final safety check: ensure contiguous memory layout
        coords = coords.contiguous()
        frames = frames.contiguous()
        gt_b = gt_b.contiguous()
        
        return {
            "coords": coords,
            "gt_bz": gt_b,
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels_tensor,
            "pil_mask": pil_mask_tensor,
            "scalars": scalars,
        }
