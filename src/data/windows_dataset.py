# src/data/windows_dataset.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pil_mask import pil_mask_from_bz

def _linspace_times(t0_utc: pd.Timestamp, hours: int) -> List[pd.Timestamp]:
    # Inclusive window: [t0 - hours, t0] at 1h cadence → hours+1 frames expected
    start = t0_utc - pd.Timedelta(hours=hours)
    return list(pd.date_range(start, t0_utc, freq="1h", inclusive="both"))

def make_expected_grid_times(t0: pd.Timestamp, input_hours: int) -> np.ndarray:
    return np.array(_linspace_times(t0, input_hours), dtype="datetime64[ns]")

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
    
    xy = torch.rand(T, P, 2, device=device) * 2.0 - 1.0  # [-1,1]^2
    t = torch.linspace(-1.0, 1.0, T, device=device)[:, None, None].expand(T, P, 1)
    coords = torch.cat([xy, t], dim=-1)  # [T,P,3]
    # Don't set requires_grad here - will be set in training loop after batching
    
    if sample_seed is not None:
        # Restore RNG state so training is unaffected
        torch.set_rng_state(rng_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state)
        elif device.type == "mps" and mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)
    
    return coords

def _bilinear_sample_field(field: np.ndarray, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Sample from [C, H, W] or [H, W] field at normalized coordinates.
    
    Args:
        field: [C,H,W] or [H,W] numpy array
        xy_norm: [N,2] torch tensor in [-1,1]^2
        
    Returns:
        sampled: [N,C] torch tensor
    """
    if field.ndim == 2:
        field = field[None, :, :]  # [1,H,W]
        
    C, H, W = field.shape
    x = (xy_norm[:, 0] + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[:, 1] + 1.0) * 0.5 * (H - 1)
    x0 = torch.clamp(x.floor().long(), 0, W - 2)
    y0 = torch.clamp(y.floor().long(), 0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wx = (x - x0.float()).unsqueeze(-1)  # [N,1]
    wy = (y - y0.float()).unsqueeze(-1)  # [N,1]
    
    # Gather 4 neighbors for each channel
    # field is numpy, indices are torch on CPU (usually) or device
    # We'll convert gather results to torch
    
    def gather(ix, iy):
        # ix, iy are [N]. We want field[:, iy, ix] -> [C, N] -> transpose to [N, C]
        # Note: numpy advanced indexing
        f_val = field[:, iy.cpu().numpy(), ix.cpu().numpy()]
        return torch.from_numpy(f_val.T).to(xy_norm.device).float()
        
    v00 = gather(x0, y0) # [N,C]
    v01 = gather(x0, y1)
    v10 = gather(x1, y0)
    v11 = gather(x1, y1)
    
    v0 = v00 * (1 - wy) + v01 * wy
    v1 = v10 * (1 - wy) + v11 * wy
    v = v0 * (1 - wx) + v1 * wx
    return v  # [N,C]

class WindowsDataset(Dataset):
    """
    Yields batches for PINN:
      - coords [T,P,3] in [-1,1]^3
      - gt_field [T,P,C] sampled from frames on observed times only (mask provided)
      - observed_mask [T]
      - labels [n_horizons] (y_geq_M_*h)
      - pil_mask [H,W] (binary) computed from ∣∇Bz∣ on the last observed frame
      
    Supports data augmentation in training mode:
      - Random horizontal flip (with field sign flip for correct physics)
      - Random 90-degree rotations
      - Gaussian noise injection
    """
    def __init__(
        self,
        windows_df: pd.DataFrame,
        frames_meta_path: str,
        npz_root: str,
        target_px: int = 256,
        input_hours: int = 48,
        horizons: List[int] = [6,12,24],
        P_per_t: int = 1024,
        pil_top_pct: float = 0.15,
        training: bool = True,
        augment: bool = True,
        noise_std: float = 0.02,
        components: List[str] = ["Bz"],
    ):
        self.df = windows_df.reset_index(drop=True)
        self.meta = pd.read_parquet(frames_meta_path)
        self.root = Path(npz_root)
        self.target_px = int(target_px)
        self.input_hours = int(input_hours)
        self.horizons = list(horizons)
        self.P_per_t = int(P_per_t)
        self.pil_top_pct = float(pil_top_pct)
        self.training = training
        self.augment = augment and training  # Only augment during training
        self.noise_std = noise_std
        self.components = list(components)
        self.n_components = len(self.components)

        # Minimal index for faster lookup
        self.meta["date_obs"] = pd.to_datetime(self.meta["date_obs"], utc=True)
        self.meta = self.meta.sort_values(["harpnum","date_obs"]).reset_index(drop=True)
        
        # Suppress timezone warnings
        import warnings
        warnings.filterwarnings('ignore', message='.*timezones available.*')

    def __len__(self) -> int:
        return len(self.df)

    def _frames_for_window(self, harpnum: int, t0: pd.Timestamp) -> Tuple[List[pd.Timestamp], List[Optional[str]]]:
        expect_times = _linspace_times(t0, self.input_hours)
        g = self.meta[self.meta["harpnum"]==int(harpnum)]
        g_times = g["date_obs"].to_numpy().astype("datetime64[ns]")
        paths = []
        for ts in expect_times:
            # nearest neighbor within 90 minutes, else None
            ts_np = np.datetime64(ts)
            if g_times.size == 0:
                paths.append(None)
                continue
            idx = np.searchsorted(g_times, ts_np, side="left")
            idx = np.clip(idx, 0, len(g_times)-1)
            nearest = g.iloc[int(idx)]
            dt = abs(pd.Timestamp(nearest["date_obs"]) - ts)
            if dt <= pd.Timedelta(minutes=90):
                paths.append(str(nearest["frame_path"]))
            else:
                paths.append(None)
        return expect_times, paths

    def _load_frame_components(self, frame_path_rel: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load all configured components from file.
        Returns:
            frame: [C, H, W] normalized field
            raw_bz_for_pil: [H, W] raw Bz for PIL mask (if available)
        """
        try:
            frame_path_rel = frame_path_rel.replace("\\", "/")
            if frame_path_rel.startswith("frames/"):
                frame_path_rel = frame_path_rel[7:]
            p = self.root / frame_path_rel
            npz = np.load(p, allow_pickle=False)
            
            channels = []
            raw_bz = None
            
            # Helper to process a single component
            def get_comp(name):
                # Map common names
                key = name
                if name == "Bz" and "Br" in npz and "Bz" not in npz:
                    key = "Br"  # Fallback
                
                if key not in npz:
                    return np.zeros((self.target_px, self.target_px), dtype=np.float32)
                
                val = npz[key].astype(np.float32)
                
                # Resize if needed
                if val.shape != (self.target_px, self.target_px):
                    # simple center crop/pad
                    H, W = val.shape
                    out = np.zeros((self.target_px, self.target_px), np.float32)
                    y0 = max(0, (self.target_px - H)//2); x0 = max(0, (self.target_px - W)//2)
                    y1 = y0 + min(H, self.target_px); x1 = x0 + min(W, self.target_px)
                    sy0 = max(0, (H - self.target_px)//2); sx0 = max(0, (W - self.target_px)//2)
                    sy1 = sy0 + (y1 - y0); sx1 = sx0 + (x1 - x0)
                    out[y0:y1, x0:x1] = val[sy0:sy1, sx0:sx1]
                    val = out
                
                # Normalize
                if not np.isfinite(val).all():
                    val[~np.isfinite(val)] = 0.0
                    
                data_range = np.abs(val).max()
                if data_range > 10: # Raw Gauss
                    val = val / 2000.0
                elif data_range > 0:
                    val = val / max(data_range, 5.0)
                
                return np.clip(val, -1.5, 1.5)

            for c in self.components:
                channels.append(get_comp(c))
            
            # Keep raw Bz for PIL if possible (re-load without heavy clip/norm if needed, 
            # but for now just use the processed one)
            # Actually PIL mask uses gradient of normalized Bz usually, which is fine.
            # We'll return the processed Bz component if it exists in components list
            if "Bz" in self.components:
                raw_bz = channels[self.components.index("Bz")]
            else:
                # Try to load just Bz for PIL
                raw_bz = get_comp("Bz")

            return np.stack(channels, axis=0), raw_bz

        except Exception as e:
            return None, None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        harp = int(row["harpnum"])
        t0 = pd.to_datetime(row["t0"], utc=True)
        times, paths = self._frames_for_window(harp, t0)
        T = len(times)

        device = torch.device("cpu")  # tensors are moved in train loop
        #  Deterministic per-sample seed for reproducible coordinates
        sample_seed = harp * 1000000 + int(t0.timestamp() % 1000000)
        coords = _coords_grid(T, self.P_per_t, device, sample_seed=sample_seed)  # [T,P,3]

        # Build observed mask & sample gt field & load full frames
        observed = np.zeros((T,), np.bool_)
        gt_field = torch.zeros(T, self.P_per_t, self.n_components, device=device)
        frames = torch.zeros(T, self.n_components, self.target_px, self.target_px, device=device)
        last_obs_bz = None
        
        for ti, p in enumerate(paths):
            if p is None:
                continue
            
            frame_stack, bz_for_pil = self._load_frame_components(p)
            if frame_stack is None:
                continue
                
            observed[ti] = True
            
            # Sample P points at this time index
            xy = coords[ti, :, 0:2]  # [P,2]
            gt = _bilinear_sample_field(frame_stack, xy)  # [P,C]
            
            # Safety check
            if not torch.isfinite(gt).all():
                gt = torch.nan_to_num(gt, nan=0.0, posinf=3.0, neginf=-3.0)
            
            gt_field[ti] = gt
            
            # Store full frame
            frame_tensor = torch.from_numpy(frame_stack).to(device)
            if not torch.isfinite(frame_tensor).all():
                frame_tensor = torch.nan_to_num(frame_tensor, nan=0.0, posinf=3.0, neginf=-3.0)
            frames[ti] = frame_tensor
            
            if bz_for_pil is not None:
                last_obs_bz = bz_for_pil

        observed_mask = torch.from_numpy(observed)
        
        # Skip samples with NO observed frames
        if not observed.any():
            observed[0] = True
            # Dummy fill
            gt_field[0] = 0.0
            frames[0] = 0.0
            observed_mask = torch.from_numpy(observed)

        # Labels
        labels = []
        for H in self.horizons:
            col = f"y_geq_M_{H}h"
            labels.append(float(bool(row[col])) if col in row else 0.0)
        labels = torch.tensor(labels).float()

        # PIL mask from last observed frame
        if last_obs_bz is not None:
            pil_mask = pil_mask_from_bz(last_obs_bz, top_percent=self.pil_top_pct)
        else:
            pil_mask = np.zeros((self.target_px, self.target_px), dtype=np.uint8)

        # ============ DATA AUGMENTATION ============
        if self.augment:
            # Random horizontal flip 
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-1])  # Flip W
                
                # Flip components?
                # Bx -> -Bx, By -> By, Bz -> -Bz (for polarity flip)
                # This depends on physics. 
                # Standard convention: flip x-axis -> Bx changes sign, Bz changes sign (polarity flip), By unchanged?
                # Or just pure image flip?
                # Existing code did: gt_bz = -gt_bz. 
                # Let's assume standard polarity flip applies to vertical field.
                # For Bx/By it's tricky without knowing if they are vector components or scalars.
                # Safest minimal change: Flip Bz sign if present.
                
                # Find Bz index
                if "Bz" in self.components:
                    idx = self.components.index("Bz")
                    frames[:, idx] = -frames[:, idx]
                    gt_field[..., idx] = -gt_field[..., idx]
                
                pil_mask = np.flip(pil_mask, axis=1).copy()
                coords = coords.clone()
                coords[..., 0] = -coords[..., 0]
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-2])  # Flip H
                if "Bz" in self.components:
                    idx = self.components.index("Bz")
                    frames[:, idx] = -frames[:, idx]
                    gt_field[..., idx] = -gt_field[..., idx]
                pil_mask = np.flip(pil_mask, axis=0).copy()
                coords = coords.clone()
                coords[..., 1] = -coords[..., 1]
            
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                frames = torch.rot90(frames, k, dims=[-2, -1])
                pil_mask = np.rot90(pil_mask, k).copy()
                coords = coords.clone()
                for _ in range(k):
                    old_x = coords[..., 0].clone()
                    old_y = coords[..., 1].clone()
                    coords[..., 0] = -old_y
                    coords[..., 1] = old_x
            
            # Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(frames) * self.noise_std
                frames = frames + noise
                gt_field = gt_field + torch.randn_like(gt_field) * (self.noise_std * 0.5)
        
        return {
            "coords": coords,                  # [T,P,3]
            "gt_bz": gt_field,                 # [T,P,C] (renamed logic, kept key for compat or update?)
                                               # Wait, key 'gt_bz' is baked into train.py. 
                                               # Better to keep key 'gt_bz' but it now holds [T,P,C] field?
                                               # Or rename to 'gt_field' and update train.py?
                                               # I'll rename to 'gt_bz' for minimal friction in train loop for now, 
                                               # but ideally 'gt_field'. 
                                               # Actually, train.py just passes it to model.
            "frames": frames,                  # [T,C,H,W]
            "observed_mask": observed_mask,    # [T]
            "labels": labels,                  # [n_horizons]
            "pil_mask": pil_mask,              # np.ndarray[H,W]
        }

