# src/models/pinn/collocation.py
from __future__ import annotations
import numpy as np
import torch
from typing import Optional

def sample_xy_from_mask(mask: Optional[np.ndarray], n: int, device: str = "cpu") -> torch.Tensor:
    """mask: [H,W] {0,1}; returns (x,y) in [-1,1]^2"""
    if mask is None:
        xy = torch.rand(n, 2, device=device) * 2.0 - 1.0
        return xy
    H, W = mask.shape
    idx = np.flatnonzero(mask.reshape(-1) > 0.5)
    if idx.size == 0:
        # uniform fallback
        xy = torch.rand(n, 2, device=device) * 2.0 - 1.0
        return xy
    choice = np.random.choice(idx, size=n, replace=True)
    ys, xs = np.unravel_index(choice, (H, W))
    xs = (xs / (W - 1)) * 2.0 - 1.0
    ys = (ys / (H - 1)) * 2.0 - 1.0
    xy = np.stack([xs, ys], axis=-1).astype(np.float32)
    return torch.from_numpy(xy).to(device)

def mix_pil_uniform(
    H: int, W: int, alpha: float, n_points: int, pil_mask: np.ndarray | None, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      coords: [N,3] in [-1,1]^3 (time uniform over [-1,1])
      p:      [N,1] mixture pdf (spatial * temporal), used for importance weights
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    n_pil = int(round(alpha * n_points))
    n_uni = n_points - n_pil

    xy_pil = sample_xy_from_mask(pil_mask if pil_mask is not None else np.zeros((H,W), np.float32), n_pil, device)
    xy_uni = torch.rand(n_uni, 2, device=device) * 2.0 - 1.0
    xy = torch.cat([xy_pil, xy_uni], dim=0)
    t  = torch.rand(n_points, 1, device=device) * 2.0 - 1.0
    coords = torch.cat([xy, t], dim=-1)

    # pdf: p = α * p_pil + (1-α) * p_uni; treat both approx uniform over their supports
    area_xy = 4.0  # square [-1,1]^2
    p_uni_xy = 1.0 / area_xy
    # conservative fallback -> 1/area if mask empty
    p_pil_xy = p_uni_xy
    p_xy = alpha * p_pil_xy + (1 - alpha) * p_uni_xy
    p_t  = 0.5  # uniform over [-1,1]
    p = torch.full((n_points, 1), p_xy * p_t, device=device)
    return coords, p

def clip_and_renorm_importance(
    p: torch.Tensor, 
    clip_quantile: float = 0.99,
    max_weight_ratio: float = 100.0,  # Hard cap on max/min ratio
    min_ess_fraction: float = 0.1,  # Minimum ESS as fraction of N
) -> tuple[torch.Tensor, float]:
    """
    Clip and renormalize importance weights to emphasize high-probability regions.
    
    FIXED: Previously computed 1/p which INVERTED the weighting, emphasizing
    quiet-sun regions instead of PIL. Now uses p directly so that PIL regions
    (high p) get higher weights in the physics loss.
    
    IMPROVED: Added ESS monitoring to prevent weights from becoming too extreme.
    If ESS drops below threshold, automatically increases clipping.
    
    Args:
        p: [N, 1] probability/importance values (higher = more important)
        clip_quantile: Quantile for soft clipping
        max_weight_ratio: Hard cap on max/min weight ratio
        min_ess_fraction: Minimum ESS as fraction of N (0.1 = at least 10% effective samples)
        
    Returns:
        w_tilde: [N, 1] normalized importance weights (mean = 1)
        threshold: float, the clipping threshold used
    """
    N = p.shape[0]
    min_ess = max(10, int(N * min_ess_fraction))  # At least 10 effective samples
    
    # Use p directly (NOT 1/p) so high-probability regions get higher weights
    w = p.clone()
    
    with torch.no_grad():
        # Iteratively adjust clipping until ESS is acceptable
        current_quantile = clip_quantile
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Soft clip by quantile to prevent extreme values
            # FIXED: Avoid torch.quantile on MPS - use sorted-based approximation
            w_sorted = torch.sort(w.flatten()).values
            idx = int(float(current_quantile) * (w_sorted.numel() - 1))
            thr = w_sorted[idx]
            
            # Hard clip to prevent extreme ratios
            min_w = w.min().clamp(min=1e-12)
            hard_thr = min_w * max_weight_ratio
            thr = min(thr, hard_thr)
            
            w_clipped = torch.minimum(w, thr)
            
            # Renormalize so mean = 1
            w_tilde = w_clipped / w_clipped.mean().clamp_min(1e-12)
            
            # Compute ESS
            w_normalized = w_tilde / w_tilde.sum()
            current_ess = 1.0 / (w_normalized ** 2).sum()
            
            if current_ess >= min_ess:
                break
            
            # ESS too low - increase clipping (reduce variance)
            # Lower quantile = more aggressive clipping
            current_quantile = max(0.5, current_quantile - 0.1)
            max_weight_ratio = max(10.0, max_weight_ratio * 0.5)
    
    # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
    return w_tilde, float(thr.detach().cpu())

def ess(weights: torch.Tensor) -> float:
    """Effective sample size for normalized weights w̃."""
    # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
    w = weights / weights.sum()
    return float(1.0 / (w**2).sum().detach().cpu())

