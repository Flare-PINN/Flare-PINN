# src/models/pinn/latent_sampling.py
"""
2nd-order differentiable latent sampling for hybrid PINN.

Replaces grid_sample to enable physics loss to train the CNN encoder.
PyTorch's grid_sample doesn't support double backward, which breaks
the weak-form MHD residuals that need ∂²B/∂x∂y.

Usage:
    from models.pinn.latent_sampling import sample_latent_soft_bilinear
    Ls = sample_latent_soft_bilinear(L, xy_norm)  # [N,C,H,W], [N,P,2] -> [N,P,C]
"""
import torch
import torch.nn.functional as F

_EPS = 1e-6

def _to_pix_coords(xy_norm, H, W):
    """Convert normalized coords [-1,1] to pixel coordinates [0, H-1] x [0, W-1]."""
    x = (xy_norm[..., 0].clamp(-1+_EPS, 1-_EPS) + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[..., 1].clamp(-1+_EPS, 1-_EPS) + 1.0) * 0.5 * (H - 1)
    return x, y

@torch.no_grad()
def _safe_shape(L, xy_norm):
    """Validate input shapes."""
    assert L.dim() == 4, f"Expected L [N,C,H,W], got {list(L.shape)}"
    assert xy_norm.shape[-1] == 2, "xy_norm must be [..., 2]"
    return L.shape

def sample_latent_nearest(L: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor latent sampling (piecewise constant w.r.t coords).
    
    - Gradients flow to L (encoder), NOT to coords.
    - 2nd-order grads w.r.t coords avoid sampler path entirely.
    - Fastest option, but may have small artifacts at pixel boundaries.
    
    Args:
        L: [N,C,H,W] - Latent feature map from encoder
        xy_norm: [N,P,2] - Normalized coordinates in [-1,1]
    
    Returns:
        sampled: [N,P,C] - Sampled latent features at each point
    """
    N, C, H, W = _safe_shape(L, xy_norm)
    P = xy_norm.size(1)
    x, y = _to_pix_coords(xy_norm, H, W)                         # [N,P]
    xi = x.round().long().clamp(0, W-1)                          # [N,P]
    yi = y.round().long().clamp(0, H-1)                          # [N,P]

    # Use advanced indexing instead of one-hot (much more memory efficient)
    # Create batch indices for gathering
    batch_idx = torch.arange(N, device=L.device)[:, None].expand(N, P)  # [N,P]
    
    # L is [N,C,H,W], we want L[n, :, yi[n,p], xi[n,p]] for each (n,p)
    # Reshape L to [N,C,H,W] and gather using advanced indexing
    out = L[batch_idx, :, yi, xi]  # [N,P,C]
    
    return out

def sample_latent_soft_bilinear(L: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Manual bilinear sampler with smooth weights.
    
    - Fully differentiable to 2nd order w.r.t. coords almost everywhere.
    - No autograd limitations of grid_sample double-backward.
    - Smooth interpolation like standard bilinear sampling.
    
    This is the RECOMMENDED sampler for hybrid PINN training.
    
    Args:
        L: [N,C,H,W] - Latent feature map from encoder
        xy_norm: [N,P,2] - Normalized coordinates in [-1,1]
    
    Returns:
        sampled: [N,P,C] - Bilinearly interpolated latent features
        
    Note:
        Gradients flow cleanly to both L (trains encoder) and coords
        (enables physics loss with second-order derivatives).
    """
    N, C, H, W = _safe_shape(L, xy_norm)
    P = xy_norm.size(1)
    x, y = _to_pix_coords(xy_norm, H, W)                         # [N,P]
    x0 = torch.floor(x).long().clamp(0, W-2); x1 = x0 + 1
    y0 = torch.floor(y).long().clamp(0, H-2); y1 = y0 + 1

    # Fractional parts (gradients flow through x,y; floor has zero grad)
    fx = x - x0.float()                                          # [N,P]
    fy = y - y0.float()                                          # [N,P]

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w01 = (1 - fx) * fy
    w10 = fx * (1 - fy)
    w11 = fx * fy                                                # all [N,P]

    # Use advanced indexing (much more memory efficient than one-hot)
    batch_idx = torch.arange(N, device=L.device)[:, None].expand(N, P)  # [N,P]
    
    # Gather at 4 corners using advanced indexing
    v00 = L[batch_idx, :, y0, x0]  # [N,P,C]
    v01 = L[batch_idx, :, y1, x0]  # [N,P,C]
    v10 = L[batch_idx, :, y0, x1]  # [N,P,C]
    v11 = L[batch_idx, :, y1, x1]  # [N,P,C]

    # Weighted sum
    out = (w00.unsqueeze(-1) * v00 + w01.unsqueeze(-1) * v01 +
           w10.unsqueeze(-1) * v10 + w11.unsqueeze(-1) * v11)   # [N,P,C]
    
    # FIXED: Handle NaN/Inf that can occur from extreme L values
    # Use any() to avoid MPS synchronization issues
    with torch.no_grad():
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return out

