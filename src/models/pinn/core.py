# src/models/pinn/core.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from dataclasses import dataclass

# ---------------- Fourier Features ---------------- #

class FourierFeatures(nn.Module):
    """
    Fourier features for (x,y,t) in [-1,1]^3 with FULL frequency set always computed.
    Call set_alpha(0..1) during training to scale higher frequencies (soft annealing).
    
    OPTIMIZED: Pre-registers frequency buffers for faster computation.
    Output order: [x, sin_f1, cos_f1, sin_f2, cos_f2, ...] (compatible with original)
    """
    def __init__(self, in_dim: int = 3, max_log2_freq: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.max_log2_freq = max_log2_freq
        self.register_buffer("_alpha", torch.tensor(1.0))
        
        # Pre-compute frequencies and register as buffer (faster than recomputing)
        freqs = torch.tensor([math.pi * (2.0 ** k) for k in range(1, max_log2_freq + 1)])
        self.register_buffer("_freqs", freqs)  # [max_log2_freq]
        
        # Pre-compute weight thresholds for annealing
        thresholds = torch.tensor([(k-1) / max_log2_freq for k in range(1, max_log2_freq + 1)])
        self.register_buffer("_thresholds", thresholds)

    def set_alpha(self, a: float) -> None:
        """Set annealing parameter (0=soft start, 1=full freqs)"""
        a = float(max(0.0, min(1.0, a)))
        self._alpha.fill_(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        # Vectorized Fourier features (much faster than loop)
        alpha = self._alpha  # Keep as tensor for vectorized ops
        
        # Compute weights: clamp((alpha - threshold) * max_log2_freq, 0, 1)
        weights = ((alpha - self._thresholds) * self.max_log2_freq).clamp(0.0, 1.0)  # [max_log2_freq]
        
        # Expand x for broadcasting: [..., in_dim, 1] * [max_log2_freq] -> [..., in_dim, max_log2_freq]
        x_expanded = x.unsqueeze(-1)  # [..., in_dim, 1]
        scaled_x = x_expanded * self._freqs  # [..., in_dim, max_log2_freq]
        
        # Compute sin and cos features
        sin_feats = torch.sin(scaled_x) * weights  # [..., in_dim, max_log2_freq]
        cos_feats = torch.cos(scaled_x) * weights  # [..., in_dim, max_log2_freq]
        
        # Collect outputs in original order: [x, sin_f1, cos_f1, sin_f2, cos_f2, ...]
        # sin_feats[..., k] and cos_feats[..., k] each have shape [..., in_dim]
        outs = [x]
        for k in range(self.max_log2_freq):
            outs.append(sin_feats[..., k])  # [..., in_dim]
            outs.append(cos_feats[..., k])  # [..., in_dim]
        
        return torch.cat(outs, dim=-1)

def fourier_out_dim(in_dim: int = 3, max_log2_freq: int = 5) -> int:
    return in_dim + 2 * max_log2_freq * in_dim

# ---------------- Backbone (coordinate MLP) ---------------- #

class ResBlock(nn.Module):
    """Residual block for deeper gradient flow with proper initialization."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        # Initialize the second linear layer with smaller weights for stable residual
        # This ensures the residual starts close to identity and learns deviations
        with torch.no_grad():
            nn.init.xavier_uniform_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)
            # Scale down the second layer for identity-like initialization
            nn.init.xavier_uniform_(self.net[2].weight, gain=0.1)
            nn.init.zeros_(self.net[2].bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int) -> nn.Sequential:
    """
    Constructs a ResNet-style MLP for better trainability at depth.
    
    Args:
        in_dim: Input dimension
        hidden: Hidden dimension
        out_dim: Output dimension
        layers: Number of hidden layers (approximate, will be converted to blocks)
    """
    # Input projection
    input_layer = nn.Linear(in_dim, hidden)
    mods = [input_layer, nn.SiLU()]
    
    # Residual blocks
    # Each block contains 2 layers. 
    # We replace (layers-1) plain layers with (layers-1)//2 blocks.
    n_blocks = max(1, (layers - 1) // 2)
    
    for _ in range(n_blocks):
        mods.append(ResBlock(hidden))
    
    # Handle odd number of layers (optional, just to stay close to requested depth)
    if (layers - 1) % 2 != 0:
        extra_layer = nn.Linear(hidden, hidden)
        mods.extend([extra_layer, nn.SiLU()])
        
    # Output projection - initialize with small weights for stable starting point
    output_layer = nn.Linear(hidden, out_dim)
    with torch.no_grad():
        # Small output initialization helps with initial gradient flow
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)
    mods.append(output_layer)
    
    return nn.Sequential(*mods)


@dataclass
class PINNBackboneOutput:
    """Standardized output from PINN backbone."""
    # Common fields
    A_z: torch.Tensor
    u_x: torch.Tensor
    u_y: torch.Tensor
    eta_raw: Optional[torch.Tensor] = None
    
    # Scalar mode specific
    B_z: Optional[torch.Tensor] = None
    
    # Vector mode specific
    B: Optional[torch.Tensor] = None        # Packed [N, 3]
    u: Optional[torch.Tensor] = None        # Packed [N, 2]
    B_x: Optional[torch.Tensor] = None
    B_y: Optional[torch.Tensor] = None
    
    def __getitem__(self, key: str) -> Optional[torch.Tensor]:
        """Backward compatibility for dict-like access."""
        return getattr(self, key)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)


class PINNBackbone(nn.Module):
    """
    Coordinate MLP with Fourier features.
    
    Supports two output modes:
    - Scalar mode (legacy): outputs A_z, B_z, u_x, u_y, [eta_raw]
    - Vector mode: outputs A_z, B (3-component), u (2-component), [eta_raw]
    
    Vector mode enables full 2.5D vector induction equation physics.
    """
    def __init__(
        self, 
        hidden: int = 384, 
        layers: int = 10,
        max_log2_freq: int = 5, 
        learn_eta: bool = False,
        vector_B: bool = False,
        hard_div_free: bool = False,
    ):
        """
        Args:
            hidden: Hidden dimension for MLP
            layers: Number of hidden layers
            max_log2_freq: Maximum Fourier frequency (log2)
            learn_eta: Learn spatially-varying resistivity
            vector_B: Output 3-component B field (Bx, By, Bz) instead of just Bz
            hard_div_free: Enforce div B = 0 by deriving B from A (requires vector_B=True)
        """
        super().__init__()
        self.ff = FourierFeatures(3, max_log2_freq)
        in_dim = fourier_out_dim(3, max_log2_freq)
        
        self.vector_B = vector_B
        self.learn_eta = learn_eta
        self.hard_div_free = hard_div_free
        
        # Compute output dimension:
        # - A_z: 1
        # - B: 3 (vector) or 1 (scalar)
        # - u: 2
        # - eta_raw: 1 (optional)
        if vector_B:
            if hard_div_free:
                # Hard div-free mode: Az(1) + Bz(1) + u(2) + [eta(1)]
                # Bx, By are derived from Az, so we don't output them directly
                out_dim = 4 + (1 if learn_eta else 0)
            else:
                # Vector mode: Az(1) + B(3) + u(2) + [eta(1)] = 6 or 7
                out_dim = 6 + (1 if learn_eta else 0)
        else:
            # Scalar mode: Az(1) + Bz(1) + ux(1) + uy(1) + [eta(1)] = 4 or 5
            out_dim = 4 + (1 if learn_eta else 0)
        
        self.net = _mlp(in_dim, hidden, out_dim, layers)

    def set_fourier_alpha(self, a: float) -> None:
        self.ff.set_alpha(a)

    def forward(self, coords: torch.Tensor) -> PINNBackboneOutput:
        """
        Forward pass.
        
        Args:
            coords: [N, 3] coordinates with requires_grad=True for autograd derivatives
            
        Returns:
            PINNBackboneOutput object with fields:
            - Vector mode: A_z, B, u, eta_raw, B_x, B_y, B_z, u_x, u_y
            - Scalar mode: A_z, B_z, u_x, u_y, eta_raw
        """
        h = self.ff(coords)
        out = self.net(h)
        
        if self.vector_B:
            if self.hard_div_free:
                # Hard div-free mode: unpack Az(1), Bz(1), u(2), [eta(1)]
                if self.learn_eta:
                    A_z, Bz, ux, uy, eta_raw = torch.split(out, 1, dim=-1)
                else:
                    A_z, Bz, ux, uy = torch.split(out, 1, dim=-1)
                    eta_raw = None
                
                # Derive Bx, By from Az to strictly enforce div B = 0
                Bx, By = B_perp_from_Az(A_z, coords)
            else:
                # Vector mode: unpack A_z(1), B(3), u(2), [eta(1)]
                if self.learn_eta:
                    A_z, Bx, By, Bz, ux, uy, eta_raw = torch.split(out, 1, dim=-1)
                else:
                    A_z, Bx, By, Bz, ux, uy = torch.split(out, 1, dim=-1)
                    eta_raw = None
            
            # Pack into vectors for new physics module
            B = torch.cat([Bx, By, Bz], dim=-1)  # [N, 3]
            u = torch.cat([ux, uy], dim=-1)      # [N, 2]
            
            return PINNBackboneOutput(
                A_z=A_z,
                B=B,
                u=u,
                eta_raw=eta_raw,
                B_x=Bx,
                B_y=By,
                B_z=Bz,
                u_x=ux,
                u_y=uy
            )
        else:
            # Scalar mode (legacy): unpack A_z(1), B_z(1), ux(1), uy(1), [eta(1)]
            if self.learn_eta:
                A_z, B_z, u_x, u_y, eta_raw = torch.split(out, 1, dim=-1)
            else:
                A_z, B_z, u_x, u_y = torch.split(out, 1, dim=-1)
                eta_raw = None
            
            return PINNBackboneOutput(
                A_z=A_z,
                B_z=B_z,
                u_x=u_x,
                u_y=u_y,
                eta_raw=eta_raw
            )

# ---------------- Helpers (hard in-plane solenoidality) ---------------- #

@torch.enable_grad()
def B_perp_from_Az(A_z: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Bx, By from A_z via B = curl(A_z zhat) => (Bx, By) = (-∂y Az, ∂x Az).
    A_z: [N,1], coords: [N,3] with requires_grad=True
    """
    assert A_z.requires_grad and coords.requires_grad
    ones = torch.ones_like(A_z)
    grads = torch.autograd.grad(A_z, coords, grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    dAz_dx = grads[..., 0:1]
    dAz_dy = grads[..., 1:2]
    Bx = -dAz_dy
    By = dAz_dx
    return Bx, By

class SpatialAttentionPool(nn.Module):
    """
    Attention-based spatial pooling that learns to focus on important regions.
    
    Key insight: Not all spatial locations are equally important for flare prediction.
    PIL regions and areas with high field gradients are more informative.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # Initialize with smaller weights for stable attention
        with torch.no_grad():
            nn.init.xavier_uniform_(self.attn[0].weight, gain=0.5)
            nn.init.zeros_(self.attn[0].bias)
            nn.init.xavier_uniform_(self.attn[2].weight, gain=0.1)
            nn.init.zeros_(self.attn[2].bias)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, P, C] features
            
        Returns:
            pooled: [B, T, C] attention-weighted features
            attn_weights: [B, T, P, 1] attention weights (for visualization)
        """
        # Compute attention scores and scale by sqrt(hidden) for numerical stability
        scores = self.attn(x) / (self.hidden ** 0.5)  # [B, T, P, 1]
        
        # Softer clamp for better gradient flow
        scores = scores.clamp(-20.0, 20.0)
        
        weights = torch.softmax(scores, dim=2)  # Softmax over spatial dim P
        
        # Handle potential NaN from softmax (extreme scores)
        weights = torch.nan_to_num(weights, nan=1.0/x.shape[2], posinf=1.0, neginf=0.0)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=2)  # [B, T, C]
        
        return pooled, weights


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling for observed frames.
    
    Learns which time steps are most predictive of future flares.
    Recent frames and frames with high activity should get more weight.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # Initialize with smaller weights for stable attention
        with torch.no_grad():
            nn.init.xavier_uniform_(self.attn[0].weight, gain=0.5)
            nn.init.zeros_(self.attn[0].bias)
            nn.init.xavier_uniform_(self.attn[2].weight, gain=0.1)
            nn.init.zeros_(self.attn[2].bias)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] features
            mask: [B, T] boolean mask for observed frames
            
        Returns:
            pooled: [B, C] attention-weighted features
            attn_weights: [B, T, 1] attention weights
        """
        # Compute attention scores with scaling for numerical stability
        scores = self.attn(x) / (self.hidden ** 0.5)  # [B, T, 1]
        
        # Mask unobserved frames with large negative value
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax over temporal dimension
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        
        # Handle all-masked case (when all frames are -inf, softmax produces NaN)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Fallback: if all weights are zero (all masked), use uniform weights on first valid frame
        weight_sum = weights.sum(dim=1, keepdim=True)  # [B, 1, 1]
        if (weight_sum == 0).any():
            # Find batches with zero weights and set uniform weights on valid frames
            zero_mask = (weight_sum.squeeze(-1) == 0)  # [B, 1]
            n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            uniform_weights = (mask.float().unsqueeze(-1) / n_valid.unsqueeze(-1))  # [B, T, 1]
            weights = torch.where(zero_mask.unsqueeze(-1), uniform_weights, weights)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [B, C]
        
        return pooled, weights


class PhysicsFeatureExtractor(nn.Module):
    """
    Extract physics-informed features from PINN outputs.
    
    Features include:
    - Field statistics (mean, std, max, min)
    - Gradient-based features (shear, PIL proximity)
    - Flow statistics (convergence, vorticity proxy)
    - Temporal derivatives (rate of change)
    
    These features capture physical indicators of flare potential.
    
    SOTA improvements:
    - Added robust statistics (median, IQR) for outlier resistance
    - Added higher-order temporal features (acceleration)
    - Added field-flow correlation features
    - Added current helicity proxy (critical for flare prediction)
    - Added magnetic complexity measure
    
    Output: 18 physics features (up from 13)
    """
    N_FEATURES = 18  # Total number of output features
    
    def __init__(self, use_extended_features: bool = True):
        super().__init__()
        self.use_extended = use_extended_features
        
    def forward(self, feats: dict[str, torch.Tensor], observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: dict with A_z, B_z, u_x, u_y each [B, T, P, C] where C can be 1 or 3
                   For 3-component mode, also expects B_x, B_y, B_z_scalar
            observed_mask: [B, T] boolean mask
            
        Returns:
            physics_features: [B, n_features] physics-derived features
        """
        # For physics features, prefer scalar Bz if available (more meaningful stats)
        # B_z_scalar is provided in 3-component mode with the actual Bz component
        if "B_z_scalar" in feats:
            B_z = feats["B_z_scalar"]  # [B, T, P, 1] - actual Bz component
        else:
            B_z = feats["B_z"]  # [B, T, P, C] - may be multi-component
            # If multi-component, take last component as Bz (convention: [Bx, By, Bz])
            if B_z.shape[-1] > 1:
                B_z = B_z[..., -1:]  # Extract Bz as [B, T, P, 1]
        
        u_x = feats["u_x"]
        u_y = feats["u_y"]
        A_z = feats["A_z"]
        # If A_z is multi-component, take last component (Az for 2.5D)
        if A_z.shape[-1] > 1:
            A_z = A_z[..., -1:]
        
        B, T, P, _ = B_z.shape
        mask_float = observed_mask[..., None, None].float()  # [B, T, 1, 1]
        mask_bool = mask_float.bool()
        
        # 1. Field magnitude statistics
        # FIXED: Use where() to avoid 0*Inf=NaN
        B_z_masked = torch.where(mask_bool, B_z, torch.zeros_like(B_z))
        # FIXED: valid_count = sum of mask values (each is 1 for valid, 0 for invalid)
        # mask_float is [B, T, 1, 1], sum over T gives count of valid timesteps
        # Multiply by P since each valid timestep contributes P spatial points
        n_valid_timesteps = mask_float.sum(dim=(1, 2, 3)).clamp(min=1.0)  # [B]
        valid_count = n_valid_timesteps * P  # [B] - total valid spatial-temporal points
        valid_count = valid_count.clamp(min=1.0)  # Safety
        
        # Mean
        bz_mean = B_z_masked.sum(dim=(1, 2, 3)) / valid_count
        
        # Stable variance: E[(X - mean)^2]
        bz_centered = torch.where(mask_bool, B_z - bz_mean.view(B, 1, 1, 1), torch.zeros_like(B_z))
        bz_var = (bz_centered ** 2).sum(dim=(1, 2, 3)) / valid_count
        bz_std = bz_var.clamp(min=1e-6).sqrt()  # Increased epsilon for stability
        
        # Max (absolute)
        # Use where to set masked values to -inf for max
        bz_abs_masked = torch.where(mask_bool, B_z.abs(), torch.tensor(-1e9, device=B_z.device))
        bz_max = bz_abs_masked.reshape(B, -1).max(dim=1)[0]
        # Clamp to 0 just in case all were masked
        bz_max = bz_max.clamp(min=0.0)
        
        # 2. Polarity mixture (indicator of PIL presence)
        # IMPROVED: Use soft thresholding for smoother gradients
        sig_pos = torch.sigmoid(10 * B_z)
        sig_neg = torch.sigmoid(-10 * B_z)
        
        bz_pos_soft = torch.where(mask_bool, sig_pos, torch.zeros_like(sig_pos))
        bz_neg_soft = torch.where(mask_bool, sig_neg, torch.zeros_like(sig_neg))
        
        bz_pos = bz_pos_soft.sum(dim=(1, 2, 3)) / valid_count
        bz_neg = bz_neg_soft.sum(dim=(1, 2, 3)) / valid_count
        polarity_balance = (bz_pos * bz_neg).sqrt()  # High when balanced = PIL present
        
        # 3. Flow statistics
        # FIXED: Use clamp before sqrt to ensure non-negative input
        u_sq = (u_x ** 2 + u_y ** 2).clamp(min=1e-12)
        u_mag = u_sq.sqrt()
        u_masked = torch.where(mask_bool, u_mag, torch.zeros_like(u_mag))
        
        u_mean = u_masked.sum(dim=(1, 2, 3)) / valid_count
        
        u_for_max = torch.where(mask_bool, u_mag, torch.full_like(u_mag, -1e9))
        u_max = u_for_max.reshape(B, -1).max(dim=1)[0].clamp(min=0.0)
        
        # 4. Flux transport: correlation of flow with field gradient
        # This is a key indicator of magnetic energy buildup
        ft_val = u_mag * B_z.abs()
        flux_transport = torch.where(mask_bool, ft_val, torch.zeros_like(ft_val)).sum(dim=(1, 2, 3)) / valid_count
        
        # 5. Temporal variability (rate of change indicator)
        if T > 1:
            # First-order difference (velocity of change)
            dBz_dt = (B_z[:, 1:] - B_z[:, :-1]).abs()
            mask_dt = mask_float[:, 1:] * mask_float[:, :-1]
            mask_dt_bool = mask_dt.bool()
            
            dBz_dt_masked = torch.where(mask_dt_bool, dBz_dt, torch.zeros_like(dBz_dt))
            temporal_var = dBz_dt_masked.sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) + 1e-8)
            
            # SOTA: Second-order difference (acceleration) - critical for flare precursors
            # Rapid acceleration often precedes flares
            if T > 2:
                d2Bz_dt2 = (dBz_dt[:, 1:] - dBz_dt[:, :-1]).abs()
                mask_dt2 = mask_dt[:, 1:] * mask_dt[:, :-1]
                mask_dt2_bool = mask_dt2.bool()
                d2Bz_dt2_masked = torch.where(mask_dt2_bool, d2Bz_dt2, torch.zeros_like(d2Bz_dt2))
                temporal_accel = d2Bz_dt2_masked.sum(dim=(1, 2, 3)) / (mask_dt2.sum(dim=(1, 2, 3)) + 1e-8)
            else:
                temporal_accel = torch.zeros(B, device=B_z.device)
            
            # SOTA: Evolution rate variance - high variance indicates unstable evolution
            dBz_dt_sq = torch.where(mask_dt_bool, dBz_dt**2, torch.zeros_like(dBz_dt))
            dBz_dt_mean_sq = temporal_var ** 2
            dBz_dt_var = (dBz_dt_sq.sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) + 1e-8)) - dBz_dt_mean_sq
            evolution_rate_var = dBz_dt_var.clamp(min=0).sqrt()
        else:
            temporal_var = torch.zeros(B, device=B_z.device)
            temporal_accel = torch.zeros(B, device=B_z.device)
            evolution_rate_var = torch.zeros(B, device=B_z.device)
        
        # 6. Vector potential complexity (proxy for current density complexity)
        az_val = A_z ** 2
        az_var = torch.where(mask_bool, az_val, torch.zeros_like(az_val)).sum(dim=(1, 2, 3)) / valid_count
        az_std = az_var.clamp(min=1e-8).sqrt()
        
        # 7. Horizontal field features (if available)
        if "B_x" in feats and "B_y" in feats:
            B_x = feats["B_x"]
            B_y = feats["B_y"]
            
            # Horizontal field magnitude - FIXED: clamp before sqrt
            bh_sq = (B_x**2 + B_y**2).clamp(min=1e-12)
            bh_mag = bh_sq.sqrt()
            bh_masked = torch.where(mask_bool, bh_mag, torch.zeros_like(bh_mag))
            bh_mean = bh_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # Total field magnitude - FIXED: clamp before sqrt  
            btot_sq = (B_x**2 + B_y**2 + B_z**2).clamp(min=1e-12)
            btot_mag = btot_sq.sqrt()
            btot_masked = torch.where(mask_bool, btot_mag, torch.zeros_like(btot_mag))
            btot_mean = btot_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # 8. Shear angle proxy: angle between horizontal field and potential field
            # For a potential field, B would be nearly vertical at PIL
            # Shear = Bh / Btot (high shear = more non-potential = flare prone)
            # FIXED: Increase epsilon for division stability
            shear_proxy = bh_mag / btot_mag.clamp(min=1e-6)
            shear_masked = torch.where(mask_bool, shear_proxy, torch.zeros_like(shear_proxy))
            shear_mean = shear_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # 9. Free energy proxy: E_free ∝ Bh² (excess over potential field)
            # This is a key predictor of flare energy
            free_energy_proxy = bh_sq  # Already computed, avoid recomputing bh_mag ** 2
            fe_masked = torch.where(mask_bool, free_energy_proxy, torch.zeros_like(free_energy_proxy))
            free_energy_mean = fe_masked.sum(dim=(1, 2, 3)) / valid_count
        else:
            bh_mean = torch.zeros_like(bz_mean)
            btot_mean = bz_mean.abs()  # Fallback to |Bz|
            shear_mean = torch.zeros_like(bz_mean)
            free_energy_mean = torch.zeros_like(bz_mean)

        # Magnetic complexity metric: kurtosis proxy
        # High kurtosis indicates non-Gaussian, complex field configurations
        bz_4th = torch.where(mask_bool, B_z ** 4, torch.zeros_like(B_z))
        bz_4th_mean = bz_4th.sum(dim=(1, 2, 3)) / valid_count
        # FIXED: Increase epsilon and clamp bz_var to avoid division by tiny values
        bz_var_safe = bz_var.clamp(min=1e-6)
        kurtosis_proxy = bz_4th_mean / (bz_var_safe ** 2 + 1e-4)
        
        # SOTA: Current helicity proxy (if we have horizontal field)
        # Hc = Jz * Bz where Jz ≈ curl(B)_z = ∂By/∂x - ∂Bx/∂y
        # We can't compute gradients here directly, so use variance ratio as proxy
        if "B_x" in feats and "B_y" in feats:
            B_x = feats["B_x"]
            B_y = feats["B_y"]
            # Twist proxy: |Bh| * |Bz| correlation (high for helical fields)
            # FIXED: Reuse bh_mag from above if available, else compute safely
            bh_sq_helicity = (B_x**2 + B_y**2).clamp(min=1e-12)
            bh_mag_helicity = bh_sq_helicity.sqrt()
            twist_proxy = torch.where(mask_bool, bh_mag_helicity * B_z.abs(), torch.zeros_like(bh_mag_helicity))
            current_helicity_proxy = twist_proxy.sum(dim=(1, 2, 3)) / valid_count
        else:
            current_helicity_proxy = torch.zeros_like(bz_mean)
        
        # SOTA: Recent evolution rate (weighted towards latest frames)
        # More weight on recent changes as they're most predictive
        if T > 1:
            # Exponentially weighted mean of |dBz/dt|
            time_weights = torch.exp(torch.linspace(-2, 0, T-1, device=B_z.device))
            time_weights = time_weights / time_weights.sum()
            dBz_dt = (B_z[:, 1:] - B_z[:, :-1]).abs()
            mask_dt = mask_float[:, 1:] * mask_float[:, :-1]
            
            # Apply time weighting
            weighted_dBz = dBz_dt * time_weights[None, :, None, None] * mask_dt
            recent_evolution = weighted_dBz.sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) * time_weights.sum() + 1e-8)
        else:
            recent_evolution = torch.zeros(B, device=B_z.device)
        
        # Concatenate ALL physics features (now 18 features for comprehensive physics)
        physics_feats = torch.stack([
            # Base field statistics (4)
            bz_mean, bz_std, bz_max, polarity_balance,
            # Flow statistics (3)
            u_mean, u_max, flux_transport,
            # Temporal features (4) - ENHANCED
            temporal_var, temporal_accel, evolution_rate_var, recent_evolution,
            # Field complexity (2)
            az_std, kurtosis_proxy,
            # Vector field features (5) - if available
            bh_mean, btot_mean, shear_mean, free_energy_mean, current_helicity_proxy
        ], dim=-1)  # [B, 18]
        
        # Handle any NaN that might occur - use smaller bounds
        physics_feats = torch.nan_to_num(physics_feats, nan=0.0, posinf=100.0, neginf=-100.0)
        physics_feats = physics_feats.clamp(-100.0, 100.0)
        
        return physics_feats


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that stays active during inference.
    
    Used for uncertainty quantification following Gal & Ghahramani (2016).
    """
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        self._mc_mode = False
    
    def set_mc_mode(self, enabled: bool):
        """Enable/disable MC dropout mode for inference."""
        self._mc_mode = enabled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Active during training OR when mc_mode is enabled
        if self.training or self._mc_mode:
            return F.dropout(x, self.p, training=True)
        return x


class ClassifierHead(nn.Module):
    """
    Advanced classifier head with attention and physics features.
    
    SOTA improvements:
    - MC Dropout for uncertainty quantification
    - Enhanced feature aggregation
    - RF-guided physics feature weighting
    - Dynamic feature dimension handling for multi-component fields
    """
    def __init__(
        self, 
        hidden: int = 256, 
        dropout: float = 0.1, 
        horizons: tuple[int, ...] = (6, 12, 24),
        use_attention: bool = True,
        use_physics_features: bool = True,
        rf_importance_weights: Optional[torch.Tensor] = None,
        n_scalar_features: int = 4,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.2,
        n_field_components: int = 1,  # NEW: Support multi-component fields
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.use_attention = use_attention
        self.use_physics_features = use_physics_features
        self.n_scalar_features = n_scalar_features
        self.mc_dropout_enabled = mc_dropout
        self.n_field_components = n_field_components
        
        # Base features: A (n_comp) + B (n_comp) + u_x (1) + u_y (1)
        # For 1-component: A_z(1) + B_z(1) + u_x(1) + u_y(1) = 4
        # For 3-component: A(3) + B(3) + u_x(1) + u_y(1) = 8
        base_features = n_field_components * 2 + 2
        
        # Spatial attention pooling
        if use_attention:
            self.spatial_attn = SpatialAttentionPool(base_features, hidden=64)
            self.temporal_attn = TemporalAttentionPool(base_features, hidden=64)
        
        # Physics feature extractor
        if use_physics_features:
            self.physics_extractor = PhysicsFeatureExtractor()
            physics_features = PhysicsFeatureExtractor.N_FEATURES  # 18 features
        else:
            physics_features = 0
        
        # RF importance weights for physics features
        # If provided, these weight the physics features by domain knowledge
        if rf_importance_weights is not None:
            self.register_buffer('rf_weights', rf_importance_weights)
        else:
            # Default: uniform weights
            self.register_buffer('rf_weights', torch.ones(physics_features) if physics_features > 0 else None)
        
        # Scalar feature output dimension (computed before dropout type selection)
        scalar_out_dim = hidden // 4 if n_scalar_features > 0 else 0
        
        # Compute total input features for final MLP
        # With attention: pooled(4) + mean(4) + std(4) + max(4) = 16
        # Plus physics features (18) + scalar features (hidden//4) = 34 + scalar_out
        # Without attention: mean(4) + std(4) + physics_features(18) + scalar = 26 + scalar_out
        if use_attention:
            mlp_input = base_features * 4 + physics_features + scalar_out_dim
        else:
            mlp_input = base_features * 2 + physics_features + scalar_out_dim
        
        # Choose dropout type based on MC dropout setting
        # FIXED: Use MCDropout consistently when mc_dropout=True
        if mc_dropout:
            self._dropout_rate = mc_dropout_rate
            drop_cls = lambda p: MCDropout(mc_dropout_rate)  # Use mc_dropout_rate, not p
        else:
            self._dropout_rate = dropout
            drop_cls = nn.Dropout
        
        # Scalar feature processing (R-value, GWPIL, etc.) with proper dropout type
        if n_scalar_features > 0:
            self.scalar_proj = nn.Sequential(
                nn.Linear(n_scalar_features, hidden // 4),
                nn.SiLU(),
                MCDropout(mc_dropout_rate * 0.5) if mc_dropout else nn.Dropout(dropout * 0.5),
            )
        else:
            self.scalar_proj = None
        
        # Final classifier MLP with residual
        pre_mlp_linear = nn.Linear(mlp_input, hidden)
        self.pre_mlp = nn.Sequential(
            pre_mlp_linear,
            nn.LayerNorm(hidden),
            nn.SiLU(),
            drop_cls(dropout),
        )
        
        hidden1 = nn.Linear(hidden, hidden // 2)
        output_linear = nn.Linear(hidden // 2, len(horizons))
        self.classifier = nn.Sequential(
            hidden1,
            nn.SiLU(),
            drop_cls(dropout),
            output_linear
        )
        
        # Initialize classifier output layer with small weights for balanced predictions
        with torch.no_grad():
            nn.init.xavier_uniform_(output_linear.weight, gain=0.1)
            # Initialize bias slightly negative for rare event prediction
            # This helps with class imbalance by starting with low positive predictions
            output_linear.bias.fill_(-1.0)
        
        # Store dropout layers for MC mode toggling
        # FIXED: Collect from ALL submodules including scalar_proj
        self._mc_dropout_layers: list[MCDropout] = []
        for m in self.modules():
            if isinstance(m, MCDropout):
                self._mc_dropout_layers.append(m)
    
    def set_mc_mode(self, enabled: bool):
        """Enable/disable Monte Carlo dropout mode for uncertainty estimation."""
        for layer in self._mc_dropout_layers:
            layer.set_mc_mode(enabled)
    
    def forward_with_uncertainty(
        self,
        feats: dict[str, torch.Tensor],
        observed_mask: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with MC Dropout uncertainty estimation.
        
        Args:
            feats: Feature dictionary
            observed_mask: Temporal mask
            scalars: Scalar features
            n_samples: Number of MC samples
        
        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Uncertainty (std of predictions)
            logits: All sampled logits [n_samples, B, horizons]
        """
        self.set_mc_mode(True)
        
        all_logits = []
        for _ in range(n_samples):
            logits = self.forward(feats, observed_mask, scalars)
            all_logits.append(logits)
        
        self.set_mc_mode(False)
        
        all_logits = torch.stack(all_logits)  # [n_samples, B, horizons]
        all_probs = torch.sigmoid(all_logits)
        
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        return mean_probs, std_probs, all_logits

    def forward(
        self, 
        feats: dict[str, torch.Tensor], 
        observed_mask: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feats: dict of tensors shaped [B, T, P, 1] for keys A_z, B_z, u_x, u_y
            observed_mask: [B, T] booleans (True for frames t <= t0)
            scalars: [B, n_scalar_features] optional scalar features (R-value, GWPIL, etc.)
            
        Returns:
            logits: [B, n_horizons] classification logits
        """
        # Concatenate base features
        # Shape depends on n_field_components: [B,T,P,4] for 1-comp, [B,T,P,8] for 3-comp
        X = torch.cat([feats["A_z"], feats["B_z"], feats["u_x"], feats["u_y"]], dim=-1)
        B, T, P, C = X.shape
        
        # Ensure mask has batch dimension
        if observed_mask.dim() == 1:
            observed_mask = observed_mask.unsqueeze(0).expand(B, -1)
        
        if self.use_attention:
            # Spatial attention pooling
            X_spatial, spatial_weights = self.spatial_attn(X)  # [B, T, 4]
            
            # Temporal attention pooling
            X_pooled, temporal_weights = self.temporal_attn(X_spatial, observed_mask)  # [B, C]
            
            # IMPROVED: Compute comprehensive statistics for better feature representation
            mask_expanded = observed_mask[..., None, None].float()  # [B, T, 1, 1]
            X_masked = X * mask_expanded
            # FIXED: valid_count = (# valid timesteps) * P spatial points
            n_valid_t = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)  # [B, 1]
            valid_count = n_valid_t * P  # [B, 1]
            valid_count = valid_count.clamp(min=1.0)  # Safety clamp
            
            # Mean
            X_mean = X_masked.sum(dim=(1, 2)) / valid_count  # [B, C]
            
            # Standard deviation (more informative than just mean)
            X_sq = (X_masked ** 2).sum(dim=(1, 2)) / valid_count
            X_var = (X_sq - X_mean ** 2).clamp(min=1e-8)
            X_std = X_var.sqrt()  # [B, C]
            
            # Max absolute value (captures peak activity)
            # Set masked values to large negative so they don't affect max
            X_for_max = X.abs() * mask_expanded + (1 - mask_expanded) * (-1e9)
            X_max = X_for_max.max(dim=2)[0].max(dim=1)[0]  # [B, C]
            
            # Combine: attention pooled + mean + std + max = 4*C features
            combined = torch.cat([X_pooled, X_mean, X_std, X_max], dim=-1)  # [B, 4*base_features]
        else:
            # Improved fallback with mean + std statistics
            mask_expanded = observed_mask[..., None, None].float()  # [B, T, 1, 1]
            X_masked = X * mask_expanded
            # FIXED: valid_count = (# valid timesteps) * P spatial points
            n_valid_t = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)  # [B, 1]
            valid_count = n_valid_t * P  # [B, 1]
            valid_count = valid_count.clamp(min=1.0)
            
            X_mean = X_masked.sum(dim=(1, 2)) / valid_count  # [B, C]
            X_sq = (X_masked ** 2).sum(dim=(1, 2)) / valid_count
            X_var = (X_sq - X_mean ** 2).clamp(min=1e-8)
            X_std = X_var.sqrt()  # [B, C]
            
            combined = torch.cat([X_mean, X_std], dim=-1)  # [B, 2*C]
        
        # Physics features (18 features from PhysicsFeatureExtractor)
        if self.use_physics_features:
            physics_feats = self.physics_extractor(feats, observed_mask)  # [B, 18]
            
            # Apply RF importance weights if available
            if self.rf_weights is not None:
                physics_feats = physics_feats * self.rf_weights
            
            combined = torch.cat([combined, physics_feats], dim=-1)
        
        # Scalar features (R-value, GWPIL, observation coverage, etc.)
        if self.scalar_proj is not None and scalars is not None:
            #  FIX #18: Normalize ALL scalar features for stability
            # Feature layout (16 features):
            #   0: r_value (log-scale, ~0-10) - already normalized
            #   1: gwpil (can be large) - needs log1p
            #   2: obs_coverage (0-1) - already normalized
            #   3: frame_count (0-48+) - needs normalization
            #   4-11: PIL evolution (various scales) - needs log1p for large values
            #   12-15: temporal stats (various scales) - needs log1p for large values
            scalars_safe = scalars.clone()
            n_feats = scalars_safe.shape[-1]
            
            # Log-transform features that can be large (indices 1, 3, and 4+)
            # Apply log1p with sign preservation for all features except r_value (0) and obs_coverage (2)
            for idx in [1, 3] + list(range(4, min(n_feats, 16))):
                if idx < n_feats:
                    scalars_safe[:, idx] = torch.log1p(scalars_safe[:, idx].abs()) * torch.sign(scalars_safe[:, idx])
            
            # Clamp to prevent extreme values
            scalars_safe = scalars_safe.clamp(-20.0, 20.0)
            
            scalar_feats = self.scalar_proj(scalars_safe)  # [B, hidden//4]
            combined = torch.cat([combined, scalar_feats], dim=-1)
        
        # NaN/Inf safety: clamp combined features before MLP
        # Use any() to avoid MPS synchronization issues
        with torch.no_grad():
            if torch.isnan(combined).any() or torch.isinf(combined).any():
                combined = torch.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)
        combined = combined.clamp(-100.0, 100.0)
        
        # MLP classifier
        h = self.pre_mlp(combined)
        
        logits = self.classifier(h)
        
        # Softer clamp on logits to allow confident predictions
        # sigmoid(-15) ≈ 3e-7, sigmoid(15) ≈ 0.9999997 - enough range for confident predictions
        logits = logits.clamp(-15.0, 15.0)
        
        return logits  # [B, n_horizons]

