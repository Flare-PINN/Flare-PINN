# src/models/pinn/hybrid_core.py
"""
Hybrid CNN-conditioned PINN backbone.
Combines spatial CNN features with coordinate-based neural fields.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from .config import EncoderConfig
from .core import FourierFeatures, fourier_out_dim
from .encoder import (
    TinyEncoder,
    TemporalEncoder,
    TransformerTemporalEncoder,
    MultiScaleEncoder,
    LightweightTemporalEncoder,
    EfficientGRUEncoder,
)
from .latent_sampling import sample_latent_soft_bilinear, sample_latent_nearest
from .film import FiLM


class HybridPINNBackbone(nn.Module):
    """
    Coordinate MLP conditioned by CNN features.
    
    Architecture:
        1. Encode observed frames → (L, g) via TemporalEncoder (or TinyEncoder)
        2. For each collocation point (x,y,t):
           - Compute Fourier features
           - Sample L at (x,y)
           - Concatenate [x, y, t, FF, L_sampled, g]
           - Pass through MLP with FiLM conditioning
        3. Output: A_z, B_z, u_x, u_y, [eta_raw]
    
    This combines:
        - CNN: Spatial inductive bias (locality, translation equivariance)
        - PINN: Physics constraints via weak-form residuals
    """
    def __init__(
        self,
        encoder_in_channels: int,
        encoder_cfg: Optional[EncoderConfig] = None,
        hidden: int = 384,
        layers: int = 10,
        max_log2_freq: int = 5,
        film_layers: tuple[int, ...] = (3, 6, 9),
        learn_eta: bool = False,
        n_field_components: int = 1,
    ):
        super().__init__()
        
        encoder_cfg = encoder_cfg or EncoderConfig()
        self.encoder_type = encoder_cfg.type
        latent_channels = encoder_cfg.latent_channels
        global_dim = encoder_cfg.global_dim
        encoder_dropout = encoder_cfg.dropout
        use_checkpoint = encoder_cfg.use_checkpoint
        self.n_field_components = n_field_components
        
        # ------------------------------------------------------------------ #
        # Encoder selection (config-driven)
        # ------------------------------------------------------------------ #
        if self.encoder_type == "transformer":
            self.encoder = TransformerTemporalEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                d_model=max(encoder_cfg.latent_channels * 4, global_dim),
                n_heads=encoder_cfg.n_heads,
                n_layers=encoder_cfg.n_transformer_layers,
                dropout=encoder_dropout,
            )
            self.effective_latent_channels = latent_channels * 2
            self.use_temporal_encoder = True
        elif self.encoder_type == "lightweight":
            self.encoder = LightweightTemporalEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                temporal_dim=latent_channels,
                dropout=encoder_dropout,
                max_frames=64,
                use_checkpoint=use_checkpoint,
            )
            self.effective_latent_channels = latent_channels * 2
            self.use_temporal_encoder = True
        elif self.encoder_type == "gru":
            self.encoder = EfficientGRUEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                dropout=encoder_dropout,
                use_checkpoint=use_checkpoint,
            )
            self.effective_latent_channels = latent_channels * 2
            self.use_temporal_encoder = True
        elif self.encoder_type == "multiscale":
            self.encoder = MultiScaleEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                dropout=encoder_dropout,
            )
            self.effective_latent_channels = latent_channels
            self.use_temporal_encoder = False
        elif self.encoder_type == "tiny":
            self.encoder = TinyEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                dropout=encoder_dropout,
                use_checkpoint=use_checkpoint,
            )
            self.effective_latent_channels = latent_channels
            self.use_temporal_encoder = False
        else:
            # Default: temporal encoder (same as previous behaviour)
            self.encoder = TemporalEncoder(
                in_channels=encoder_in_channels,
                latent_channels=latent_channels,
                global_dim=global_dim,
                dropout=encoder_dropout
            )
            self.effective_latent_channels = latent_channels * 2
            self.use_temporal_encoder = True
        
        # Fourier features
        self.ff = FourierFeatures(3, max_log2_freq)
        ff_dim = fourier_out_dim(3, max_log2_freq)
        
        # Input: [x, y, t, FF, L_sampled, g]
        in_dim = 3 + ff_dim + self.effective_latent_channels + global_dim
        
        # Output dimension:
        # A (potential): n_field_components (scalar potential for 2D B, vector for 3D B?)
        # Actually for 2.5D:
        # If 1 comp (Bz) -> Az (1) + Bz (1) + ux, uy (2) = 4
        # If 3 comp (Bx,By,Bz) -> B (3) + u (3?) + A (3?)
        # For minimal refactor:
        # Always output:
        # - A (potential): n_field_components
        # - B (field): n_field_components
        # - u (velocity): 2 (assuming 2.5D flow) or 3? Sticking to 2 for now as u_x, u_y are hardcoded elsewhere.
        # - eta: 1 if learned
        
        self.n_u = 2 # ux, uy
        out_dim = n_field_components * 2 + self.n_u
        if learn_eta:
            out_dim += 1
        
        # Coordinate MLP with FiLM conditioning
        self.film_layers = set(film_layers)
        self.layers_list = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.film_modules = nn.ModuleDict()
        
        # First layer
        self.layers_list.append(nn.Linear(in_dim, hidden))
        self.activations.append(nn.SiLU())
        
        # Hidden layers with FiLM
        for i in range(1, layers):
            self.layers_list.append(nn.Linear(hidden, hidden))
            self.activations.append(nn.SiLU())
            if (i + 1) in self.film_layers:  # Layer indexing: 1-based
                self.film_modules[str(i + 1)] = FiLM(hidden, global_dim)
        
        # Output head with small initialization for stable starting point
        self.head = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.head.weight, gain=0.1)
            nn.init.zeros_(self.head.bias)
            
        self.learn_eta = learn_eta
        self.latent_channels = latent_channels
        self.global_dim = global_dim
    
    def set_fourier_alpha(self, a: float) -> None:
        """Annealing for Fourier features."""
        self.ff.set_alpha(a)
    
    def encode(
        self, 
        frames: torch.Tensor, 
        observed_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observed frames into latent features.
        
        Args:
            frames: [T, C, H, W] or [1, C, H, W]
            observed_mask: [T] boolean mask (required for TemporalEncoder)
        
        Returns:
            L: [N, C_latent, H, W]
            g: [N, D_global]
        """
        if self.use_temporal_encoder:
            if observed_mask is None:
                raise ValueError(f"observed_mask required for {self.encoder_type} encoder")
            L, g, *_ = self.encoder(frames, observed_mask)
        else:
            # Aggregate observed frames (or all frames if mask not provided)
            obs_frames = frames if observed_mask is None else frames[observed_mask]
            if obs_frames.shape[0] == 0:
                H, W = frames.shape[-2:]
                device = frames.device
                latent_ch = self.effective_latent_channels
                L = torch.zeros(1, latent_ch, H, W, device=device)
                g = torch.zeros(1, self.global_dim, device=device)
                return L, g
            
            if obs_frames.dim() == 3:
                frames_input = obs_frames.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                frames_input = obs_frames.mean(dim=0, keepdim=True)
            
            L, g = self.encoder(frames_input)
        
        # NaN/Inf safety and clamping for stability
        # FIXED: Use any() checks to avoid MPS synchronization issues
        with torch.no_grad():
            if torch.isnan(L).any() or torch.isinf(L).any():
                L = torch.nan_to_num(L, nan=0.0, posinf=1.0, neginf=-1.0)
        L = L.clamp(-10.0, 10.0)
        
        with torch.no_grad():
            if torch.isnan(g).any() or torch.isinf(g).any():
                g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)
        g = g.clamp(-10.0, 10.0)
        
        return L, g
    
    def forward(
        self,
        coords: torch.Tensor,
        L: torch.Tensor,
        g: torch.Tensor,
        use_nearest: bool = False
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Query neural field at collocation points with CNN conditioning.
        
        Args:
            coords: [N, 3] - Normalized coordinates (x, y, t) in [-1, 1]^3
            L: [1, C_latent, H, W] - Cached latent map (broadcast to N)
            g: [1, D_global] - Cached global code (broadcast to N)
            use_nearest: If True, use nearest-neighbor sampling (faster, no coord gradients)
        
        Returns:
            dict with A, B, u, eta_raw (tensors)
        """
        N = coords.shape[0]
        
        # Extract spatial coordinates for L sampling
        xy = coords[:, :2]  # [N, 2]
        
        # Reshape for batch processing: [N, 2] -> [1, N, 2] for sampler
        # The samplers expect [batch, points, 2]
        xy_batched = xy.unsqueeze(0)  # [1, N, 2]
        
        # Sample latent at collocation points with 2nd-order differentiable sampler
        if use_nearest:
            L_sampled = sample_latent_nearest(L, xy_batched)  # [1, N, C]
        else:
            L_sampled = sample_latent_soft_bilinear(L, xy_batched)  # [1, N, C]
        
        L_sampled = L_sampled.squeeze(0)  # [N, C]
        
        # Broadcast global code
        if g.shape[0] == 1:
            g_expanded = g.expand(N, -1)  # [N, D]
        else:
            g_expanded = g
        
        # Fourier features
        ff = self.ff(coords)  # [N, FF_dim]
        
        # FIXED: NaN/Inf safety checks before concatenation
        # Use any() checks to avoid MPS synchronization issues
        with torch.no_grad():
            if torch.isnan(L_sampled).any() or torch.isinf(L_sampled).any():
                L_sampled = torch.nan_to_num(L_sampled, nan=0.0, posinf=1.0, neginf=-1.0)
        L_sampled = L_sampled.clamp(-10.0, 10.0)
        
        with torch.no_grad():
            if torch.isnan(g_expanded).any() or torch.isinf(g_expanded).any():
                g_expanded = torch.nan_to_num(g_expanded, nan=0.0, posinf=1.0, neginf=-1.0)
        g_expanded = g_expanded.clamp(-10.0, 10.0)
        
        # Concatenate all inputs
        z = torch.cat([coords, ff, L_sampled, g_expanded], dim=-1)  # [N, in_dim]
        
        # Forward through MLP with FiLM
        h = z
        for i, (layer, act) in enumerate(zip(self.layers_list, self.activations), start=1):
            h = act(layer(h))
            # Apply FiLM conditioning (if applicable for this layer)
            if i in self.film_layers:
                h = self.film_modules[str(i)](h, g_expanded)
        
        # Output head
        out = self.head(h)
        
        #  SAFETY: Tight clamp on outputs to prevent NaN in physics gradients
        # ±10 chosen because: typical field values are O(1), sigmoid(10)≈1, tanh(10)≈1
        # Beyond ±10: gradients vanish, physics residuals explode, autograd.grad fails
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = out.clamp(-10.0, 10.0)
        
        # Split outputs
        # Layout: [A (C), B (C), u (2), eta (1?)]
        C = self.n_field_components
        
        start = 0
        A = out[..., start:start+C]
        start += C
        B_field = out[..., start:start+C]
        start += C
        u = out[..., start:start+self.n_u]
        start += self.n_u
        
        if self.learn_eta:
            eta_raw = out[..., start:start+1]
        else:
            eta_raw = None
        
        # Back-compat: map 1-component result to explicit Bz/Az keys if N=1
        # But new API prefers "B", "A", "u"
        return {
            "A": A,
            "B": B_field,
            "u": u,
            "eta_raw": eta_raw,
            # Keep old keys for safety if needed (only valid for C=1)
            "A_z": A if C == 1 else None,
            "B_z": B_field if C == 1 else None,
            "u_x": u[..., 0:1],
            "u_y": u[..., 1:2]
        }
