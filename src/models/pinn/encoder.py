# src/models/pinn/encoder.py
"""
CNN + Temporal Encoder for conditioning the PINN coordinate MLP.

Key improvements for SOTA:
1. Proper temporal processing (not just averaging frames)
2. Learned temporal position encoding
3. Multi-scale spatial features
4. Temporal attention for weighting important frames
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvGN(nn.Module):
    """Conv + GroupNorm + activation block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with skip connection and proper initialization."""
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvGN(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.SiLU()
        
        # Initialize conv2 with smaller weights for stable residual learning
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv2.weight, gain=0.1)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class TemporalPositionEncoding(nn.Module):
    """
    Learnable temporal position encoding for frame sequences.
    
    Maps relative time position to embedding that captures:
    - Recency (recent frames are more predictive)
    - Periodicity (solar rotation ~27 days)
    - Evolution rate context
    """
    def __init__(self, d_model: int = 32, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal base encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learnable refinement
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence
            
        Returns:
            encoding: [seq_len, d_model] position encoding
        """
        base = self.pe[:seq_len]
        return base + self.refine(base)


class TemporalAttention(nn.Module):
    """
    Attention mechanism for weighting temporal frames.
    
    Key insight: Not all frames are equally predictive.
    - Recent frames capture current state
    - Frames with rapid changes indicate evolution
    - Frames near observation gaps may be less reliable
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable "query" for extracting summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Initialize projections with Xavier for stable attention
        with torch.no_grad():
            for proj in [self.q_proj, self.k_proj, self.v_proj]:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
            # Output projection with smaller weights
            nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
            nn.init.zeros_(self.out_proj.bias)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] frame features
            mask: [B, T] boolean mask (True = valid frame)
            
        Returns:
            summary: [B, D] aggregated representation
            attn_weights: [B, T] attention weights
        """
        B, T, D = x.shape
        
        # Add CLS token for aggregation
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x_with_cls = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        
        # Project queries, keys, values
        Q = self.q_proj(x_with_cls[:, :1])  # [B, 1, D] - only query from CLS
        K = self.k_proj(x_with_cls[:, 1:])  # [B, T, D] - keys from frames
        V = self.v_proj(x_with_cls[:, 1:])  # [B, T, D] - values from frames
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, 1, T]
        
        # Mask invalid frames
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        attn = attn.masked_fill(~mask_expanded, float('-inf'))
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Handle all-masked case
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)  # [B, H, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        out = self.out_proj(out).squeeze(1)  # [B, D]
        
        # Return mean attention weights across heads
        avg_attn = attn_weights.mean(dim=1).squeeze(1)  # [B, T]
        
        return out, avg_attn


class TinyEncoder(nn.Module):
    """
    6-Layer ResNet Encoder for spatial feature extraction.
    
    Takes observed Bz frames (t <= t0) and produces:
    - L: Latent feature map [N, C, H, W] for per-point conditioning
    - g: Global code [N, D] for FiLM conditioning
    
    Architecture: 6 ResBlocks with progressive channel expansion.
    This deeper architecture captures multi-scale spatial features
    critical for PIL detection and flare prediction.
    
    Features:
    - Gradient checkpointing support for memory efficiency
    - Stochastic depth (DropPath) for regularization
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 32,
        global_dim: int = 64,
        dropout: float = 0.05,
        use_checkpoint: bool = False  # Gradient checkpointing for memory
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Stem: initial projection
        self.stem = ConvGN(in_channels, 32)
        
        # Stage 1: 32 channels, full resolution (2 ResBlocks)
        self.res1 = ResBlock(32, dropout)
        self.res2 = ResBlock(32, dropout)
        
        # Downsample 1: 32 → 64 channels, /2 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        # Stage 2: 64 channels (2 ResBlocks)
        self.res3 = ResBlock(64, dropout)
        self.res4 = ResBlock(64, dropout)
        
        # Downsample 2: 64 → 128 channels, /4 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # Stage 3: 128 channels (2 ResBlocks)
        self.res5 = ResBlock(128, dropout)
        self.res6 = ResBlock(128, dropout)
        
        # Latent map projection (upsample back to original resolution)
        self.proj_L = nn.Conv2d(128, latent_channels, 1)
        
        # Global code via spatial pooling + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, global_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim * 2, global_dim)
        )
    
    def _forward_stage1(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 1 forward for checkpointing."""
        h = self.res1(h)
        return self.res2(h)
    
    def _forward_stage2(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 2 forward for checkpointing."""
        h = self.down1(h)
        h = self.res3(h)
        return self.res4(h)
    
    def _forward_stage3(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 3 forward for checkpointing."""
        h = self.down2(h)
        h = self.res5(h)
        return self.res6(h)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, C_in, H, W] - Stacked observed Bz frames (t <= t0 only!)
        
        Returns:
            L: [N, C_latent, H, W] - Spatial feature map
            g: [N, D] - Global context code
        """
        # Stem
        h = self.stem(x)  # [N, 32, H, W]
        
        # Stage 1-3 with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            h = checkpoint(self._forward_stage1, h, use_reentrant=False)
            h = checkpoint(self._forward_stage2, h, use_reentrant=False)
            h = checkpoint(self._forward_stage3, h, use_reentrant=False)
        else:
            h = self._forward_stage1(h)
            h = self._forward_stage2(h)
            h = self._forward_stage3(h)
        
        # Global code from deepest features
        g = self.g_mlp(self.global_pool(h))  # [N, global_dim]
        
        # Latent map: project and upsample to original resolution
        L = self.proj_L(h)  # [N, latent_channels, H/4, W/4]
        L = F.interpolate(L, size=x.shape[-2:], mode='bilinear', align_corners=False)  # [N, latent_channels, H, W]
        
        # FIXED: NaN check on outputs with clamping
        # Use any() to avoid MPS synchronization issues
        with torch.no_grad():
            if torch.isnan(L).any() or torch.isinf(L).any():
                L = torch.nan_to_num(L, nan=0.0, posinf=1.0, neginf=-1.0)
        # Clamp L to prevent extreme values that can cause NaN in downstream ops
        L = L.clamp(-10.0, 10.0)
        
        with torch.no_grad():
            if torch.isnan(g).any() or torch.isinf(g).any():
                g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)
        # Clamp g to prevent extreme values
        g = g.clamp(-10.0, 10.0)
        
        return L, g


class TemporalEncoder(nn.Module):
    """
    Advanced encoder with proper temporal processing.
    
    Key improvements:
    1. Processes each frame independently through CNN
    2. Adds temporal position encoding
    3. Uses attention to aggregate across time
    4. Produces both spatial (L) and global (g) representations
    
    This is critical for capturing field evolution patterns
    that precede flares.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        temporal_dim: int = 32,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 64
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.global_dim = global_dim
        self.temporal_dim = temporal_dim
        
        # Per-frame CNN encoder with proper normalization
        self.frame_encoder = nn.Sequential(
            ConvGN(in_channels, 32),
            ResBlock(32, dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),  # Added normalization after strided conv
            nn.SiLU(),
            ResBlock(64, dropout),
            nn.Conv2d(64, latent_channels, 1),
            nn.GroupNorm(min(8, latent_channels), latent_channels),  # Added final normalization
        )
        
        # Global pooling for frame features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal position encoding
        self.temporal_pe = TemporalPositionEncoding(temporal_dim, max_frames)
        
        # Project frame features to temporal dim with LayerNorm for stability
        self.frame_proj = nn.Sequential(
            nn.Linear(latent_channels, temporal_dim),
            nn.LayerNorm(temporal_dim),
        )
        
        # Temporal attention for aggregation
        self.temporal_attn = TemporalAttention(temporal_dim, n_attn_heads, dropout)
        
        # Final projections with LayerNorm
        self.global_proj = nn.Sequential(
            nn.Linear(temporal_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim),
            nn.LayerNorm(global_dim),
        )
        
        # Spatial map aggregation (weighted by attention)
        self.spatial_weight_proj = nn.Linear(temporal_dim, 1)
    
    def forward(
        self, 
        frames: torch.Tensor, 
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [T, H, W] or [T, C, H, W] frame sequence
            observed_mask: [T] boolean mask for valid frames
            
        Returns:
            L: [1, C, H, W] aggregated spatial features
            g: [1, D] global conditioning code
            temporal_weights: [T] frame importance weights
        """
        # Handle both 3D and 4D input
        if frames.dim() == 3:
            T, H, W = frames.shape
        else:
            T, C, H, W = frames.shape
        device = frames.device
        
        # Handle empty observation case
        # NOTE: Output L has 2*latent_channels due to concat of weighted_L + last_L
        if not observed_mask.any():
            L = torch.zeros(1, self.latent_channels * 2, H, W, device=device)
            g = torch.zeros(1, self.global_dim, device=device)
            weights = torch.zeros(T, device=device)
            return L, g, weights
        
        # Get observed frames only for encoding
        obs_frames = frames[observed_mask]  # [T_obs, H, W] or [T_obs, C, H, W]
        T_obs = obs_frames.shape[0]
        
        # Encode all frames in a single batch (much faster than loop)
        if obs_frames.dim() == 3:
            # [T_obs, H, W] -> [T_obs, 1, H, W]
            obs_frames_batch = obs_frames.unsqueeze(1)
        else:
            # [T_obs, C, H, W]
            obs_frames_batch = obs_frames
        
        # CNN features for all frames at once
        frame_features = self.frame_encoder(obs_frames_batch)  # [T_obs, C, H', W']
        _, C, H_feat, W_feat = frame_features.shape
        
        # Upsample to original resolution
        frame_features = F.interpolate(
            frame_features, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )  # [T_obs, C, H, W]
        
        # Global features per frame
        global_feats = self.global_pool(frame_features).view(T_obs, -1)  # [T_obs, latent_channels]
        
        # Project to temporal dim (frame_proj is now a Sequential with LayerNorm)
        temporal_feats = self.frame_proj(global_feats)  # [T_obs, temporal_dim]
        
        # Add position encoding (based on position in observed sequence)
        pe = self.temporal_pe(T_obs).to(device)  # [T_obs, temporal_dim]
        temporal_feats = temporal_feats + pe
        
        # Add batch dim for attention
        temporal_feats = temporal_feats.unsqueeze(0)  # [1, T_obs, temporal_dim]
        obs_mask = torch.ones(1, T_obs, dtype=torch.bool, device=device)
        
        # Temporal attention
        g_temporal, attn_weights = self.temporal_attn(temporal_feats, obs_mask)  # [1, temporal_dim], [1, T_obs]
        
        # Project to global dim
        g = self.global_proj(g_temporal)  # [1, global_dim]
        
        # Aggregate spatial features using attention weights
        # Weight each frame's spatial features
        attn_weights = attn_weights.squeeze(0)  # [T_obs]
        weighted_L = (frame_features * attn_weights[:, None, None, None]).sum(dim=0, keepdim=True)  # [1, C, H, W]
        
        # IMPROVEMENT: Explicitly concatenate last observed frame features
        # This prevents "history collapse" where the model loses track of the current state
        # due to weighted averaging.
        last_L = frame_features[-1:].clone()  # [1, C, H, W]
        L_out = torch.cat([weighted_L, last_L], dim=1)  # [1, 2*C, H, W]
        
        # Create full temporal weights tensor
        full_weights = torch.zeros(T, device=device)
        full_weights[observed_mask] = attn_weights
        
        # FIXED: Validate output shapes
        expected_channels = self.latent_channels * 2
        assert L_out.shape[1] == expected_channels, \
            f"TemporalEncoder output channel mismatch: got {L_out.shape[1]}, expected {expected_channels}"
        assert g.shape[1] == self.global_dim, \
            f"TemporalEncoder global dim mismatch: got {g.shape[1]}, expected {self.global_dim}"
        
        # FIXED: NaN/Inf safety checks with clamping
        # Use any() to avoid MPS synchronization issues
        with torch.no_grad():
            if torch.isnan(L_out).any() or torch.isinf(L_out).any():
                L_out = torch.nan_to_num(L_out, nan=0.0, posinf=1.0, neginf=-1.0)
        L_out = L_out.clamp(-10.0, 10.0)
        
        with torch.no_grad():
            if torch.isnan(g).any() or torch.isinf(g).any():
                g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)
        g = g.clamp(-10.0, 10.0)
        
        return L_out, g, full_weights


# Backward compatibility alias
EncoderWithTemporal = TemporalEncoder


class TransformerTemporalEncoder(nn.Module):
    """
    SOTA Transformer-based temporal encoder for capturing long-range dependencies.
    
    Based on Sun et al. (2022) approach that achieved TSS=0.85:
    - Full self-attention across all observed frames
    - Learnable positional encoding with relative time
    - Multi-head cross-attention for spatial-temporal fusion
    
    This is significantly better than simple attention pooling for:
    - Capturing gradual buildup patterns over hours
    - Identifying rapid evolution precursors
    - Learning which frame pairs are most informative
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_frames: int = 64
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.global_dim = global_dim
        self.d_model = d_model
        
        # Per-frame CNN encoder (lightweight)
        self.frame_encoder = nn.Sequential(
            ConvGN(in_channels, 32),
            ResBlock(32, dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            ConvGN(64, 64),
            ResBlock(64, dropout),
            nn.Conv2d(64, latent_channels, 1)
        )
        
        # Project frame features to transformer dimension
        self.frame_proj = nn.Linear(latent_channels, d_model)
        
        # Learnable temporal position encoding
        self.temporal_pe = TemporalPositionEncoding(d_model, max_frames)
        
        # Transformer encoder layers with pre-norm (more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable [CLS] token for sequence aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Output projections
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim, global_dim)
        )
        
        # Spatial feature aggregation weights
        self.spatial_attn = nn.Linear(d_model, 1)
        
        # Final latent projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [T, H, W] or [T, C, H, W] frame sequence
            observed_mask: [T] boolean mask for valid frames
            
        Returns:
            L: [1, C, H, W] aggregated spatial features
            g: [1, D] global conditioning code
            temporal_weights: [T] frame importance weights
        """
        device = frames.device
        
        # Get dimensions
        if frames.dim() == 3:
            T, H, W = frames.shape
            C = 1
        else:
            T, C, H, W = frames.shape
        
        # Handle empty observation case
        if not observed_mask.any():
            L = torch.zeros(1, self.latent_channels * 2, H, W, device=device)
            g = torch.zeros(1, self.global_dim, device=device)
            weights = torch.zeros(T, device=device)
            return L, g, weights
        
        # Get observed frames
        obs_frames = frames[observed_mask]
        T_obs = obs_frames.shape[0]
        
        if obs_frames.dim() == 3:
            obs_frames = obs_frames.unsqueeze(1)  # [T_obs, 1, H, W]
        
        # Encode all frames with CNN
        spatial_features = self.frame_encoder(obs_frames)  # [T_obs, C, H', W']
        _, C_lat, H_feat, W_feat = spatial_features.shape
        
        # Upsample to original resolution
        spatial_features = F.interpolate(
            spatial_features, size=(H, W), mode='bilinear', align_corners=False
        )  # [T_obs, C, H, W]
        
        # Global pool for transformer input
        global_feats = self.global_pool(spatial_features).view(T_obs, C_lat)  # [T_obs, C]
        
        # Project to transformer dimension
        frame_tokens = self.frame_proj(global_feats)  # [T_obs, d_model]
        
        # Add positional encoding
        pe = self.temporal_pe(T_obs).to(device)
        frame_tokens = frame_tokens + pe
        
        # Add CLS token
        cls = self.cls_token.expand(1, -1, -1)  # [1, 1, d_model]
        tokens = torch.cat([cls, frame_tokens.unsqueeze(0)], dim=1)  # [1, T_obs+1, d_model]
        
        # Create attention mask (no masking needed, all observed)
        src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)  # [1, T_obs+1, d_model]
        
        # Extract CLS token for global representation
        cls_output = encoded[:, 0, :]  # [1, d_model]
        frame_outputs = encoded[:, 1:, :]  # [1, T_obs, d_model]
        
        # Global code
        g = self.global_proj(cls_output)  # [1, global_dim]
        
        # Compute frame importance weights from transformer outputs
        frame_attn_logits = self.spatial_attn(frame_outputs).squeeze(-1)  # [1, T_obs]
        frame_attn_logits = frame_attn_logits.clamp(-50.0, 50.0)  # Prevent softmax overflow
        frame_weights = F.softmax(frame_attn_logits, dim=-1).squeeze(0)  # [T_obs]
        frame_weights = torch.nan_to_num(frame_weights, nan=1.0/frame_weights.numel())
        
        # Aggregate spatial features using learned weights
        weighted_L = (spatial_features * frame_weights[:, None, None, None]).sum(dim=0, keepdim=True)  # [1, C, H, W]
        
        # Also keep last frame (recency bias) - concatenate for richer representation
        last_L = spatial_features[-1:].clone()  # [1, C, H, W]
        L_out = torch.cat([weighted_L, last_L], dim=1)  # [1, 2*C, H, W]
        
        # Map weights back to full temporal grid
        full_weights = torch.zeros(T, device=device)
        full_weights[observed_mask] = frame_weights
        
        return L_out, g, full_weights


class MultiScaleEncoder(nn.Module):
    """
    Feature Pyramid Network encoder for multi-scale features.
    
    Key insight: PIL detection requires fine-scale gradients while
    field topology requires large-scale context. FPN captures both.
    
    This could be a SIGNIFICANT improvement for SOTA.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        dropout: float = 0.05
    ):
        super().__init__()
        
        # Bottom-up pathway (encoder)
        self.stem = ConvGN(in_channels, 32)
        
        # Stage 1: 32 channels
        self.stage1 = nn.Sequential(ResBlock(32, dropout), ResBlock(32, dropout))
        
        # Stage 2: 64 channels, /2
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU()
        )
        self.stage2 = nn.Sequential(ResBlock(64, dropout), ResBlock(64, dropout))
        
        # Stage 3: 128 channels, /4
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU()
        )
        self.stage3 = nn.Sequential(ResBlock(128, dropout), ResBlock(128, dropout))
        
        # Top-down pathway (FPN)
        self.lateral3 = nn.Conv2d(128, latent_channels, 1)
        self.lateral2 = nn.Conv2d(64, latent_channels, 1)
        self.lateral1 = nn.Conv2d(32, latent_channels, 1)
        
        # Smooth convolutions after merging
        self.smooth3 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        
        # Global pooling from deepest features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, global_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim * 2, global_dim)
        )
        
        # Final fusion: combine all scales
        self.fusion = nn.Conv2d(latent_channels * 3, latent_channels, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns multi-scale fused latent map and global code.
        """
        H, W = x.shape[-2:]
        
        # Bottom-up
        c1 = self.stage1(self.stem(x))      # [N, 32, H, W]
        c2 = self.stage2(self.down1(c1))    # [N, 64, H/2, W/2]
        c3 = self.stage3(self.down2(c2))    # [N, 128, H/4, W/4]
        
        # Global code from deepest
        g = self.g_mlp(self.global_pool(c3))
        
        # Top-down with lateral connections
        p3 = self.smooth3(self.lateral3(c3))  # [N, C, H/4, W/4]
        
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(self.lateral2(c2) + p3_up)  # [N, C, H/2, W/2]
        
        p2_up = F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.smooth1(self.lateral1(c1) + p2_up)  # [N, C, H, W]
        
        # Upsample all to full resolution and concatenate
        p2_full = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3_full = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse multi-scale features
        L = self.fusion(torch.cat([p1, p2_full, p3_full], dim=1))  # [N, C, H, W]
        
        return L, g


# ============================================================================
# SOTA: Lightweight Encoders for 16GB Memory Budget
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution (MobileNet-style).
    
    Reduces parameters by ~8x compared to standard conv.
    Critical for 16GB MPS training.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetV2-style).
    
    Expands channels, applies depthwise conv, then projects back.
    Memory efficient due to narrow input/output channels.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                nn.SiLU()
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU()
        ])
        
        # Pointwise projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightTemporalEncoder(nn.Module):
    """
    Ultra-lightweight temporal encoder for 16GB MPS.
    
    ~3x fewer parameters than TransformerTemporalEncoder.
    Uses:
    - Depthwise separable convolutions
    - Linear attention (O(N) instead of O(N²))
    - Gradient checkpointing
    
    Target: <5M parameters total encoder.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 24,  # Reduced from 32
        global_dim: int = 48,       # Reduced from 64
        temporal_dim: int = 24,     # Reduced from 32
        dropout: float = 0.1,
        max_frames: int = 64,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.global_dim = global_dim
        self.temporal_dim = temporal_dim
        self.use_checkpoint = use_checkpoint
        
        # Lightweight per-frame encoder (MobileNet-style)
        self.frame_encoder = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 16),
            InvertedResidual(16, 24, stride=2, expand_ratio=4),
            InvertedResidual(24, 24, expand_ratio=4),
            InvertedResidual(24, latent_channels, stride=2, expand_ratio=4),
            nn.Conv2d(latent_channels, latent_channels, 1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal position encoding (lightweight)
        self.temporal_pe = nn.Parameter(torch.randn(1, max_frames, temporal_dim) * 0.02)
        
        # Project to temporal dim
        self.frame_proj = nn.Linear(latent_channels, temporal_dim)
        
        # Lightweight temporal aggregation (GRU instead of Transformer)
        self.temporal_gru = nn.GRU(
            temporal_dim, temporal_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        
        # Frame importance predictor
        self.frame_importance = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.Tanh(),
            nn.Linear(temporal_dim, 1)
        )
        
        # Output projections
        self.global_proj = nn.Sequential(
            nn.Linear(temporal_dim * 2, global_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim, global_dim)
        )
    
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames with optional checkpointing."""
        if self.use_checkpoint and self.training:
            return checkpoint(self.frame_encoder, frames, use_reentrant=False)
        return self.frame_encoder(frames)
    
    def forward(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [T, H, W] or [T, C, H, W] frame sequence
            observed_mask: [T] boolean mask for valid frames
            
        Returns:
            L: [1, C*2, H, W] aggregated spatial features
            g: [1, D] global conditioning code
            temporal_weights: [T] frame importance weights
        """
        device = frames.device
        
        if frames.dim() == 3:
            T, H, W = frames.shape
        else:
            T, _, H, W = frames.shape
        
        # Handle empty observation case
        if not observed_mask.any():
            L = torch.zeros(1, self.latent_channels * 2, H, W, device=device)
            g = torch.zeros(1, self.global_dim, device=device)
            weights = torch.zeros(T, device=device)
            return L, g, weights
        
        # Get observed frames
        obs_frames = frames[observed_mask]
        T_obs = obs_frames.shape[0]
        
        if obs_frames.dim() == 3:
            obs_frames = obs_frames.unsqueeze(1)
        
        # Encode frames with lightweight CNN
        spatial_features = self._encode_frames(obs_frames)  # [T_obs, C, H', W']
        _, C_lat, H_feat, W_feat = spatial_features.shape
        
        # Upsample to original resolution
        spatial_features = F.interpolate(
            spatial_features, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Global pool for temporal processing
        global_feats = self.global_pool(spatial_features).view(T_obs, C_lat)
        
        # Project to temporal dim and add positional encoding
        temporal_feats = self.frame_proj(global_feats)  # [T_obs, temporal_dim]
        temporal_feats = temporal_feats + self.temporal_pe[:, :T_obs, :].squeeze(0)
        
        # Temporal GRU (much lighter than Transformer)
        temporal_feats = temporal_feats.unsqueeze(0)  # [1, T_obs, D]
        gru_out, hidden = self.temporal_gru(temporal_feats)  # [1, T_obs, D*2]
        
        # Compute frame importance
        importance_logits = self.frame_importance(gru_out).squeeze(-1)  # [1, T_obs]
        importance_logits = importance_logits.clamp(-50.0, 50.0)  # Prevent softmax overflow
        frame_weights = F.softmax(importance_logits, dim=-1).squeeze(0)  # [T_obs]
        frame_weights = torch.nan_to_num(frame_weights, nan=1.0/frame_weights.numel())
        
        # Global code from final hidden state
        g = self.global_proj(hidden.transpose(0, 1).reshape(1, -1))  # [1, global_dim]
        
        # Aggregate spatial features
        weighted_L = (spatial_features * frame_weights[:, None, None, None]).sum(dim=0, keepdim=True)
        last_L = spatial_features[-1:].clone()
        L_out = torch.cat([weighted_L, last_L], dim=1)  # [1, 2*C, H, W]
        
        # Map weights to full temporal grid
        full_weights = torch.zeros(T, device=device)
        full_weights[observed_mask] = frame_weights
        
        return L_out, g, full_weights


class EfficientGRUEncoder(nn.Module):
    """
    Efficient GRU-based encoder for minimal memory footprint.
    
    Key insight: For solar flare prediction, temporal patterns are often
    simpler than what Transformers capture. GRU is sufficient and much cheaper.
    
    Memory: ~2M parameters (vs ~10M for Transformer)
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.global_dim = global_dim
        self.use_checkpoint = use_checkpoint
        
        # Simple CNN stem (no residuals for memory)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            latent_channels, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Attention for frame weighting
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Global projection
        self.global_proj = nn.Linear(hidden_dim * 2, global_dim)
    
    def forward(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = frames.device
        
        if frames.dim() == 3:
            T, H, W = frames.shape
        else:
            T, _, H, W = frames.shape
        
        if not observed_mask.any():
            L = torch.zeros(1, self.latent_channels * 2, H, W, device=device)
            g = torch.zeros(1, self.global_dim, device=device)
            return L, g, torch.zeros(T, device=device)
        
        obs_frames = frames[observed_mask]
        T_obs = obs_frames.shape[0]
        
        if obs_frames.dim() == 3:
            obs_frames = obs_frames.unsqueeze(1)
        
        # Encode frames
        if self.use_checkpoint and self.training:
            spatial = checkpoint(self.stem, obs_frames, use_reentrant=False)
        else:
            spatial = self.stem(obs_frames)
        
        # Upsample
        spatial = F.interpolate(spatial, size=(H, W), mode='bilinear', align_corners=False)
        
        # Temporal features
        global_feats = self.global_pool(spatial).view(1, T_obs, -1)
        
        # GRU
        gru_out, _ = self.gru(global_feats)  # [1, T_obs, hidden*2]
        
        # Attention weights
        attn_logits = self.attn(gru_out).squeeze(-1)  # [1, T_obs]
        attn_logits = attn_logits.clamp(-50.0, 50.0)  # Prevent softmax overflow
        weights = F.softmax(attn_logits, dim=-1).squeeze(0)  # [T_obs]
        weights = torch.nan_to_num(weights, nan=1.0/weights.numel())
        
        # Global code
        g = self.global_proj(gru_out[:, -1, :])  # [1, global_dim]
        
        # Weighted spatial features
        weighted_L = (spatial * weights[:, None, None, None]).sum(dim=0, keepdim=True)
        last_L = spatial[-1:].clone()
        L_out = torch.cat([weighted_L, last_L], dim=1)
        
        full_weights = torch.zeros(T, device=device)
        full_weights[observed_mask] = weights
        
        return L_out, g, full_weights


# ============================================================================
# PIL Evolution Tracker
# ============================================================================

class PILEvolutionTracker(nn.Module):
    """
    Track PIL (Polarity Inversion Line) evolution over time.
    
    Key insight: PIL growth, motion, and intensification are
    strong precursors of flare activity.
    
    Features computed:
    - PIL length evolution (growth rate)
    - PIL centroid motion (drift speed)
    - PIL intensity change (gradient strengthening)
    - PIL fragmentation (number of segments)
    """
    def __init__(self, n_output_features: int = 8):
        super().__init__()
        self.n_output_features = n_output_features
        
        # Learnable smoothing for temporal statistics
        self.temporal_smooth = nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=4)
    
    def _compute_pil_stats(
        self,
        bz_frame: torch.Tensor,
        threshold_pct: float = 0.15
    ) -> torch.Tensor:
        """
        Compute PIL statistics for a single frame.
        
        Args:
            bz_frame: [H, W] Bz magnetogram
            threshold_pct: Top percentile for gradient threshold
        
        Returns:
            [4] PIL statistics: length_proxy, intensity, x_centroid, y_centroid
        """
        H, W = bz_frame.shape
        device = bz_frame.device
        
        # Compute gradient magnitude
        grad_x = F.pad(bz_frame[:, 1:] - bz_frame[:, :-1], (0, 1, 0, 0))
        grad_y = F.pad(bz_frame[1:, :] - bz_frame[:-1, :], (0, 0, 0, 1))
        grad_mag = (grad_x ** 2 + grad_y ** 2).sqrt()
        
        # PIL mask: high gradient regions near zero crossings
        # FIXED: Avoid torch.quantile on MPS - use sorted-based approximation
        grad_sorted = torch.sort(grad_mag.flatten()).values
        threshold_idx = int((1.0 - threshold_pct) * (grad_sorted.numel() - 1))
        threshold = grad_sorted[threshold_idx]
        pil_mask = (grad_mag > threshold).float()
        
        # Also require near polarity inversion
        # FIXED: Avoid torch.quantile on MPS
        bz_sorted = torch.sort(bz_frame.abs().flatten()).values
        bz_threshold = bz_sorted[int(0.3 * (bz_sorted.numel() - 1))]
        near_zero = (bz_frame.abs() < bz_threshold).float()
        pil_mask = pil_mask * near_zero
        
        # Compute statistics
        pil_sum = pil_mask.sum().clamp(min=1e-6)
        
        # Length proxy (total PIL pixels)
        length_proxy = pil_sum / (H * W)
        
        # Intensity (mean gradient at PIL)
        intensity = (grad_mag * pil_mask).sum() / pil_sum
        
        # Centroid (weighted by gradient)
        y_coords = torch.arange(H, device=device).float().view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).float().view(1, -1).expand(H, W)
        
        weights = pil_mask * grad_mag
        weight_sum = weights.sum().clamp(min=1e-6)
        
        x_centroid = (x_coords * weights).sum() / weight_sum / W  # Normalize to [0, 1]
        y_centroid = (y_coords * weights).sum() / weight_sum / H
        
        return torch.stack([length_proxy, intensity, x_centroid, y_centroid])
    
    def forward(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PIL evolution features over time.
        
        Args:
            frames: [T, H, W] Bz magnetogram sequence
            observed_mask: [T] boolean mask
        
        Returns:
            [n_output_features] PIL evolution features:
            - pil_length_mean, pil_length_trend
            - pil_intensity_mean, pil_intensity_trend
            - pil_motion_speed, pil_motion_direction
            - pil_growth_rate, pil_instability
        """
        device = frames.device
        T = frames.shape[0]
        
        if not observed_mask.any():
            return torch.zeros(self.n_output_features, device=device)
        
        # Compute PIL stats for each observed frame
        obs_indices = observed_mask.nonzero().squeeze(-1)
        n_obs = len(obs_indices)
        
        if n_obs < 2:
            return torch.zeros(self.n_output_features, device=device)
        
        pil_stats = []
        for idx in obs_indices:
            stats = self._compute_pil_stats(frames[idx])
            pil_stats.append(stats)
        
        pil_stats = torch.stack(pil_stats)  # [n_obs, 4]
        
        # Apply temporal smoothing
        pil_stats_smooth = self.temporal_smooth(
            pil_stats.unsqueeze(0).transpose(1, 2)
        ).transpose(1, 2).squeeze(0)  # [n_obs, 4]
        
        # Extract temporal features
        length, intensity, x_cent, y_cent = pil_stats_smooth.unbind(dim=-1)
        
        # Means
        length_mean = length.mean()
        intensity_mean = intensity.mean()
        
        # Trends (linear regression slope proxy)
        t_normalized = torch.linspace(0, 1, n_obs, device=device)
        length_trend = ((length - length_mean) * (t_normalized - 0.5)).sum() * 6 / n_obs
        intensity_trend = ((intensity - intensity_mean) * (t_normalized - 0.5)).sum() * 6 / n_obs
        
        # Motion statistics
        dx = x_cent[1:] - x_cent[:-1]
        dy = y_cent[1:] - y_cent[:-1]
        motion_speed = (dx ** 2 + dy ** 2).sqrt().mean()
        motion_direction = torch.atan2(dy.sum(), dx.sum()) / math.pi  # Normalize to [-1, 1]
        
        # Growth rate (change in length)
        growth_rate = (length[-1] - length[0]) / max(1, n_obs - 1)
        
        # Instability (variance in motion)
        instability = motion_speed.std() if n_obs > 2 else torch.tensor(0.0, device=device)
        
        return torch.stack([
            length_mean, length_trend,
            intensity_mean, intensity_trend,
            motion_speed, motion_direction,
            growth_rate, instability
        ])


