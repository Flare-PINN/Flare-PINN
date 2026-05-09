# src/models/pinn/config.py
"""
Pydantic configuration models for PINN training.
Provides type-safe, validated configuration management.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class FourierConfig(BaseModel):
    """Fourier feature configuration."""
    max_log2_freq: int = Field(default=5, ge=1, le=10, description="Maximum log2 frequency for Fourier features")
    ramp_frac: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of training for frequency ramping")
    
    @field_validator('max_log2_freq')
    @classmethod
    def validate_freq(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_log2_freq must be at least 1")
        return v


class EncoderConfig(BaseModel):
    """CNN/Temporal encoder configuration."""
    type: Literal["tiny", "temporal", "transformer", "multiscale", "lightweight", "gru"] = Field(
        default="temporal",
        description="Encoder type: tiny (simple), temporal (attention), transformer (SOTA), multiscale (FPN), lightweight (MobileNet-style), gru (memory-efficient)"
    )
    latent_channels: int = Field(default=32, ge=16, le=128)
    global_dim: int = Field(default=64, ge=32, le=256)
    n_transformer_layers: int = Field(default=3, ge=1, le=6, description="Transformer encoder layers (for transformer type)")
    n_heads: int = Field(default=4, ge=1, le=8, description="Attention heads")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_checkpoint: bool = Field(default=True, description="Use gradient checkpointing to reduce memory")


class ModelConfig(BaseModel):
    """PINN backbone configuration."""
    model_config = {"protected_namespaces": ()}
    
    model_type: Literal["mlp", "hybrid", "cnn_gru_direct"] = Field(default="mlp", description="Model architecture: 'mlp' (pure coordinate), 'hybrid' (CNN+MLP), or 'cnn_gru_direct' (standalone CNN-GRU)")
    # in_channels is now typically derived from data.components, but kept here for manual override/back-compat
    in_channels: Optional[int] = Field(default=None, description="Input channels (1=Bz, 3=Bx,By,Bz). If None, derived from data config.")
    hidden: int = Field(default=384, ge=64, le=2048, description="Hidden dimension for MLP")
    layers: int = Field(default=10, ge=2, le=20, description="Number of hidden layers")
    learn_eta: bool = Field(default=False, description="Learn spatially-varying resistivity η(x,y,t)")
    eta_scalar: float = Field(default=0.01, ge=1e-6, le=1.0, description="Fixed scalar resistivity if not learning")
    fourier: FourierConfig = Field(default_factory=FourierConfig)
    vector_B: bool = Field(
        default=False, 
        description="Enable 3-component B field (Bx, By, Bz) with full vector induction physics"
    )
    hard_div_free: bool = Field(
        default=False,
        description="Enforce div B = 0 by defining B = curl A (requires vector_B=True)"
    )
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)


class ClassifierConfig(BaseModel):
    """Flare prediction classifier head configuration."""
    hidden: int = Field(default=256, ge=64, le=1024)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    horizons: tuple[int, ...] = Field(default=(6, 12, 24), description="Prediction horizons in hours")
    loss_type: Literal["bce", "focal", "asymmetric", "cb_focal", "poly_focal"] = Field(
        default="cb_focal",  # SOTA: class-balanced focal for severe imbalance
        description="Loss function: bce, focal, asymmetric, cb_focal (class-balanced), poly_focal"
    )
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0, description="Focal loss alpha (weight for positive class)")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0, description="Focal loss gamma (focusing parameter)")
    asymmetric_gamma_neg: float = Field(default=4.0, ge=0.0, le=8.0, description="Asymmetric focal loss gamma for negatives")
    pos_weight: Optional[float] = Field(default=None, ge=1.0, description="Positive class weight for BCE")
    cb_beta: float = Field(default=0.9999, ge=0.9, le=0.99999, description="Class-balanced loss beta (higher = more weighting)")
    poly_epsilon: float = Field(default=1.0, ge=0.0, le=5.0, description="PolyLoss epsilon coefficient")
    use_rf_guidance: bool = Field(default=False, description="Use Random Forest feature importance weighting")
    rf_weights_path: Optional[Path] = Field(default=None, description="Path to pre-computed RF importance weights pickle")
    use_attention: bool = Field(default=True, description="Use spatial and temporal attention in classifier")
    use_physics_features: bool = Field(default=True, description="Use physics-derived features in classifier")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.2, description="Label smoothing factor for focal loss")
    confidence_penalty: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight for confidence/entropy regularization")
    gradient_penalty: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight for gradient penalty (Lipschitz regularization)")
    mixup_alpha: float = Field(default=0.0, ge=0.0, le=1.0, description="Mixup augmentation alpha (0 = disabled)")
    
    # SOTA: Uncertainty quantification
    mc_dropout: bool = Field(
        default=False,
        description="Enable Monte Carlo Dropout for uncertainty quantification during inference"
    )
    mc_samples: int = Field(
        default=10, ge=3, le=50,
        description="Number of MC dropout samples for uncertainty estimation"
    )
    mc_dropout_rate: float = Field(
        default=0.2, ge=0.1, le=0.5,
        description="Dropout rate for MC Dropout (can be different from training dropout)"
    )
    
    # SOTA: Calibration
    use_temperature_scaling: bool = Field(
        default=True,
        description="Apply temperature scaling for probability calibration (post-training)"
    )
    
    # Distribution shift handling
    use_domain_adaptation: bool = Field(
        default=False,
        description="Enable domain adaptation layers for solar cycle shift (cycle 24 -> 25)"
    )
    
    @field_validator('rf_weights_path', mode='before')
    @classmethod
    def validate_rf_path(cls, v):
        if v is not None:
            return Path(v)
        return v


class PhysicsConfig(BaseModel):
    """Physics-informed loss configuration."""
    enable: bool = Field(default=True, description="Enable physics loss")
    resistive: bool = Field(default=False, description="Include resistive (diffusion) terms")
    boundary_terms: bool = Field(default=False, description="Include boundary integration terms")
    lambda_phys_schedule: list[list[float]] = Field(
        default=[[0.0, 0.0], [0.3, 0.0], [0.8, 2.0], [1.0, 3.0]],
        description="Piecewise-linear schedule [[frac, weight], ...] for physics loss ramp-up"
    )
    
    # Vector physics options (only used when model.vector_B=True)
    div_B_weight: float = Field(
        default=1.0, 
        ge=0.0, 
        le=10.0,
        description="Weight for solenoidal constraint ∇·B=0 (vector mode only)"
    )
    enforce_div_free_u: bool = Field(
        default=False,
        description="Enforce divergence-free velocity (incompressible flow constraint)"
    )
    component_weights: tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="Weights for (Bx, By, Bz) component losses (vector mode only)"
    )
    
    # SOTA: Advanced physics options
    #   IMPORTANT: use_uncertainty_weighting and use_gradnorm are MUTUALLY EXCLUSIVE!
    # - use_uncertainty_weighting: Per-component learned weights within physics (Kendall et al. 2018)
    # - use_gradnorm: Per-task balancing (cls vs physics) via gradient norms (Chen et al. 2018)
    # If both enabled, use_gradnorm takes precedence and disables uncertainty weighting.
    use_uncertainty_weighting: bool = Field(
        default=False,  # FIXED: Changed default to False - prefer GradNorm for multi-task
        description="Use learned uncertainty weighting for physics component balancing (Kendall et al. 2018). Disabled when use_gradnorm=True."
    )
    
    # LRA: Learning Rate Annealing (Wang et al. 2021)
    use_lra: bool = Field(
        default=False,
        description="Use Learning Rate Annealing for automatic physics/data gradient balancing"
    )
    lra_alpha: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="LRA moving average coefficient for gradient statistics"
    )
    lra_update_freq: int = Field(
        default=1,
        ge=1,
        description="How often to update LRA weights (every N steps)"
    )
    
    # GradNorm: Gradient Normalization (Chen et al. 2018)
    # More principled than LRA - balances gradient scales across tasks
    use_gradnorm: bool = Field(
        default=False,
        description="Use GradNorm for multi-task gradient balancing (better than LRA)"
    )
    gradnorm_alpha: float = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="GradNorm restoring force (α): higher = stronger equalization"
    )
    gradnorm_update_freq: int = Field(
        default=5,
        ge=1,
        description="How often to update GradNorm weights (every N steps)"
    )
    
    # Causal Training: Temporal weighting for time-dependent PDEs
    #   There are TWO causal mechanisms - only use ONE at a time:
    #   1. use_causal_training: Weights collocation points BEFORE physics (in hybrid_model.py)
    #   2. use_causal_weighting: Weights residuals AFTER physics (in physics.py)
    # Using both will cause double-weighting! Prefer use_causal_training for stability.
    use_causal_training: bool = Field(
        default=False,
        description="Weight collocation points by temporal causality BEFORE physics computation. Earlier times (t≈0) get higher weight. Mutually exclusive with use_causal_weighting!"
    )
    causal_decay: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Exponential decay rate for use_causal_training: w(t) = exp(-decay * t_norm). Higher = stronger early-time preference."
    )
    enforce_force_free: bool = Field(
        default=False,
        description="Add force-free constraint (J × B ≈ 0) for low-β corona"
    )
    force_free_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for force-free constraint"
    )

    # SOTA: Causal Weighting (Wang et al., 2022) - Alternative approach
    #   This is DIFFERENT from use_causal_training above!
    # use_causal_weighting: Weights residuals AFTER physics computation based on cumulative loss
    # use_causal_training: Weights collocation points BEFORE physics (simpler, more stable)
    # Only use ONE of these, not both!
    use_causal_weighting: bool = Field(
        default=False,
        description="Weight residuals AFTER physics computation based on cumulative loss at earlier times. Mutually exclusive with use_causal_training! Prefer use_causal_training for stability."
    )
    causal_tol: float = Field(
        default=1.0,
        description="Tolerance parameter for use_causal_weighting: w(t) = exp(-tol * cumsum_loss(<t)). Higher = stronger early-time preference."
    )
    
    # Gradient stability
    enable_gradient_clamping: bool = Field(
        default=True,
        description="Enable soft clamping of spatial gradients (prevents explosion)"
    )
    gradient_clamp_value: float = Field(
        default=100.0,
        description="Max value for gradient clamping"
    )
    
    # SOTA: Curriculum learning for physics
    curriculum_strategy: Literal["linear", "exponential", "adaptive", "warmup_hold"] = Field(
        default="warmup_hold",
        description="Physics loss curriculum strategy: linear (gradual), exponential (accelerating), adaptive (loss-based), warmup_hold (delay then constant)"
    )
    warmup_fraction: float = Field(
        default=0.3, ge=0.0, le=0.8,
        description="Fraction of training to delay/warmup physics loss"
    )
    
    # Gradient scaling for physics loss (prevents dominating classification)
    physics_grad_scale: float = Field(
        default=0.5, ge=0.001, le=2.0,
        description="Scale factor for physics gradients (lower = less interference with classification)"
    )
    # Strong-form PINN baseline (experimental)
    use_strong_form: bool = Field(
        default=False,
        description="Use strong-form (pointwise) PDE residuals instead of weak-form (integral). For baseline comparison only - typically unstable on noisy data."
    )
    strong_form_use_huber: bool = Field(
        default=True,
        description="Use pseudo-Huber loss for strong-form (more stable than MSE)"
    )
    strong_form_huber_delta: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Huber delta parameter for strong-form"
    )
    strong_form_edge_margin: int = Field(
        default=0, ge=0, le=20,
        description="Pixels to exclude from edges in strong-form (noisy boundary handling)"
    )
    strong_form_chunk_size: int = Field(
        default=256, ge=64, le=100000,
        description="Chunk size for strong-form gradient computation (memory management). Set very high to disable chunking."
    )
    
    # Collocation point reduction for memory
    max_collocation_during_warmup: int = Field(
        default=4096, ge=512, le=16384,
        description="Reduced collocation points during physics warmup (memory optimization)"
    )
    
    # MPS physics mode: create_graph=False is more stable on MPS but breaks physics training!
    # FIXED: Default to False because True disables gradient flow from physics residuals.
    mps_fast_physics: bool = Field(
        default=False,
        description="On MPS, disable create_graph in autograd. WARNING: Setting True prevents physics parameters from learning!"
    )
    
    @field_validator('lambda_phys_schedule')
    @classmethod
    def validate_schedule(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("lambda_phys_schedule cannot be empty")
        for point in v:
            if len(point) != 2:
                raise ValueError(f"Each schedule point must be [frac, weight], got {point}")
            if not (0.0 <= point[0] <= 1.0):
                raise ValueError(f"Schedule fraction must be in [0,1], got {point[0]}")
        return sorted(v, key=lambda p: p[0])
    
    @field_validator('component_weights', mode='before')
    @classmethod
    def validate_component_weights(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v
    
    @model_validator(mode='after')
    def validate_physics_conflicts(self) -> PhysicsConfig:
        """Validate physics configuration for conflicting options."""
        import warnings
        
        # Check causal training conflict
        if self.use_causal_training and self.use_causal_weighting:
            raise ValueError(
                "Cannot use both use_causal_training and use_causal_weighting! "
                "use_causal_training weights collocation points BEFORE physics (recommended). "
                "use_causal_weighting weights residuals AFTER physics. "
                "Using both causes double-weighting. Set one to False."
            )
        
        # GradNorm takes precedence over uncertainty weighting - just warn
        if self.use_gradnorm and self.use_uncertainty_weighting:
            warnings.warn(
                "Both use_gradnorm and use_uncertainty_weighting are enabled. "
                "GradNorm will handle task-level balancing; uncertainty weighting in physics module is redundant."
            )
        
        # LRA and GradNorm are MUTUALLY EXCLUSIVE - raise error
        if self.use_lra and self.use_gradnorm:
            raise ValueError(
                "Cannot enable both use_lra and use_gradnorm! "
                "They are mutually exclusive loss balancing mechanisms. "
                "Use GradNorm for multi-task learning (recommended) or LRA for gradient magnitude balancing."
            )
        
        return self


class EtaConfig(BaseModel):
    """Resistivity configuration."""
    min: float = Field(default=1e-4, ge=1e-10, le=1e-2)
    max: float = Field(default=1.0, ge=1e-2, le=10.0)
    tv_weight: float = Field(default=1e-3, ge=0.0, le=1.0, description="Total variation regularization weight")
    
    @model_validator(mode='after')
    def validate_bounds(self) -> EtaConfig:
        if self.min >= self.max:
            raise ValueError(f"eta.min ({self.min}) must be < eta.max ({self.max})")
        return self


class LossWeightsConfig(BaseModel):
    """Loss term weights."""
    cls: float = Field(default=1.0, ge=0.0, description="Classification loss weight")
    data: float = Field(default=1.0, ge=0.0, description="Data fitting (L1 on Bz) weight")
    curl_consistency: float = Field(default=0.1, ge=0.0, description="Curl consistency weight (optional)")


class CollocationConfig(BaseModel):
    """Collocation point sampling configuration."""
    n_max: int = Field(default=20000, ge=1000, le=100000, description="Maximum collocation points per batch")
    alpha_start: float = Field(default=0.5, ge=0.0, le=1.0, description="Initial PIL bias weight")
    alpha_end: float = Field(default=0.8, ge=0.0, le=1.0, description="Final PIL bias weight")
    impw_clip_quantile: float = Field(default=0.99, ge=0.5, le=0.999, description="Importance weight clipping quantile")
    
    # Adaptive Resampling (Residual-based RAR - McClenny & Braga-Neto 2020)
    use_adaptive_resampling: bool = Field(
        default=False,
        description="Resample collocation points based on physics residual magnitude"
    )
    adaptive_resample_freq: int = Field(
        default=10,
        ge=1,
        description="How often to update collocation distribution (every N steps)"
    )
    adaptive_keep_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Fraction of highest-residual points to keep (rest resampled uniformly)"
    )
    
    @model_validator(mode='after')
    def validate_alpha_progression(self) -> CollocationConfig:
        if self.alpha_start > self.alpha_end:
            raise ValueError(f"alpha_start ({self.alpha_start}) should be <= alpha_end ({self.alpha_end})")
        return self


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: Literal["cosine", "step", "constant"] = Field(default="cosine")
    warmup_steps: int = Field(default=1000, ge=0, le=50000)
    min_lr: float = Field(default=1e-6, ge=0.0, le=1e-2)


class SamplerConfig(BaseModel):
    """Data sampling configuration for class imbalance."""
    strategy: Literal["random", "class_balanced", "sqrt_balanced"] = Field(default="class_balanced")
    positive_multiplier: float = Field(default=10.0, ge=1.0, le=200.0, description="Oversampling multiplier for positive class (up to 200x for severe imbalance)")
    smoothing: float = Field(default=0.5, ge=0.0, le=1.0)


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    steps: int = Field(default=50000, ge=100, le=1000000)
    batch_size: int = Field(default=1, ge=1, le=64)
    num_workers: int = Field(default=0, ge=0, le=32, description="Number of DataLoader workers")
    lr: float = Field(default=1e-3, ge=1e-6, le=1e-1)
    grad_clip: float = Field(default=1.0, ge=0.0, le=100.0)
    amp: bool = Field(default=True, description="Automatic mixed precision")
    log_every: int = Field(default=25, ge=1)
    eval_every: int = Field(default=500, ge=10)
    checkpoint_every: int = Field(default=5000, ge=100)
    checkpoint_dir: Optional[Path] = Field(default=None, description="Directory for checkpoints")
    keep_last_n_checkpoints: int = Field(
        default=3, ge=1, le=100,
        description="Number of numbered checkpoints to retain alongside best_model.pt"
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    use_ema: bool = Field(default=True, description="Use Exponential Moving Average for model weights")
    ema_decay: float = Field(default=0.999, ge=0.9, le=0.9999, description="EMA decay rate")
    
    # Memory optimization settings (critical for 16GB MPS)
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, le=32,
        description="Gradient accumulation steps (effective batch = batch_size * accum_steps)"
    )
    memory_cleanup_every: int = Field(
        default=50, ge=10, le=1000,
        description="Clear memory cache every N steps (MPS optimization)"
    )
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Use gradient checkpointing to trade compute for memory"
    )
    max_collocation_points: int = Field(
        default=8192, ge=1024, le=65536,
        description="Max collocation points per sample (reduce for memory)"
    )
    
    # MPS auto-restart to clear memory leaks
    auto_restart_every: int = Field(
        default=0, ge=0, le=100000,
        description="Auto-restart every N steps to clear MPS memory (0=disabled)"
    )
    
    # LR Schedule behavior on resume
    fresh_lr_schedule: bool = Field(
        default=False,
        description="If True, reset LR schedule on resume (good for curriculum changes). If False, resume schedule from checkpoint."
    )
    
    # LR scheduler horizon override (for matching sweep schedules)
    lr_total_steps: Optional[int] = Field(
        default=None, ge=100, le=1000000,
        description="Override LR scheduler horizon (for reproducing sweeps). If None, uses train.steps"
    )
    
    # Plateau rollback settings
    plateau_patience: int = Field(default=3, ge=1, le=20, description="Consecutive val drops before rollback")
    plateau_lr_factor: float = Field(default=0.5, ge=0.1, le=0.9, description="LR multiplier on rollback")
    selection_metric: Literal["tss_24h", "mean_tss", "max_tss"] = Field(
        default="tss_24h",
        description="Validation metric used for checkpoint selection and plateau tracking."
    )
    
    # Freeze classifier (Option B: only train physics)
    freeze_classifier: bool = Field(default=False, description="Freeze classifier & encoder, only train physics heads")
    
    # SAM optimizer (experimental - high memory cost)
    use_sam: bool = Field(default=False, description="Use Sharpness-Aware Minimization (doubles memory)")
    sam_rho: float = Field(default=0.05, ge=0.01, le=0.5, description="SAM perturbation radius")
    
    @field_validator('checkpoint_dir', mode='before')
    @classmethod
    def validate_checkpoint_dir(cls, v):
        if v is not None:
            return Path(v)
        return v


class DataConfig(BaseModel):
    """Dataset configuration."""
    use_real: bool = Field(default=False, description="Use real SHARP data (True) or dummy data (False)")
    use_consolidated: bool = Field(default=False, description="Use consolidated per-HARP dataset for faster I/O")
    consolidated_dir: Optional[Path] = Field(default=None, description="Path to consolidated HARP bundles")
    windows_parquet: Optional[Path] = None
    frames_meta_parquet: Optional[Path] = None
    npz_root: Optional[Path] = None
    target_size: int = Field(default=128, ge=64, le=1024, description="Spatial resolution (pixels)")
    input_hours: int = Field(default=48, ge=6, le=120, description="Input time window (hours)")
    P_per_t: int = Field(default=1024, ge=256, le=8192, description="Points sampled per time slice")
    pil_top_pct: float = Field(default=0.15, ge=0.01, le=0.5, description="Top % of |∇Bz| for PIL mask")
    val_fraction: float = Field(default=0.15, ge=0.05, le=0.3, description="Validation set fraction")
    components: list[str] = Field(
        default_factory=lambda: ["Bz"],
        description="List of field components to load (e.g., ['Bz'] or ['Bx', 'By', 'Bz'])"
    )
    scalar_features: list[str] = Field(
        default_factory=lambda: ["r_value", "gwpil", "obs_coverage", "frame_count"],
        description="Scalar features to use (computed from data). Default: R-value, GWPIL, observation coverage, frame count"
    )
    
    # SOTA: Additional feature engineering
    # NOTE: use_previous_flare_activity is disabled by default because
    # ConsolidatedWindowsDataset does NOT compute these features yet.
    # Enable only if you implement the feature computation in the dataset.
    use_previous_flare_activity: bool = Field(
        default=False,  # FIXED: Was True but features weren't computed, causing dimension mismatch!
        description="Include previous flare activity (24h, 48h, 72h lookback) as features"
    )
    use_pil_evolution: bool = Field(
        default=True,
        description="Compute PIL evolution features (growth, motion, intensity)"
    )
    use_temporal_statistics: bool = Field(
        default=True,
        description="Compute temporal statistics of R-value, GWPIL (trend, acceleration)"
    )
    use_goes_xray: bool = Field(
        default=False,
        description="Include GOES X-ray flux history as auxiliary input (requires additional data)"
    )
    
    # Multi-scale features (memory-efficient)
    multi_scale_crops: bool = Field(
        default=False,
        description="Use multi-scale spatial crops (2x, 4x zoom on PIL region)"
    )
    multi_scale_memory_efficient: bool = Field(
        default=True,
        description="Process multi-scale crops sequentially to reduce memory"
    )
    
    @property
    def n_components(self) -> int:
        return len(self.components)
    
    @property
    def n_scalar_features(self) -> int:
        """Compute total number of scalar features including derived ones."""
        n = len(self.scalar_features)
        if self.use_previous_flare_activity:
            n += 6  # flare_24h, flare_48h, flare_72h (C, M, X class counts)
        if self.use_pil_evolution:
            n += 8  # PIL evolution features
        if self.use_temporal_statistics:
            n += 4  # R-value trend, GWPIL trend, etc.
        return n
    
    # Dummy data settings
    dummy_T: int = Field(default=8, ge=2, le=48, description="Dummy time steps")
    dummy_H: int = Field(default=64, ge=32, le=256, description="Dummy spatial height")
    dummy_W: int = Field(default=64, ge=32, le=256, description="Dummy spatial width")
    dummy_num_samples: int = Field(default=2048, ge=64, le=10000, description="Dummy dataset size")
    
    @model_validator(mode='after')
    def validate_real_data_paths(self) -> DataConfig:
        if self.use_real:
            if self.windows_parquet is None:
                raise ValueError("windows_parquet is required when use_real=True")
            if self.use_consolidated:
                if self.consolidated_dir is None:
                    raise ValueError("consolidated_dir is required when use_consolidated=True")
            else:
                if self.frames_meta_parquet is None:
                    raise ValueError("frames_meta_parquet is required when use_real=True and not using consolidated")
                if self.npz_root is None:
                    raise ValueError("npz_root is required when use_real=True and not using consolidated")
        return self
    
    @field_validator('windows_parquet', 'frames_meta_parquet', 'npz_root', 'consolidated_dir', mode='before')
    @classmethod
    def validate_paths(cls, v):
        if v is not None:
            return Path(v)
        return v


class PINNConfig(BaseModel):
    """Complete PINN training configuration."""
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cuda", pattern="^(cuda|cpu|mps|cuda:[0-9]+)$")
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    eta: EtaConfig = Field(default_factory=EtaConfig)
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)
    collocation: CollocationConfig = Field(default_factory=CollocationConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> PINNConfig:
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self.model_dump(mode='json', exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def model_dump_summary(self) -> dict:
        """Generate human-readable summary of key settings."""
        return {
            "model": f"{self.model.layers}x{self.model.hidden} (fourier={self.model.fourier.max_log2_freq})",
            "classifier": f"{len(self.classifier.horizons)} horizons, {self.classifier.loss_type}",
            "physics": f"enabled={self.physics.enable}, resistive={self.physics.resistive}",
            "training": f"{self.train.steps} steps @ lr={self.train.lr}, batch={self.train.batch_size}",
            "data": f"real={self.data.use_real}, size={self.data.target_size}x{self.data.target_size}, hours={self.data.input_hours}",
        }
