# src/models/pinn/hybrid_model.py
"""
Hybrid CNN-conditioned PINN model for solar flare prediction.

Combines CNN spatial features with physics-informed coordinate fields.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn

from .collocation import clip_and_renorm_importance
from .config import PINNConfig
from .core import ClassifierHead
from .hybrid_core import HybridPINNBackbone
from .latent_sampling import sample_latent_soft_bilinear
from .losses import (
    asymmetric_focal_loss,
    bce_logits,
    class_balanced_focal_loss,
    confidence_penalty,
    focal_loss,
    focal_loss_with_label_smoothing,
    interp_schedule,
    l1_data,
    poly_focal_loss,
)
from .physics import VectorInduction2p5D, WeakFormInduction2p5D
from baselines.strong_form import StrongFormInduction2p5D, StrongFormConfig


# =============================================================================
# Typed Output Models
# =============================================================================

@dataclass
class FieldOutputs:
    """Neural field outputs at collocation points."""
    A: torch.Tensor    # [N, C] potential
    B: torch.Tensor    # [N, C] magnetic field
    u: torch.Tensor    # [N, 2] velocity (ux, uy)
    eta_raw: Optional[torch.Tensor]
    
    @property
    def B_z(self) -> torch.Tensor:
        """Get Bz component (last component of B field)."""
        # For C=1: B is Bz directly
        # For C=3: B = [Bx, By, Bz], so Bz is index 2
        if self.B.shape[-1] == 1:
            return self.B
        return self.B[..., 2:3]
    
    @property
    def B_x(self) -> Optional[torch.Tensor]:
        """Get Bx component (first component, only valid for 3-component B)."""
        if self.B.shape[-1] >= 3:
            return self.B[..., 0:1]
        return None
    
    @property
    def B_y(self) -> Optional[torch.Tensor]:
        """Get By component (second component, only valid for 3-component B)."""
        if self.B.shape[-1] >= 3:
            return self.B[..., 1:2]
        return None
    
    @property
    def u_x(self) -> torch.Tensor:
        """Get ux velocity component."""
        return self.u[..., 0:1]
    
    @property
    def u_y(self) -> torch.Tensor:
        """Get uy velocity component."""
        return self.u[..., 1:2]

@dataclass
class HybridPINNOutput:
    """Complete output from hybrid PINN forward pass."""
    logits: torch.Tensor
    probs: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    u: torch.Tensor
    # Legacy compatibility fields
    A_z: Optional[torch.Tensor] = None
    B_z: Optional[torch.Tensor] = None
    u_x: Optional[torch.Tensor] = None
    u_y: Optional[torch.Tensor] = None
    B_x: Optional[torch.Tensor] = None
    B_y: Optional[torch.Tensor] = None
    # Loss components (Optional since they're computed only in train mode)
    loss_cls: Optional[torch.Tensor] = None
    loss_data: Optional[torch.Tensor] = None
    loss_phys: Optional[torch.Tensor] = None
    loss_curl: Optional[torch.Tensor] = None
    loss_total: Optional[torch.Tensor] = None
    # Diagnostic info
    ess: float = 0.0
    lambda_phys: float = 0.0
    fourier_alpha: float = 0.0


# =============================================================================
# Main Model Class
# =============================================================================

class HybridPINNModel(nn.Module):
    """
    Hybrid CNN-PINN for solar flare prediction.
    """
    
    def __init__(self, cfg: PINNConfig, encoder_in_channels: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        
        # Determine input channels for CNN
        if encoder_in_channels is None:
            # Derived from data config if available, else ModelConfig
            if cfg.model.in_channels is not None:
                in_channels = cfg.model.in_channels
            else:
                in_channels = cfg.data.n_components
        else:
            in_channels = encoder_in_channels
            
        self.n_components = cfg.data.n_components
        
        # Hybrid backbone (CNN encoder + conditioned MLP)
        self.backbone = HybridPINNBackbone(
            encoder_in_channels=in_channels,
            encoder_cfg=cfg.model.encoder,
            hidden=cfg.model.hidden,
            layers=cfg.model.layers,
            max_log2_freq=cfg.model.fourier.max_log2_freq,
            film_layers=(3, 6, 9),
            learn_eta=cfg.model.learn_eta,
            n_field_components=self.n_components
        )
        
        # Load RF importance weights if configured
        rf_weights = self._get_rf_weights(cfg)
        
        # Classifier head with optional RF guidance
        use_attention = getattr(cfg.classifier, 'use_attention', True)
        use_physics_features = getattr(cfg.classifier, 'use_physics_features', True)
        
        # Scalar features count from config
        # CRITICAL FIX: Use the computed n_scalar_features property which accounts for
        # base features + PIL evolution (8) + temporal statistics (4) = up to 16 features
        # The dataset (ConsolidatedWindowsDataset) produces 16 features when SOTA features enabled
        n_scalar_features = cfg.data.n_scalar_features  # Use computed property from DataConfig
        
        self.classifier = ClassifierHead(
            hidden=cfg.classifier.hidden,
            dropout=cfg.classifier.dropout,
            horizons=cfg.classifier.horizons,
            use_attention=use_attention,
            use_physics_features=use_physics_features,
            rf_importance_weights=rf_weights,
            n_scalar_features=n_scalar_features,
            n_field_components=self.n_components,  # FIXED: Pass component count for proper dimensioning
        )
        
        # Physics module - choose between weak-form (default) or strong-form (baseline)
        use_strong_form = getattr(cfg.physics, 'use_strong_form', False)
        
        if use_strong_form:
            # Strong-form baseline: pointwise PDE residuals with autograd 2nd derivatives
            # Use for baseline comparison only - typically unstable on noisy data
            strong_cfg = StrongFormConfig(
                eta=cfg.model.eta_scalar,
                w_div=getattr(cfg.physics, 'div_B_weight', 1.0),
                use_huber=getattr(cfg.physics, 'strong_form_use_huber', True),
                huber_delta=getattr(cfg.physics, 'strong_form_huber_delta', 1.0),
                chunk_size=getattr(cfg.physics, 'strong_form_chunk_size', 256),
                use_resistive=cfg.physics.resistive,
                edge_margin=getattr(cfg.physics, 'strong_form_edge_margin', 0),
            )
            self.physics = StrongFormInduction2p5D(strong_cfg)
            self._use_strong_form = True
        elif self.n_components > 1:
            # Vector physics for multi-component fields (weak-form)
            self.physics = VectorInduction2p5D(
                physics_cfg=cfg.physics,
                eta_cfg=cfg.eta,
                n_fourier_modes=3,
                n_random_tests=2,
            )
            self._use_strong_form = False
        else:
            # Scalar physics for single-component (Bz only)
            # WeakFormInduction2p5D is an alias for VectorInduction2p5D with legacy return type
            self.physics = WeakFormInduction2p5D(
                physics_cfg=cfg.physics,
                eta_cfg=cfg.eta,
                n_fourier_modes=3,
                n_random_tests=2,
            )
            self._use_strong_form = False
        
        # Training state
        self.register_buffer("_train_frac", torch.tensor(0.0))
        
        # Track class counts for class-balanced loss (updated during training)
        self.register_buffer("_n_negative", torch.tensor(1000.0))
        self.register_buffer("_n_positive", torch.tensor(50.0))
        self._samples_seen = 0
        
        # LRA (Learning Rate Annealing) state - Wang et al. 2021
        # Running average of gradient norms for automatic loss balancing
        self.register_buffer("_lra_grad_norm_data", torch.tensor(1.0))
        self.register_buffer("_lra_grad_norm_phys", torch.tensor(1.0))
        self.register_buffer("_lra_lambda", torch.tensor(1.0))
        self.register_buffer("_lra_step", torch.tensor(0))  #  FIX: Buffer for checkpoint persistence
        
        # GradNorm state - Chen et al. 2018 (Multi-task learning)
        # Automatically balances task weights by equalizing gradient scales
        n_tasks = 2  # Classification + Physics
        self.register_buffer("_gradnorm_weights", torch.ones(n_tasks))
        self.register_buffer("_gradnorm_init_losses", torch.ones(n_tasks))
        self.register_buffer("_gradnorm_init_set", torch.tensor(False))
        self.register_buffer("_gradnorm_step", torch.tensor(0))  #  FIX: Buffer for checkpoint persistence
        
        # Adaptive collocation state - McClenny & Braga-Neto 2020 (RAR)
        # Store residuals for resampling
        self.register_buffer("_residual_history", torch.zeros(4096))
        self._collocation_step = 0
    
    def _get_rf_weights(self, cfg: PINNConfig) -> Optional[torch.Tensor]:
        """Get RF importance weights from file or defaults."""
        if not cfg.classifier.use_rf_guidance:
            return None
            
        if cfg.classifier.rf_weights_path is not None:
            return self._load_rf_weights(cfg.classifier.rf_weights_path)
        
        # Default domain-knowledge weights based on solar flare literature
        # Updated for 18 physics features (matches PhysicsFeatureExtractor.N_FEATURES)
        # Feature order: bz_mean, bz_std, bz_max, polarity_balance, 
        #                u_mean, u_max, flux_transport,
        #                temporal_var, temporal_accel, evolution_rate_var, recent_evolution,
        #                az_std, kurtosis_proxy,
        #                bh_mean, btot_mean, shear_mean, free_energy_mean, current_helicity_proxy
        rf_weights = torch.tensor([
            # Base field statistics (4)
            1.0,   # bz_mean - moderate importance
            1.5,   # bz_std - high importance (variability)
            1.2,   # bz_max - high importance (peak field)
            2.0,   # polarity_balance - CRITICAL (PIL indicator)
            # Flow statistics (3)
            0.8,   # u_mean - moderate (flow)
            1.0,   # u_max - moderate (peak flow)
            1.8,   # flux_transport - high (PIL activity)
            # Temporal features (4) - CRITICAL for flare prediction
            1.5,   # temporal_var - high (field evolution)
            2.5,   # temporal_accel - VERY HIGH (acceleration precedes flares)
            2.0,   # evolution_rate_var - high (instability indicator)
            2.2,   # recent_evolution - VERY HIGH (latest changes most predictive)
            # Field complexity (2)
            0.7,   # az_std - lower (indirect)
            1.5,   # kurtosis_proxy - high (non-Gaussian = complex field)
            # Vector field features (5)
            1.2,   # bh_mean - high (horizontal field)
            1.5,   # btot_mean - high (total energy)
            2.0,   # shear_mean - CRITICAL (non-potentiality indicator)
            2.5,   # free_energy_mean - CRITICAL (flare energy reservoir)
            2.8,   # current_helicity_proxy - HIGHEST (twist/writhe = flare trigger)
        ], dtype=torch.float32)
        
        # Validate feature count matches PhysicsFeatureExtractor
        from .core import PhysicsFeatureExtractor
        assert len(rf_weights) == PhysicsFeatureExtractor.N_FEATURES, (
            f"RF weights count ({len(rf_weights)}) != PhysicsFeatureExtractor.N_FEATURES ({PhysicsFeatureExtractor.N_FEATURES})"
        )
        # Normalize to sum to n_features for scale preservation
        n_features = len(rf_weights)
        return rf_weights / rf_weights.sum() * n_features
    
    def _load_rf_weights(self, path: Path) -> Optional[torch.Tensor]:
        """Load RF importance weights from pickle file."""
        try:
            from .rf_guidance import RFImportances
            rf_imp = RFImportances.load(path)
            
            # Map to the 11 physics features used in ClassifierHead
            feature_mapping = [
                'bz_mean', 'bz_std', 'bz_max', 'polarity_balance',
                'mean_gradient', 'max_gradient', 'gwpil',
                'temporal_variation', 'spatial_entropy',
                'bz_mean', 'bz_mean'  # Fallback/Proxy for horizontal/total field
            ]
            
            weights = [rf_imp.get_weight(name) for name in feature_mapping]
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            # Normalize to preserve feature scale
            return weights_tensor / weights_tensor.sum() * len(weights)
        except Exception as e:
            warnings.warn(f"Failed to load RF weights from {path}: {e}. Using uniform weights.")
            return None
    
    # =========================================================================
    # Curriculum / Schedule Methods
    # =========================================================================
    
    def set_train_frac(self, frac: float) -> None:
        """Update training progress fraction [0, 1]."""
        frac = float(max(0.0, min(1.0, frac)))
        self._train_frac.fill_(frac)
        ramp = self.cfg.model.fourier.ramp_frac
        alpha = min(1.0, frac / ramp) if ramp > 0 else 1.0
        self.backbone.set_fourier_alpha(alpha)
    
    def set_mc_mode(self, enabled: bool) -> None:
        """Enable/disable Monte Carlo dropout for uncertainty estimation."""
        if hasattr(self.classifier, 'set_mc_mode'):
            self.classifier.set_mc_mode(enabled)
    
    def predict_with_uncertainty(
        self,
        coords: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with MC Dropout uncertainty estimation.
        
        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Uncertainty (standard deviation)
        """
        self.eval()
        self.set_mc_mode(True)
        
        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(
                    coords, frames=frames, observed_mask=observed_mask,
                    scalars=scalars, mode="eval"
                )
                all_probs.append(out.probs)
        
        self.set_mc_mode(False)
        
        all_probs = torch.stack(all_probs)  # [n_samples, B, horizons]
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        return mean_probs, std_probs
    
    def get_lambda_phys(self) -> float:
        """Get physics loss weight based on LRA or schedule."""
        if not self.cfg.physics.enable:
            return 0.0
        
        # If LRA is enabled, it controls lambda directly (no schedule needed)
        # But we use a warmup-aware max cap to prevent instability early on
        if getattr(self.cfg.physics, 'use_lra', False):
            frac = float(self._train_frac.detach())
            # Warmup: max lambda starts at 0.1 and grows to 1.0 over first 30% of training
            max_lambda = min(1.0, 0.1 + frac * 3.0)  # 0.1 at start, 1.0 at frac=0.3
            return float(self._lra_lambda.detach().clamp(0.01, max_lambda))
        
        # Without LRA, use the manual schedule
        frac = float(self._train_frac.detach())
        lam = interp_schedule(self.cfg.physics.lambda_phys_schedule, frac)
        return min(lam, 50.0)  # Safety cap
    
    def update_lra(
        self,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
    ) -> float:
        """
        Update LRA (Learning Rate Annealing) weights based on gradient norms.
        
        Wang et al. 2021: "Understanding and Mitigating Gradient Pathologies in PINNs"
        
        The key idea: Balance gradient magnitudes so both data and physics losses
        contribute equally to the optimization.
        
        λ_phys = mean(|∇L_data|) / mean(|∇L_phys|)
        
        Args:
            loss_data: Data/classification loss (scalar tensor)
            loss_phys: Physics loss (scalar tensor)
            
        Returns:
            Updated lambda_phys value
        """
        if not self.cfg.physics.enable or not getattr(self.cfg.physics, 'use_lra', False):
            return self.get_lambda_phys()
        
        self._lra_step.add_(1)  #  FIX: In-place increment for buffer
        update_freq = getattr(self.cfg.physics, 'lra_update_freq', 1)
        
        #  FIX: Safely get current lambda value to avoid MPS hangs
        def get_current_lambda() -> float:
            return float(self._lra_lambda.detach().cpu())
        
        # Only update every N steps to reduce compute overhead
        if int(self._lra_step.item()) % update_freq != 0:
            return get_current_lambda()
        
        # Skip if losses don't require grad
        if not loss_data.requires_grad or not loss_phys.requires_grad:
            return get_current_lambda()
        
        # Skip if losses are invalid
        with torch.no_grad():
            if not torch.isfinite(loss_data) or not torch.isfinite(loss_phys):
                return get_current_lambda()
            
            #  FIX: Use detach().cpu() to avoid MPS hangs
            loss_phys_val = float(loss_phys.detach().cpu())
            loss_data_val = float(loss_data.detach().cpu())
        
        #  FIX: Raised threshold for stability
        if loss_phys_val < 1e-6 or loss_data_val < 1e-8:
            return get_current_lambda()
        
        # Compute gradient norms for each loss
        # We need to compute gradients w.r.t. a shared parameter set
        try:
            # Get a representative set of parameters (MLP layers)
            params = [p for p in self.backbone.parameters() if p.requires_grad]
            if not params:
                return get_current_lambda()
            
            # Compute gradient of data loss
            grads_data = torch.autograd.grad(
                loss_data, params, 
                retain_graph=True, 
                allow_unused=True,
                create_graph=False  # No need for 2nd order here
            )
            
            #  FIX: Check if we got any valid gradients
            valid_grads_data = [g for g in grads_data if g is not None]
            if not valid_grads_data:
                return get_current_lambda()
            
            grad_norm_data = sum(
                g.abs().mean() for g in valid_grads_data
            ) / len(valid_grads_data)
            
            # Compute gradient of physics loss
            grads_phys = torch.autograd.grad(
                loss_phys, params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False
            )
            
            valid_grads_phys = [g for g in grads_phys if g is not None]
            if not valid_grads_phys:
                return get_current_lambda()
            
            grad_norm_phys = sum(
                g.abs().mean() for g in valid_grads_phys
            ) / len(valid_grads_phys)
            
            #  FIX: Check for NaN/Inf in gradient norms
            with torch.no_grad():
                if not torch.isfinite(grad_norm_data) or not torch.isfinite(grad_norm_phys):
                    return get_current_lambda()
                
                #  FIX: Get values safely
                gnorm_data_val = float(grad_norm_data.detach().cpu())
                gnorm_phys_val = float(grad_norm_phys.detach().cpu())
            
            #  FIX: Skip if gradient norms are too small
            if gnorm_data_val < 1e-8 or gnorm_phys_val < 1e-8:
                return get_current_lambda()
            
        except RuntimeError:
            # Gradient computation failed, keep current lambda
            return get_current_lambda()
        
        # Update running averages with exponential moving average (in no_grad)
        with torch.no_grad():
            alpha = getattr(self.cfg.physics, 'lra_alpha', 0.9)
            self._lra_grad_norm_data.mul_(alpha).add_(gnorm_data_val, alpha=1-alpha)
            self._lra_grad_norm_phys.mul_(alpha).add_(gnorm_phys_val, alpha=1-alpha)
            
            #  FIX: Get EMA values safely
            ema_data = float(self._lra_grad_norm_data.detach().cpu())
            ema_phys = float(self._lra_grad_norm_phys.detach().cpu())
            
            # Compute adaptive lambda: balance gradient magnitudes
            # λ_phys = |∇L_data| / |∇L_phys|
            # This makes physics gradients have same magnitude as data gradients
            ratio = ema_data / max(ema_phys, 1e-8)
            
            # Clamp to reasonable range (max 2.0 so physics doesn't dominate classification)
            new_lambda = max(0.01, min(2.0, ratio))
            
            #  FIX: Check for NaN before storing
            if not math.isfinite(new_lambda):
                return get_current_lambda()
            
            self._lra_lambda.fill_(new_lambda)
            
            return new_lambda
    
    def update_gradnorm(
        self,
        loss_cls: torch.Tensor,
        loss_phys: torch.Tensor,
        shared_params: list[nn.Parameter],
    ) -> tuple[float, float]:
        """
        Update GradNorm weights for automatic multi-task balancing.
        
        Chen et al. 2018: "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
        
        SIMPLIFIED VERSION: More stable for MPS and curriculum learning.
        
        Key insight: Instead of full GradNorm algorithm, use a simpler approach:
        - Compute gradient norms for each task
        - Adjust weights to equalize gradient magnitudes
        - Use heavy smoothing to prevent oscillations
        
        Args:
            loss_cls: Classification loss (scalar)
            loss_phys: Physics loss (scalar)
            shared_params: Shared parameters to compute gradients on (e.g., backbone)
            
        Returns:
            (weight_cls, weight_phys): Task weights
        """
        if not getattr(self.cfg.physics, 'use_gradnorm', False):
            return (1.0, self.get_lambda_phys())
        
        self._gradnorm_step.add_(1)
        step_val = int(self._gradnorm_step.item())
        update_freq = getattr(self.cfg.physics, 'gradnorm_update_freq', 10)
        
        # Helper to safely get current weights
        def get_current_weights() -> tuple[float, float]:
            w0 = float(self._gradnorm_weights[0].detach().cpu())
            w1 = float(self._gradnorm_weights[1].detach().cpu())
            # Safety: ensure weights are valid
            w0 = max(0.5, min(2.0, w0)) if math.isfinite(w0) else 1.0
            w1 = max(0.5, min(2.0, w1)) if math.isfinite(w1) else 1.0
            return (w0, w1)
        
        # Only update every N steps
        if step_val % update_freq != 0:
            return get_current_weights()
        
        # Validate losses
        if not loss_cls.requires_grad or not loss_phys.requires_grad:
            return get_current_weights()
        
        with torch.no_grad():
            if not torch.isfinite(loss_cls) or not torch.isfinite(loss_phys):
                return get_current_weights()
            loss_cls_val = float(loss_cls.detach().cpu())
            loss_phys_val = float(loss_phys.detach().cpu())
        
        # Skip if losses are invalid (allow VERY small physics loss for GradNorm)
        if loss_cls_val < 1e-8 or loss_cls_val > 100.0:
            return get_current_weights()
        # Physics loss can be extremely small (1e-11) at start - still compute gradients
        if loss_phys_val < 1e-15 or loss_phys_val > 100.0:
            return get_current_weights()
        
        # Initialize reference losses on first valid step
        if not self._gradnorm_init_set.item():
            with torch.no_grad():
                # Use current losses as reference (they'll be ~1.0 rate initially)
                self._gradnorm_init_losses[0] = max(loss_cls_val, 0.01)
                self._gradnorm_init_losses[1] = max(loss_phys_val, 0.01)
                self._gradnorm_init_set.fill_(True)
            return (1.0, 1.0)
        
        if not shared_params:
            return get_current_weights()
        
        # Compute gradient norms (simplified: use mean absolute gradient)
        try:
            # Classification gradients
            grads_cls = torch.autograd.grad(
                loss_cls, shared_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
            valid_cls = [g for g in grads_cls if g is not None and torch.isfinite(g).all()]
            if len(valid_cls) < len(shared_params) // 4:  # Need at least 25% valid grads
                return get_current_weights()
            gnorm_cls = sum(g.abs().mean() for g in valid_cls) / len(valid_cls)
            
            # Physics gradients
            grads_phys = torch.autograd.grad(
                loss_phys, shared_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
            valid_phys = [g for g in grads_phys if g is not None and torch.isfinite(g).all()]
            if len(valid_phys) < len(shared_params) // 4:
                return get_current_weights()
            gnorm_phys = sum(g.abs().mean() for g in valid_phys) / len(valid_phys)
            
            # Validate gradient norms
            with torch.no_grad():
                if not torch.isfinite(gnorm_cls) or not torch.isfinite(gnorm_phys):
                    return get_current_weights()
                gnorm_cls_val = float(gnorm_cls.detach().cpu())
                gnorm_phys_val = float(gnorm_phys.detach().cpu())
            
            # Allow very small gradient norms (physics can have tiny grads early on)
            if gnorm_cls_val < 1e-12 or gnorm_phys_val < 1e-12:
                return get_current_weights()
            
        except RuntimeError:
            return get_current_weights()
        
        # Simplified weight update
        with torch.no_grad():
            # Goal: equalize gradient magnitudes
            # If phys grads are 10x larger, reduce w_phys by ~10x
            ratio = gnorm_cls_val / max(gnorm_phys_val, 1e-8)
            
            # Clamp ratio to prevent extreme adjustments
            ratio = max(0.1, min(10.0, ratio))
            
            # Current weights
            w_cls = float(self._gradnorm_weights[0].detach().cpu())
            w_phys = float(self._gradnorm_weights[1].detach().cpu())
            
            # Target: w_phys should scale by ratio to equalize grads
            # But we use heavy smoothing to prevent oscillations
            lr = 0.02  # Very slow adaptation
            
            # Simple rule: adjust w_phys based on gradient ratio
            # If phys grads dominate (ratio < 1), reduce w_phys
            # If cls grads dominate (ratio > 1), increase w_phys
            target_w_phys = w_phys * ratio
            target_w_phys = max(0.5, min(2.0, target_w_phys))  # Tight clamp
            
            # Smooth update
            new_w_phys = w_phys * (1 - lr) + target_w_phys * lr
            new_w_phys = max(0.5, min(2.0, new_w_phys))
            
            # Keep cls weight at 1.0 (anchor)
            new_w_cls = 1.0
            
            # Validate
            if not math.isfinite(new_w_cls) or not math.isfinite(new_w_phys):
                return get_current_weights()
            
            # Store
            self._gradnorm_weights[0] = new_w_cls
            self._gradnorm_weights[1] = new_w_phys
            
            return (new_w_cls, new_w_phys)
    
    def adaptive_resample_collocation(
        self,
        coords: torch.Tensor,
        model_func: Callable[[torch.Tensor], dict],
        imp_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive collocation point resampling based on physics residual magnitude.
        
        McClenny & Braga-Neto 2020: "Self-Adaptive Physics-Informed Neural Networks"
        (RAR - Residual Adaptive Resampling)
        
        Key idea: Focus compute on regions where physics is violated most.
        
        Algorithm:
        1. Compute physics residual at each collocation point
        2. Keep top-k% highest residual points (hard regions)
        3. Resample remaining points uniformly (exploration)
        
        Args:
            coords: [N, 3] collocation points
            model_func: Function mapping coords -> field dict
            imp_weights: [N, 1] current importance weights
            
        Returns:
            (new_coords, new_imp_weights): Resampled points and weights
        """
        if not getattr(self.cfg.collocation, 'use_adaptive_resampling', False):
            return coords, imp_weights
        
        self._collocation_step += 1
        resample_freq = getattr(self.cfg.collocation, 'adaptive_resample_freq', 10)
        
        # Only resample every N steps
        if self._collocation_step % resample_freq != 0:
            return coords, imp_weights
        
        N = coords.shape[0]
        keep_ratio = getattr(self.cfg.collocation, 'adaptive_keep_ratio', 0.3)
        n_keep = int(keep_ratio * N)
        n_resample = N - n_keep
        
        # Safety check: need at least some points to keep and resample
        if n_keep < 10 or n_resample < 10:
            return coords, imp_weights
        
        # Evaluate model to get residuals
        # We need to enable gradients for residual computation
        with torch.enable_grad():
            # Ensure coords require grad
            if not coords.requires_grad:
                coords.requires_grad_(True)
            
            try:
                field_dict = model_func(coords)
            except RuntimeError:
                # Model forward failed, fall back to current weights
                return coords, imp_weights
            
            # Extract B and u for residual computation
            # Handle both scalar and vector modes
            if "B" in field_dict:
                B = field_dict["B"]
            else:
                B = field_dict.get("B_z")
                if B is None:
                    return coords, imp_weights
            
            if "u" in field_dict:
                u = field_dict["u"]
            else:
                u_x = field_dict.get("u_x")
                u_y = field_dict.get("u_y")
                if u_x is None or u_y is None:
                    return coords, imp_weights
                u = torch.cat([u_x, u_y], dim=-1)

            # Compute physics residual magnitude at each point
            # For weak-form induction: residual ≈ |∂B/∂t + ∇×(u×B) - η∇²B|
            # As a proxy, use the gradient magnitudes
            try:
                B_grads = torch.autograd.grad(
                    B.sum(), coords,
                    create_graph=False, retain_graph=True,
                    allow_unused=True
                )[0]
                
                u_grads = torch.autograd.grad(
                    u.sum(), coords,
                    create_graph=False, retain_graph=True,
                    allow_unused=True
                )[0]
                
                if B_grads is None or u_grads is None:
                    return coords, imp_weights
                
                # Check for NaN/Inf in gradients
                if torch.isnan(B_grads).any() or torch.isinf(B_grads).any():
                    B_grads = torch.nan_to_num(B_grads, nan=0.0, posinf=1.0, neginf=-1.0)
                if torch.isnan(u_grads).any() or torch.isinf(u_grads).any():
                    u_grads = torch.nan_to_num(u_grads, nan=0.0, posinf=1.0, neginf=-1.0)
                
                B_grad_mag = B_grads.norm(dim=-1)
                u_grad_mag = u_grads.norm(dim=-1)
                
                # Combined residual proxy
                residual_mag = B_grad_mag + u_grad_mag
                
                # Safety: handle NaN in residual magnitudes
                if torch.isnan(residual_mag).any():
                    residual_mag = torch.nan_to_num(residual_mag, nan=0.0)
                
            except RuntimeError:
                # Gradient computation failed, fall back to current weights
                return coords, imp_weights
            
            # Store for monitoring
            with torch.no_grad():
                if residual_mag.numel() <= self._residual_history.numel():
                    self._residual_history[:residual_mag.numel()] = residual_mag.detach()
            
            # Get top-k highest residual points
            _, top_indices = torch.topk(residual_mag, n_keep, largest=True)
            
            # Resample remaining points uniformly in [-1, 1]^3
            new_coords_resample = torch.rand(
                n_resample, 3,
                device=coords.device,
                dtype=coords.dtype
            ) * 2.0 - 1.0
            
            # Combine: keep high-residual points + new uniform samples
            #  CRITICAL FIX: NEVER detach coords_keep, even with mps_fast_physics!
            # Detaching breaks gradient flow from physics loss → backbone → gradnorm.
            # This was causing "NaN/Inf gradients in 68/72 params" because gradients
            # couldn't flow back to most backbone parameters.
            # 
            # The mps_fast_physics flag only affects create_graph (2nd-order grads),
            # NOT the 1st-order gradient flow needed for optimizer.step().
            coords_keep = coords[top_indices]  # ALWAYS preserve gradient connection!
            new_coords = torch.cat([coords_keep, new_coords_resample], dim=0)
            new_coords.requires_grad_(True)
            
            #  FIX: Create NEW importance weights for the resampled points
            # Kept points: use their original importance weights, scaled up (2x for hard regions)
            # Resampled points: uniform importance weight (1.0)
            with torch.no_grad():
                # Get importance weights for kept points
                kept_weights = imp_weights[top_indices] * 2.0  # 2x weight for hard regions
                
                # New points get uniform weight
                uniform_weights = torch.ones(n_resample, 1, device=imp_weights.device)
                
                # Combine weights
                new_imp_weights = torch.cat([kept_weights, uniform_weights], dim=0)
                
                # Renormalize so weights sum to N (mean = 1)
                new_imp_weights = new_imp_weights * (N / new_imp_weights.sum().clamp(min=1e-6))
            
        return new_coords, new_imp_weights
    
    def get_collocation_alpha(self) -> float:
        """Get PIL importance weight based on training progress."""
        # FIXED: Avoid .item() which can hang on MPS
        frac = float(self._train_frac.detach())
        a0 = self.cfg.collocation.alpha_start
        a1 = self.cfg.collocation.alpha_end
        return a0 + (a1 - a0) * frac
    
    # =========================================================================
    # Decomposed Forward Components
    # =========================================================================
    
    def encode_frames(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observed frames into latent representation.
        
        Args:
            frames: [T, H, W] - Bz frames for entire window
            observed_mask: [T] - Which frames are observed (t <= t0)
        
        Returns:
            L: [1, C, H, W] - Latent feature map
            g: [1, D] - Global code
        """
        return self.backbone.encode(frames, observed_mask)
    
    def query_field(
        self,
        coords: torch.Tensor,
        L: torch.Tensor,
        g: torch.Tensor
    ) -> FieldOutputs:
        """
        Query neural field at collocation points.
        """
        coords.requires_grad_(True)
        out = self.backbone(coords, L, g, use_nearest=False)
        
        #  SAFETY: Clamp outputs to match training behavior (forward_batched)
        # Prevents train/test skew where inference sees wild values > 10.0
        for k in ["A", "B", "u"]:
            if k in out:
                out[k] = out[k].clamp(-10.0, 10.0)
        
        return FieldOutputs(
            A=out["A"],
            B=out["B"],
            u=out["u"],
            eta_raw=out["eta_raw"]
        )
    
    def update_class_counts(self, labels: torch.Tensor) -> None:
        """Update running class counts for class-balanced loss.
        
        Note: For multi-horizon training, we use fixed per-horizon statistics
        from the dataset rather than dynamic counts, since dynamic counting
        across all horizons dilutes the per-horizon imbalance ratios.
        """
        with torch.no_grad():
            # Count positives and negatives across all horizons
            # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
            n_pos = float(labels.sum().detach().cpu())
            n_neg = labels.numel() - n_pos
            
            # Safety check for valid counts
            if not (0 <= n_pos <= labels.numel()) or not (0 <= n_neg <= labels.numel()):
                return  # Skip update if counts are invalid
            
            # Exponential moving average update
            momentum = 0.99 if self._samples_seen > 100 else 0.5
            self._n_positive.lerp_(torch.tensor(max(1.0, n_pos), device=labels.device), 1 - momentum)
            self._n_negative.lerp_(torch.tensor(max(1.0, n_neg), device=labels.device), 1 - momentum)
            
            # Clamp buffers to prevent overflow
            self._n_positive.clamp_(1.0, 1e6)
            self._n_negative.clamp_(1.0, 1e6)
            
            self._samples_seen += 1
    
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss based on config."""
        # Update class counts for class-balanced loss
        if self.training:
            self.update_class_counts(labels)
        
        # Get current class count estimates with safety checks
        # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
        n_neg = float(self._n_negative.detach().cpu())
        n_pos = float(self._n_positive.detach().cpu())
        
        # Clamp to valid range before converting to int
        n_neg = max(1.0, min(1e6, n_neg))
        n_pos = max(1.0, min(1e6, n_pos))
        
        samples_per_class = (int(n_neg), int(n_pos))
        
        label_smoothing = getattr(self.cfg.classifier, 'label_smoothing', 0.0)
        loss_type = self.cfg.classifier.loss_type
        
        if loss_type == "cb_focal":
            # SOTA: Class-balanced focal loss for severe imbalance
            # Use per-horizon class balancing for multi-horizon training
            cb_beta = getattr(self.cfg.classifier, 'cb_beta', 0.9999)
            
            # Fixed per-horizon positive rates from dataset statistics
            # These are more accurate than dynamic counting across all horizons
            horizon_pos_rates = {
                6: 0.012,   # 355/28780 = 1.2%
                12: 0.021,  # 615/28780 = 2.1%
                24: 0.035,  # 1011/28780 = 3.5%
                48: 0.052,  # ~1500/28780 = ~5.2% (estimated)
                72: 0.070,  # ~2000/28780 = ~7% (estimated)
            }
            
            horizons = self.cfg.classifier.horizons
            n_horizons = len(horizons)
            
            if n_horizons > 1 and logits.shape[-1] == n_horizons:
                # Per-horizon class-balanced focal loss with slight 24h emphasis
                losses = []
                # Equal weights: let model learn shared representations
                gentle_weights = {6: 1.0, 12: 1.0, 24: 1.0, 48: 1.0, 72: 1.0}
                weights = []
                
                for i, h in enumerate(horizons):
                    pos_rate = horizon_pos_rates.get(h, 0.03)  # default 3%
                    # Estimate samples: assume 28780 total samples
                    n_total = 28780
                    n_pos_h = max(1, int(pos_rate * n_total))
                    n_neg_h = n_total - n_pos_h
                    
                    loss_h = class_balanced_focal_loss(
                        logits[:, i:i+1], labels[:, i:i+1],
                        samples_per_class=(n_neg_h, n_pos_h),
                        beta=cb_beta,
                        gamma=self.cfg.classifier.focal_gamma
                    )
                    losses.append(loss_h)
                    weights.append(gentle_weights.get(h, 1.0))
                
                # Weighted mean with gentle 24h emphasis
                weights_t = torch.tensor(weights, device=logits.device)
                weights_t = weights_t / weights_t.sum() * len(horizons)  # normalize
                loss = (torch.stack(losses) * weights_t).mean()
            else:
                # Single horizon or fallback: use dynamic counts
                loss = class_balanced_focal_loss(
                    logits, labels,
                    samples_per_class=samples_per_class,
                    beta=cb_beta,
                    gamma=self.cfg.classifier.focal_gamma
                )
        elif loss_type == "poly_focal":
            # PolyLoss + Focal for better calibration
            poly_epsilon = getattr(self.cfg.classifier, 'poly_epsilon', 1.0)
            loss = poly_focal_loss(
                logits, labels,
                epsilon=poly_epsilon,
                alpha=self.cfg.classifier.focal_alpha,
                gamma=self.cfg.classifier.focal_gamma
            )
        elif loss_type == "asymmetric":
            gamma_neg = getattr(self.cfg.classifier, 'asymmetric_gamma_neg', 4.0)
            loss = asymmetric_focal_loss(
                logits, labels, gamma_pos=0.0, gamma_neg=gamma_neg, clip=0.05
            )
        elif loss_type == "focal":
            if label_smoothing > 0:
                loss = focal_loss_with_label_smoothing(
                    logits, labels,
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma,
                    smoothing=label_smoothing
                )
            else:
                loss = focal_loss(
                    logits, labels,
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma
                )
        else:  # bce
            loss = bce_logits(logits, labels, pos_weight=self.cfg.classifier.pos_weight)
        
        # Add confidence penalty for calibration
        conf_penalty_weight = getattr(self.cfg.classifier, 'confidence_penalty', 0.0)
        if conf_penalty_weight > 0:
            loss = loss + confidence_penalty(probs, conf_penalty_weight)
        
        #  SAFETY: Classification loss should be bounded and non-NaN
        # Clamp to prevent extreme values and handle NaN/Inf
        loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)
        loss = loss.clamp(min=0.0, max=100.0)
        
        return loss
    
    def compute_data_loss(
        self,
        B_field: torch.Tensor,
        gt_field: torch.Tensor,
        observed_mask: torch.Tensor,
        B: int, T: int, P: int
    ) -> torch.Tensor:
        """Compute data fitting loss (L1 on field components)."""
        if observed_mask.dim() == 1:
            mask_expanded = observed_mask[None, :, None, None].expand(B, T, P, 1)
        else:
            mask_expanded = observed_mask[:, :, None, None].expand(B, T, P, 1)
        
        # gt_field: [B, T, P, C] (assuming batch dim added in forward)
        # B_field: [B*T*P, C] -> reshape
        B_reshaped = B_field.reshape(B, T, P, -1)
        
        # Handle dimension mismatch between prediction and ground truth
        pred_components = B_reshaped.shape[-1]
        gt_components = gt_field.shape[-1]
        
        if pred_components != gt_components:
            # Match dimensions: prefer using Bz which is the last component
            if pred_components > gt_components:
                # Prediction has more components - slice to match GT
                # If GT is 1 component (Bz), take last component of prediction
                B_reshaped = B_reshaped[..., -gt_components:]
            else:
                # GT has more components - slice GT to match prediction
                gt_field = gt_field[..., -pred_components:]
             
        return l1_data(B_reshaped, gt_field, mask_expanded)

    def compute_curl_loss(
        self,
        A_field: torch.Tensor,
        coords_flat: torch.Tensor,
        gt_field: torch.Tensor,
        observed_mask: torch.Tensor,
        B: int, T: int, P: int
    ) -> torch.Tensor:
        """Compute curl consistency loss."""
        # Only applicable if we can map A -> B via curl.
        # For 2.5D with Az: Bx = -dAz/dy, By = dAz/dx
        # This requires Az to be 1 component and gt to have Bx/By.
        # If n_components=1 (Bz), we don't have Bx/By in gt (unless we loaded them separately?)
        # New dataset loads all 'components'. If components=['Bz'], we don't have Bx/By.
        # So this loss is only valid if we have Bx/By in 'components' OR if we hacked it.
        return torch.tensor(0.0, device=A_field.device)

    
    def compute_physics_loss(
        self,
        coords_flat: torch.Tensor,
        L: torch.Tensor,
        g: torch.Tensor,
        pil_mask: Optional[torch.Tensor],
        lambda_phys: float
    ) -> tuple[torch.Tensor, float]:
        """
        Compute physics loss with comprehensive stability measures.
        
        Supports both scalar (Bz only) and vector (Bx, By, Bz) physics.
        Uses physics_grad_scale from config to prevent physics loss
        from dominating the classification objective.
        """
        device = coords_flat.device
        N = coords_flat.shape[0]
        
        # Early return if too few points
        if N < 64:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        n_max = self.cfg.collocation.n_max
        
        #  FIX: Removed premature uniform subsampling here. 
        # We must compute importance weights for ALL points first, then subsample based on those weights.
        # Uniform sampling first destroys the ability to find rare "hard" regions (PIL).
        
        alpha = self.get_collocation_alpha()
        
        # Get physics gradient scale from config (default 0.1 for stability)
        physics_grad_scale = getattr(self.cfg.physics, 'physics_grad_scale', 0.1)
        
        # Compute importance weights
        if pil_mask is not None:
            xy_coords = coords_flat[:, :2]
            H_pil, W_pil = pil_mask.shape
            
            # Map coords to PIL mask indices
            x_px = ((xy_coords[:, 0] + 1.0) * 0.5 * (W_pil - 1)).long().clamp(0, W_pil - 1)
            y_px = ((xy_coords[:, 1] + 1.0) * 0.5 * (H_pil - 1)).long().clamp(0, H_pil - 1)
            
            # pil_mask is now a tensor - index directly
            pil_values = pil_mask[y_px, x_px].float()
            
            p_uniform = 0.125
            with torch.no_grad():
                pil_sum = float(pil_mask.sum().detach().cpu())
            pil_frac = pil_sum / (H_pil * W_pil) if pil_sum > 0 else 1.0
            p_pil = (1.0 / (4.0 * pil_frac) * 0.5) if pil_frac > 0 else p_uniform
            p_spatial = alpha * p_pil * pil_values + (1 - alpha) * p_uniform
            p_spatial = p_spatial.clamp(min=1e-6, max=10.0)
            imp_weights, _ = clip_and_renorm_importance(
                p_spatial.unsqueeze(-1), 
                self.cfg.collocation.impw_clip_quantile
            )
        else:
            imp_weights = torch.ones((N, 1), device=device)
        
        # OPTIMIZATION: Subsample points based on importance weights if N > n_max
        if N > n_max:
            # Use importance weights as probability for sampling
            # This implements "Importance Sampling" (selecting points) vs "Importance Weighting" (weighting loss)
            # Since we already calculated weights for 'weighting', we can use them for sampling.
            # If we sample proportional to weights, the resulting weights should be 1.0 (unbiased).
            # Or we can just WeightedRandomSample and keep weights=1.
            # BUT, standard PINN usually does weighting.
            # Hybrid approach: Sample n_max points with probability proportional to weights.
            # Then re-weighting is not needed (weights=1), OR we keep weights?
            #
            # Let's stick to "Subsample then Weight" for stability, or "Uniform Subsample then Weight".
            # But uniform subsample misses PIL points.
            #
            # Safest Approach: Weighted Random Sampling without replacement?
            # Or simpler: Multinomial sampling.
            
            with torch.no_grad():
                probs = imp_weights.squeeze(-1)
                # Ensure valid probs
                probs = torch.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=0.0).clamp(min=1e-6)
                indices = torch.multinomial(probs, n_max, replacement=False if N >= n_max else True)
            
            coords_flat = coords_flat[indices]
            
            # Re-compute weights for the subset? 
            # If we sampled proportional to p(x), the estimator is mean(L(x)). Weights cancel out.
            # imp_weights = 1/N * p(x) / q(x) where q(x) = p(x). So weight is const.
            # So we can set weights to 1.0 (or uniform mean).
            imp_weights = torch.ones((n_max, 1), device=device)
            
            # NOTE: We lose the specific numerical value of weights but preserve distribution.
            # This assumes imp_weights were density-based.
            
            # Update N for subsequent checks
            N = n_max
        
        # Apply causal weighting if enabled
        if getattr(self.cfg.physics, 'use_causal_training', False):
            t_coords = coords_flat[:, 2:3].clamp(-1.0, 1.0)
            t_normalized = (t_coords + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
            
            decay = getattr(self.cfg.physics, 'causal_decay', 2.0)
            decay_arg = (decay * t_normalized).clamp(max=5.0)  # Reduced from 10.0
            causal_weights = torch.exp(-decay_arg).clamp(min=0.01, max=1.0)
            
            # Handle NaN/Inf
            with torch.no_grad():
                if torch.isnan(causal_weights).any() or torch.isinf(causal_weights).any():
                    causal_weights = torch.ones_like(causal_weights)
            
            # Combine weights
            imp_weights = imp_weights * causal_weights
            
            # Normalize to mean=1 to preserve relative weights
            with torch.no_grad():
                mean_weight = imp_weights.mean()
            if mean_weight > 1e-6:
                imp_weights = imp_weights / mean_weight
        
        # Compute ESS for diagnostics
        with torch.no_grad():
            w_sum_val = float(imp_weights.sum().cpu())
            w_sq_sum_val = float((imp_weights ** 2).sum().cpu())
        ess = (w_sum_val ** 2 / max(w_sq_sum_val, 1e-8))
        
        # Physics residual
        eta_mode = "field" if self.cfg.model.learn_eta else "scalar"
        
        # Wrap backbone to output compatible format for Physics module
        def model_wrapper(c):
            # Simple clamp for stability
            c = c.clamp(-1.0, 1.0)
            
            out = self.backbone(c, L, g, use_nearest=False)
            
            # Clamp outputs for stability
            B = out["B"].clamp(-10.0, 10.0)
            u = out["u"].clamp(-5.0, 5.0)
            
            if self.n_components == 1:
                return {
                    "B_z": B,
                    "u_x": u[..., 0:1],
                    "u_y": u[..., 1:2],
                    "eta_raw": out["eta_raw"]
                }
            else:
                return {
                    "B": B,
                    "u": u,
                    "B_x": B[..., 0:1],
                    "B_y": B[..., 1:2],
                    "B_z": B[..., 2:3],
                    "u_x": u[..., 0:1],
                    "u_y": u[..., 1:2],
                    "eta_raw": out["eta_raw"]
                }
        
        # Apply adaptive resampling only if enabled and not too frequent
        use_rar = getattr(self.cfg.collocation, 'use_adaptive_resampling', False)
        if use_rar:
            coords_flat, imp_weights = self.adaptive_resample_collocation(
                coords_flat, model_wrapper, imp_weights
            )
        
        # Compute physics loss
        try:
            if getattr(self, '_use_strong_form', False):
                # Strong-form: simpler interface, no eta_mode
                loss_phys_raw, stats = self.physics(
                    model_wrapper,
                    coords_flat,
                    imp_weights.squeeze(-1) if imp_weights.dim() > 1 else imp_weights,
                )
            else:
                # Weak-form: standard interface
                loss_phys_raw, _ = self.physics(
                    model_wrapper,
                    coords_flat,
                    imp_weights,
                    eta_mode=eta_mode,
                    eta_scalar=self.cfg.model.eta_scalar
                )
        except RuntimeError as e:
            # If physics computation fails, return zero loss with gradient connection
            warnings.warn(f"Physics computation failed: {e}")
            return (coords_flat * 0).sum(), 0.0
        
        # Handle NaN/Inf while preserving gradient connection
        with torch.no_grad():
            is_finite = torch.isfinite(loss_phys_raw).all()
        if not is_finite:
            # Log when we hit NaN/Inf
            if getattr(self, '_use_strong_form', False):
                warnings.warn(f"Strong-form produced NaN/Inf loss - replacing with 0")
            # Create a connected zero
            loss_phys_raw = (coords_flat * 0).sum()
        else:
            # Clamp to reasonable range
            loss_phys_raw = loss_phys_raw.abs().clamp(max=20.0)
        
        # Scale physics loss
        # - physics_grad_scale: overall scaling factor (0.1 = 10% of raw)
        # - lambda_phys: curriculum/schedule weight (0→1 over training)
        scaled_loss = physics_grad_scale * lambda_phys * loss_phys_raw
        
        return scaled_loss, ess
    
    def _aggregate_losses(
        self,
        loss_cls: torch.Tensor,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
        loss_curl: torch.Tensor,
        lambda_phys: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Aggregate losses with comprehensive safety checks.
        
        Applies GradNorm weights when enabled for automatic task balancing.
        Ensures gradient connection is maintained even when losses are invalid.
        """
        # Helper to safely check if a loss is valid
        def is_valid_loss(loss: torch.Tensor) -> bool:
            if loss.numel() == 0:
                return False
            with torch.no_grad():
                val = float(loss.detach().cpu())
                return math.isfinite(val) and 0.0 <= val < 1000.0
        
        # Get GradNorm weights if enabled
        use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False) and self.cfg.physics.enable
        if use_gradnorm:
            w_cls = float(self._gradnorm_weights[0].detach().cpu())
            w_phys = float(self._gradnorm_weights[1].detach().cpu())
            # Safety clamp
            w_cls = max(0.5, min(2.0, w_cls)) if math.isfinite(w_cls) else 1.0
            w_phys = max(0.5, min(2.0, w_phys)) if math.isfinite(w_phys) else 1.0
        else:
            w_cls = self.cfg.loss_weights.cls
            w_phys = 1.0  # lambda_phys already applied to loss_phys
        
        # Initialize loss_total with a connected zero
        loss_total = None
        
        # Add classification loss (always first to ensure gradient connection)
        if is_valid_loss(loss_cls):
            loss_total = w_cls * loss_cls.clamp(max=100.0)
        
        # Add data fitting loss
        if is_valid_loss(loss_data):
            data_weight = self.cfg.loss_weights.data
            data_loss = data_weight * loss_data.clamp(max=100.0)
            loss_total = data_loss if loss_total is None else loss_total + data_loss
        
        # Add curl consistency loss
        if is_valid_loss(loss_curl) and self.cfg.loss_weights.curl_consistency > 0:
            curl_loss = self.cfg.loss_weights.curl_consistency * loss_curl.clamp(max=50.0)
            loss_total = curl_loss if loss_total is None else loss_total + curl_loss
        
        # Add physics loss (with GradNorm weight if enabled).
        # Threshold must match should_compute_phys in forward paths (avoid paying
        # strong-form / AD costs while physics is still off the objective).
        if is_valid_loss(loss_phys) and lambda_phys > 0.01:
            phys_loss = w_phys * loss_phys.clamp(max=50.0)
            loss_total = phys_loss if loss_total is None else loss_total + phys_loss
        
        # Fallback: create connected zero if no valid losses
        if loss_total is None:
            for param in self.parameters():
                if param.requires_grad:
                    loss_total = (param * 0.0).sum() + 0.01
                    break
            else:
                loss_total = torch.tensor(0.01, device=device, requires_grad=True)
        
        # Final safety check
        with torch.no_grad():
            if not torch.isfinite(loss_total):
                # Replace with connected small loss
                for param in self.parameters():
                    if param.requires_grad:
                        return (param * 0.0).sum() + 0.01
                return torch.tensor(0.01, device=device, requires_grad=True)
        
        return loss_total
    
    # =========================================================================
    # Main Forward Pass
    # =========================================================================
    
    def forward_batched(
        self,
        coords: torch.Tensor,
        frames: torch.Tensor,
        gt_bz: torch.Tensor,
        observed_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pil_mask: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> HybridPINNOutput:
        """
        OPTIMIZED batched forward pass - processes entire batch at once.
        
        Args:
            coords: [B, T, P, 3] collocation points
            frames: [B, T, C, H, W] input frames
            gt_bz: [B, T, P, n_comp] ground truth field
            observed_mask: [B, T] observation mask
            labels: [B, n_horizons] flare labels
            pil_mask: [B, H, W] PIL masks
            scalars: [B, n_scalars] scalar features
            mode: "train" or "eval"
        """
        device = coords.device
        B, T, P, _ = coords.shape
        
        # Flatten batch dimension for encoder - process all samples together
        # Encode each sample's frames (can't fully batch due to variable observed frames)
        # But we can still be smarter by avoiding Python loops where possible
        
        all_L = []
        all_g = []
        
        for b in range(B):
            L_b, g_b = self.encode_frames(frames[b], observed_mask[b])
            all_L.append(L_b)
            all_g.append(g_b)
        
        L = torch.cat(all_L, dim=0)  # [B, C_latent, H, W]
        g = torch.cat(all_g, dim=0)  # [B, D_global]
        
        # BATCHED field query - this is the main speedup
        # Flatten coords for batch MLP forward: [B, T, P, 3] -> [B*T*P, 3]
        coords_flat = coords.reshape(-1, 3).contiguous()
        coords_flat.requires_grad_(True)
        
        # Expand L and g for each point in batch
        # L: [B, C, H, W] -> need to sample for each of B*T*P points
        # But points from sample b should use L[b]
        
        # Create batch indices for efficient sampling
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, P).reshape(-1)
        
        # Sample L at all coords in one go (optimized)
        xy_all = coords_flat[:, :2]  # [B*T*P, 2]
        xy_batched = xy_all.unsqueeze(1)  # [B*T*P, 1, 2]
        
        # Batch sampling using grid_sample - need to handle per-sample L
        L_sampled_list = []
        chunk_size = T * P
        for b in range(B):
            start = b * chunk_size
            end = start + chunk_size
            L_b_sampled = sample_latent_soft_bilinear(
                L[b:b+1], xy_all[start:end].unsqueeze(0)
            ).squeeze(0)  # [T*P, C]
            L_sampled_list.append(L_b_sampled)
        L_sampled = torch.cat(L_sampled_list, dim=0)  # [B*T*P, C]
        
        # Expand g for all points
        g_expanded = g[batch_idx]  # [B*T*P, D]
        
        # Fourier features
        ff = self.backbone.ff(coords_flat)  # [B*T*P, FF_dim]
        
        # Simple clamp for stability
        L_sampled = L_sampled.clamp(-10.0, 10.0)
        g_expanded = g_expanded.clamp(-10.0, 10.0)
        
        # Concatenate inputs for MLP
        z = torch.cat([coords_flat, ff, L_sampled, g_expanded], dim=-1)
        
        # Forward through MLP layers with FiLM (now batched!)
        h = z
        for i, (layer, act) in enumerate(zip(self.backbone.layers_list, self.backbone.activations), start=1):
            h = act(layer(h))
            if i in self.backbone.film_layers:
                h = self.backbone.film_modules[str(i)](h, g_expanded)
        
        out = self.backbone.head(h)
        
        #  SAFETY: Tight clamp on outputs to prevent NaN in physics gradients
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = out.clamp(-10.0, 10.0)
        
        # Split outputs
        C = self.n_components
        A = out[..., :C]
        B_field = out[..., C:2*C]
        u = out[..., 2*C:2*C+2]
        
        # Reshape for classifier: [B*T*P, C] -> [B, T, P, C]
        feats_dict = {
            "A_z": A.reshape(B, T, P, -1),
            "B_z": B_field.reshape(B, T, P, -1),
            "u_x": u[..., 0:1].reshape(B, T, P, 1),
            "u_y": u[..., 1:2].reshape(B, T, P, 1),
        }
        
        if C >= 3:
            feats_dict["B_x"] = B_field[..., 0:1].reshape(B, T, P, 1)
            feats_dict["B_y"] = B_field[..., 1:2].reshape(B, T, P, 1)
            feats_dict["B_z_scalar"] = B_field[..., 2:3].reshape(B, T, P, 1)
        
        # BATCHED classification
        logits = self.classifier(feats_dict, observed_mask, scalars=scalars)  # [B, n_horizons]
        probs = torch.sigmoid(logits).clamp(0.0, 1.0)
        
        # Compute losses (batched)
        loss_cls = torch.tensor(0.0, device=device)
        loss_data = torch.tensor(0.0, device=device)
        loss_phys = torch.tensor(0.0, device=device)
        ess_val = 0.0
        
        # Get initial lambda (from schedule or LRA state)
        lambda_phys = self.get_lambda_phys()
        use_lra = getattr(self.cfg.physics, 'use_lra', False) and self.cfg.physics.enable
        
        if mode == "train":
            if labels is not None:
                loss_cls = self.compute_classification_loss(logits, labels, probs)
            
            if gt_bz is not None:
                loss_data = self.compute_data_loss(
                    B_field, gt_bz, observed_mask, B, T, P
                )
            
            # Physics loss - compute on multiple samples for better gradient estimation
            # Only when physics is on the training objective (same threshold as _aggregate_losses).
            use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False) and self.cfg.physics.enable
            should_compute_phys = (lambda_phys > 0.01) and observed_mask.any()
            if should_compute_phys:
                #  FIX: Average physics loss over multiple samples (not just first)
                # This reduces variance and makes GradNorm more stable
                n_phys_samples = min(B, 2)  # Use up to 2 samples for physics
                phys_losses = []
                for phys_i in range(n_phys_samples):
                    pil_mask_i = pil_mask[phys_i] if pil_mask is not None else None
                    coords_i = coords[phys_i].reshape(-1, 3).contiguous()
                    coords_i.requires_grad_(True)
                    #  FIX: Pass actual lambda_phys (not 1.0) so the schedule is applied
                    loss_phys_i, ess_i = self.compute_physics_loss(
                        coords_i, L[phys_i:phys_i+1], g[phys_i:phys_i+1], pil_mask_i, lambda_phys
                    )
                    phys_losses.append(loss_phys_i)
                    if phys_i == 0:
                        ess_val = ess_i
                
                # Average physics losses
                if len(phys_losses) > 1:
                    loss_phys = sum(phys_losses) / len(phys_losses)
                else:
                    loss_phys = phys_losses[0]
            
            # LRA: Update adaptive lambda based on gradient norms
            if use_lra and loss_cls.abs() > 1e-10 and loss_phys.abs() > 1e-10:
                # Combine data losses for LRA (cls + data)
                loss_data_combined = loss_cls + loss_data
                lambda_phys = self.update_lra(loss_data_combined, loss_phys)
            
            #  FIXED: Update GradNorm weights in batched path too
            use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False) and self.cfg.physics.enable
            if use_gradnorm and loss_cls.requires_grad and loss_phys.requires_grad:
                shared_params = [p for p in self.backbone.parameters() if p.requires_grad]
                w_cls, w_phys = self.update_gradnorm(loss_cls, loss_phys, shared_params)
        
        # Aggregate losses (with updated lambda if LRA, or GradNorm weights if enabled)
        loss_total = self._aggregate_losses(
            loss_cls, loss_data, loss_phys, 
            torch.tensor(0.0, device=device), 
            lambda_phys, device
        )
        
        #  FIX: Explicit memory cleanup for MPS to prevent leaks
        del all_L, all_g, L_sampled_list, z, h
        
        return HybridPINNOutput(
            logits=logits,
            probs=probs,
            A=A,
            B=B_field,
            u=u,
            loss_cls=loss_cls,
            loss_data=loss_data,
            loss_phys=loss_phys,
            loss_curl=torch.tensor(0.0, device=device),
            loss_total=loss_total,
            ess=ess_val,
            lambda_phys=lambda_phys,
            fourier_alpha=float(self.backbone.ff._alpha.detach().cpu())
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
        gt_bz: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pil_mask: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> HybridPINNOutput:
        """
        Forward pass.
        """
        device = coords.device
        T, P = coords.shape[0], coords.shape[1]
        B = 1
        
        # Simple input clamp (inputs from data don't need gradient preservation)
        coords = coords.clamp(-1.0, 1.0)
        if frames is not None:
            frames = frames.clamp(-10.0, 10.0)
        if gt_bz is not None:
            gt_bz = gt_bz.clamp(-10.0, 10.0)
        if scalars is not None:
            scalars = scalars.clamp(-100.0, 100.0)
        
        # 1. Encode frames
        has_observed = observed_mask is not None and observed_mask.any()
        
        if frames is not None and observed_mask is not None:
            L, g = self.encode_frames(frames, observed_mask)
        else:
            H_spatial = 128
            latent_ch = self.backbone.effective_latent_channels
            L = torch.zeros(1, latent_ch, H_spatial, H_spatial, device=device)
            g = torch.zeros(1, 64, device=device)
            has_observed = False
        
        # 2. Query field
        coords_flat = coords.reshape(-1, 3).contiguous()
        field = self.query_field(coords_flat, L, g)
        
        # 3. Classification
        if observed_mask is None:
            observed_mask = torch.ones(T, dtype=torch.bool, device=device)
        
        # Construct feats_dict for ClassifierHead (which expects specific keys)
        # We map our generic fields to what ClassifierHead expects
        # For multi-component (3), A_z and B_z contain full field tensors [B,T,P,C]
        # For single-component (1), they contain scalar fields [B,T,P,1]
        feats_dict = {
            "A_z": field.A.reshape(B, T, P, -1),  # [B,T,P,C] where C=n_components
            "B_z": field.B.reshape(B, T, P, -1),  # [B,T,P,C] where C=n_components
            "u_x": field.u.reshape(B, T, P, -1)[..., 0:1],
            "u_y": field.u.reshape(B, T, P, -1)[..., 1:2],
        }
        # For physics features, we need individual B components (if 3-component mode)
        if self.n_components >= 3 and field.B_x is not None and field.B_y is not None:
            feats_dict["B_x"] = field.B_x.reshape(B, T, P, 1)
            feats_dict["B_y"] = field.B_y.reshape(B, T, P, 1)
            # Also provide scalar Bz for PhysicsFeatureExtractor compatibility
            feats_dict["B_z_scalar"] = field.B_z.reshape(B, T, P, 1)
        
        # Prepare scalars for classifier (ensure batch dimension and correct size)
        if scalars is not None:
            if scalars.dim() == 1:
                scalars = scalars.unsqueeze(0)  # [n_scalars] -> [1, n_scalars]
            
            # SAFETY: Pad/truncate scalars to match classifier expectation
            expected_dim = self.classifier.n_scalar_features
            actual_dim = scalars.shape[-1]
            if actual_dim < expected_dim:
                pad = torch.zeros(scalars.shape[0], expected_dim - actual_dim, 
                                  device=scalars.device, dtype=scalars.dtype)
                scalars = torch.cat([scalars, pad], dim=-1)
            elif actual_dim > expected_dim:
                scalars = scalars[..., :expected_dim]
        
        logits = self.classifier(feats_dict, observed_mask.unsqueeze(0), scalars=scalars)
        probs = torch.sigmoid(logits)
        
        # Safety: clamp probs to valid range (prevents any floating point weirdness)
        probs = probs.clamp(0.0, 1.0)
        
        # 4. Compute losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_data = torch.tensor(0.0, device=device)
        loss_phys = torch.tensor(0.0, device=device)
        loss_curl = torch.tensor(0.0, device=device)
        ess_val = 0.0
        lambda_phys = self.get_lambda_phys()
        
        if mode == "train":
            # Classification loss
            if labels is not None:
                loss_cls = self.compute_classification_loss(logits, labels, probs)
            
            # Data fitting loss
            if gt_bz is not None:
                # gt_bz is now [T, P, C] - rename argument ideally but keeping for signature compat
                if gt_bz.dim() == 3:
                    gt_target = gt_bz.unsqueeze(0) # [1, T, P, C]
                else:
                    gt_target = gt_bz
                    
                loss_data = self.compute_data_loss(
                    field.B, gt_target, observed_mask, B, T, P
                )
                
                if self.cfg.loss_weights.curl_consistency > 0:
                    loss_curl = self.compute_curl_loss(
                        field.A, coords_flat, gt_target, observed_mask, B, T, P
                    )
            
            # Physics loss
            # FIXED: Use any() for NaN/Inf checks to avoid MPS synchronization issues
            with torch.no_grad():
                has_invalid_B = torch.isnan(field.B).any() or torch.isinf(field.B).any()
                has_invalid_u = torch.isnan(field.u).any() or torch.isinf(field.u).any()
            has_valid_outputs = not (has_invalid_B or has_invalid_u)
            
            # Same curriculum gate as _aggregate_losses / forward_batched.
            use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False) and self.cfg.physics.enable
            should_compute_phys = (lambda_phys > 0.01) and has_observed and has_valid_outputs
            
            if should_compute_phys:
                loss_phys, ess_val = self.compute_physics_loss(
                    coords_flat, L, g, pil_mask, lambda_phys
                )
                
                #  FIXED: Update GradNorm weights (if enabled)
                # These weights are stored in buffers and applied in _aggregate_losses()
                if use_gradnorm and loss_cls.requires_grad and loss_phys.requires_grad:
                    shared_params = [p for p in self.backbone.parameters() if p.requires_grad]
                    w_cls, w_phys = self.update_gradnorm(loss_cls, loss_phys, shared_params)
                    # Weights are stored in self._gradnorm_weights and used in _aggregate_losses()
        
        # 5. Aggregate losses
        loss_total = self._aggregate_losses(
            loss_cls, loss_data, loss_phys, loss_curl, lambda_phys, device
        )
        
        return HybridPINNOutput(
            logits=logits,
            probs=probs,
            A=field.A,
            B=field.B,
            u=field.u,
            # Fill legacy fields for return type compatibility if needed
            A_z=field.A, # placeholder
            B_z=field.B, # placeholder
            u_x=field.u[..., 0:1],
            u_y=field.u[..., 1:2],
            B_x=None,
            B_y=None,
            loss_cls=loss_cls,
            loss_data=loss_data,
            loss_phys=loss_phys,
            loss_curl=loss_curl,
            loss_total=loss_total,
            ess=ess_val,
            lambda_phys=lambda_phys,
            # FIXED: Avoid .item() which can hang on MPS after physics computation
            fourier_alpha=float(self.backbone.ff._alpha.detach().cpu())
        )
