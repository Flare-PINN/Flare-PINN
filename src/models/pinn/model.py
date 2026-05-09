# src/models/pinn/model.py
"""
Unified PINN model for 2.5D magnetic field evolution and flare prediction.

Combines:
  - Coordinate-based neural field for (A_z, B_z, u_x, u_y)
  - Physics-informed weak-form induction equation
  - Classification head for multi-horizon flare prediction
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import PINNConfig
from .core import PINNBackbone, ClassifierHead, B_perp_from_Az
from .physics import WeakFormInduction2p5D, VectorInduction2p5D, VectorPhysicsResidualInfo
from .losses import focal_loss, bce_logits, l1_data, interp_schedule
from .collocation import clip_and_renorm_importance


@dataclass
class PINNOutput:
    """Complete forward pass output."""
    # Predictions
    logits: torch.Tensor              # [B, n_horizons] classifier logits
    probs: torch.Tensor               # [B, n_horizons] probabilities (sigmoid)
    
    # Fields at collocation points
    A_z: torch.Tensor                 # [N, 1] vector potential
    B_z: torch.Tensor                 # [N, 1] out-of-plane field
    u_x: torch.Tensor                 # [N, 1] flow x-component
    u_y: torch.Tensor                 # [N, 1] flow y-component
    B_x: Optional[torch.Tensor]       # [N, 1] in-plane field (if computed)
    B_y: Optional[torch.Tensor]       # [N, 1] in-plane field (if computed)
    
    # Packed vector fields (for vector mode)
    B: Optional[torch.Tensor] = None  # [N, 3] full B vector (Bx, By, Bz)
    u: Optional[torch.Tensor] = None  # [N, 2] velocity vector (ux, uy)
    
    # Losses
    loss_cls: torch.Tensor = None     # Classification loss
    loss_data: torch.Tensor = None    # Data fitting loss (L1 on Bz)
    loss_phys: torch.Tensor = None    # Physics residual loss
    loss_total: torch.Tensor = None   # Weighted total
    
    # Diagnostics
    ess: float = 0.0                  # Effective sample size (collocation quality)
    lambda_phys: float = 0.0          # Current physics weight
    fourier_alpha: float = 1.0        # Current Fourier frequency weight
    
    # Physics diagnostics (for vector mode)
    physics_info: Optional[VectorPhysicsResidualInfo] = None


class PINNModel(nn.Module):
    """
    Complete PINN system for solar flare prediction.
    
    Architecture:
      1. Coordinate MLP backbone (x,y,t) -> (A_z, B, u) where B can be scalar or vector
      2. Classifier head: pooled features -> multi-horizon logits
      3. Physics loss: weak-form 2.5D induction equation on collocation points
      
    Training stages (curriculum):
      P0: Data-driven only (λ_phys=0)
      P1-P4: Gradual physics ramp-up via lambda_phys_schedule
      
    Vector Mode (new):
      When cfg.model.vector_B=True, uses full 3-component B field physics:
        - ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η·Jz]
        - ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η·Jz]
        - ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
      Plus solenoidal constraint: ∂x·Bx + ∂y·By = 0
    """
    
    def __init__(self, cfg: PINNConfig):
        super().__init__()
        self.cfg = cfg
        
        # Check if vector B mode is enabled
        vector_B = getattr(cfg.model, 'vector_B', False)
        self.vector_B = vector_B
        
        # Core components
        self.backbone = PINNBackbone(
            hidden=cfg.model.hidden,
            layers=cfg.model.layers,
            max_log2_freq=cfg.model.fourier.max_log2_freq,
            learn_eta=cfg.model.learn_eta,
            vector_B=vector_B,
            hard_div_free=getattr(cfg.model, 'hard_div_free', False),
        )
        
        # Scalar features count from config (default: R-value, GWPIL, obs_coverage, frame_count)
        n_scalar_features = len(cfg.data.scalar_features) if cfg.data.scalar_features else 4
        
        # Read classifier config options
        use_attention = getattr(cfg.classifier, 'use_attention', True)
        use_physics_features = getattr(cfg.classifier, 'use_physics_features', True)
        
        # Determine field component count: 3 for vector_B mode, 1 otherwise
        n_field_components = 3 if vector_B else 1
        
        self.classifier = ClassifierHead(
            hidden=cfg.classifier.hidden,
            dropout=cfg.classifier.dropout,
            horizons=cfg.classifier.horizons,
            use_attention=use_attention,
            use_physics_features=use_physics_features,
            n_scalar_features=n_scalar_features,
            n_field_components=n_field_components,  # FIXED: Support multi-component fields
        )
        
        # Physics module: use VectorInduction2p5D for vector mode
        # Get physics config options
        
        if vector_B:
            self.physics = VectorInduction2p5D(
                physics_cfg=cfg.physics,
                eta_cfg=cfg.eta,
                n_fourier_modes=3,
                n_random_tests=2,
            )
        else:
            # Legacy scalar physics (backward compatible via alias)
            # WeakFormInduction2p5D extends VectorInduction2p5D with legacy return format
            self.physics = WeakFormInduction2p5D(
                physics_cfg=cfg.physics,
                eta_cfg=cfg.eta,
                n_fourier_modes=3,
                n_random_tests=2,
            )
        
        # Training state
        self.register_buffer("_train_frac", torch.tensor(0.0))  # progress ∈ [0,1]
    
    def set_train_frac(self, frac: float) -> None:
        """Update training progress (0=start, 1=end) for curriculum scheduling."""
        frac = float(max(0.0, min(1.0, frac)))
        self._train_frac.fill_(frac)
        
        # Update Fourier frequency annealing
        ramp = self.cfg.model.fourier.ramp_frac
        if ramp > 0:
            alpha = min(1.0, frac / ramp)
        else:
            alpha = 1.0
        self.backbone.set_fourier_alpha(alpha)
    
    def get_lambda_phys(self) -> float:
        """Get current physics loss weight from schedule."""
        if not self.cfg.physics.enable:
            return 0.0
        # FIXED: Avoid .item() which can hang on MPS
        frac = float(self._train_frac.detach())
        return interp_schedule(self.cfg.physics.lambda_phys_schedule, frac)
    
    def get_collocation_alpha(self) -> float:
        """Get current PIL bias weight (interpolates from alpha_start to alpha_end)."""
        # FIXED: Avoid .item() which can hang on MPS
        frac = float(self._train_frac.detach())
        a0 = self.cfg.collocation.alpha_start
        a1 = self.cfg.collocation.alpha_end
        return a0 + (a1 - a0) * frac
    
    def forward(
        self,
        coords: torch.Tensor,              # [T, P, 3] or [N, 3] collocation coords
        gt_bz: Optional[torch.Tensor] = None,     # [T, P, 1] ground truth Bz (optional)
        observed_mask: Optional[torch.Tensor] = None,  # [T] bool (which frames are observed)
        labels: Optional[torch.Tensor] = None,         # [B, n_horizons] flare labels
        pil_mask: Optional[np.ndarray] = None,         # [H, W] PIL mask for importance sampling
        scalars: Optional[torch.Tensor] = None,        # [B, n_scalars] scalar features (R-value, GWPIL, etc.)
        mode: str = "train"                # "train" or "eval"
    ) -> PINNOutput:
        """
        Forward pass with optional losses.
        
        Args:
            coords: Collocation points (already sampled for training)
                    - Training: [T, P, 3] structured by time
                    - Eval: [N, 3] arbitrary points
            gt_bz: Ground truth Bz at coords (for data loss)
            observed_mask: Which time steps have valid data
            labels: Classification targets for loss computation
            pil_mask: Polarity inversion line mask (for collocation diagnostics)
            mode: "train" (compute losses) or "eval" (inference only)
        
        Returns:
            PINNOutput with predictions, fields, and losses
        """
        device = coords.device
        
        # Flatten coords for backbone
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3).contiguous()
        coords_flat = coords_flat.requires_grad_(True)  # Ensure in-place returns self
        
        # Verify requires_grad is actually set (MPS can be quirky)
        assert coords_flat.requires_grad, "coords_flat must require grad for physics loss"
        
        # Infer batch size from shape or labels
        if len(original_shape) == 3:
            # [T, P, 3] - single sample, batch size = 1
            T, P = original_shape[0], original_shape[1]
            B = 1
        elif len(original_shape) == 4:
            # [B, T, P, 3] - batched
            B, T, P = original_shape[0], original_shape[1], original_shape[2]
        else:
            raise ValueError(f"Unexpected coords shape: {original_shape}. Expected [T,P,3] or [B,T,P,3]")
        
        # Forward through backbone
        out = self.backbone(coords_flat)
        A_z = out["A_z"]
        eta_raw = out.get("eta_raw")
        
        # Handle vector vs scalar mode
        if self.vector_B:
            # Vector mode: B and u are packed tensors
            B_vec = out["B"]         # [N, 3]
            u_vec = out["u"]         # [N, 2]
            B_x = out["B_x"]
            B_y = out["B_y"]
            B_z = out["B_z"]
            u_x = out["u_x"]
            u_y = out["u_y"]
        else:
            # Scalar mode (legacy)
            B_z = out["B_z"]
            u_x = out["u_x"]
            u_y = out["u_y"]
            B_vec = None
            u_vec = None
            B_x, B_y = None, None
            
            # Optionally compute in-plane field from vector potential
            if self.cfg.loss_weights.curl_consistency > 0:
                if A_z.requires_grad and coords_flat.requires_grad:
                    B_x, B_y = B_perp_from_Az(A_z, coords_flat)
        
        # Reshape for classifier: flatten -> [B, T, P, 1]
        feats_dict = {
            "A_z": A_z.reshape(B, T, P, 1),
            "B_z": B_z.reshape(B, T, P, 1),
            "u_x": u_x.reshape(B, T, P, 1),
            "u_y": u_y.reshape(B, T, P, 1),
        }
        
        # Add horizontal field components if available (vector mode)
        if B_x is not None and B_y is not None:
            feats_dict["B_x"] = B_x.reshape(B, T, P, 1)
            feats_dict["B_y"] = B_y.reshape(B, T, P, 1)
        
        # Classification head
        if observed_mask is None:
            observed_mask = torch.ones(T, dtype=torch.bool, device=device)
        
        # Prepare scalars for classifier (ensure batch dimension)
        if scalars is not None and scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)  # [n_scalars] -> [1, n_scalars]
        
        logits = self.classifier(feats_dict, observed_mask.unsqueeze(0), scalars=scalars)  # [B, n_horizons]
        probs = torch.sigmoid(logits)
        
        # Safety: clamp probs to valid range (prevents any floating point weirdness)
        probs = probs.clamp(0.0, 1.0)
        
        # ============ Compute Losses ============
        
        # Note: Fallback losses use (logits * 0).sum() to create zero tensors 
        # that ARE connected to the computation graph (not leaf tensors).
        # This ensures gradients can flow through even when some losses are disabled.
        
        # 1. Classification loss
        if mode == "train" and labels is not None:
            if self.cfg.classifier.loss_type == "focal":
                loss_cls = focal_loss(
                    logits, labels,  # Pass logits, not probs!
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma
                )
            else:  # bce
                loss_cls = bce_logits(logits, labels, pos_weight=self.cfg.classifier.pos_weight)
        else:
            # Zero loss but connected to computation graph
            loss_cls = (logits * 0).sum()
        
        # 2. Data fitting loss (L1 on observed Bz)
        if mode == "train" and gt_bz is not None:
            # Mask: only compute loss on observed frames
            # Fix broadcasting: observed_mask is [T], need [B, T, P, 1]
            if observed_mask.dim() == 1:
                mask_expanded = observed_mask[None, :, None, None].expand(B, T, P, 1)
            else:
                mask_expanded = observed_mask[:, :, None, None].expand(B, T, P, 1)
            
            # Ensure gt_bz has batch dimension
            if gt_bz.dim() == 3:
                gt_bz = gt_bz.unsqueeze(0)  # [T,P,1] -> [1,T,P,1]
            loss_data = l1_data(feats_dict["B_z"], gt_bz, mask_expanded)
        else:
            # Zero loss but connected to computation graph
            loss_data = (B_z * 0).sum()
        
        # 3. Physics loss (weak-form induction)
        loss_phys = (logits * 0).sum()  # Default: zero but graph-connected
        ess_val = 0.0
        lambda_phys = self.get_lambda_phys()
        phys_info = None  # Will be VectorPhysicsResidualInfo in vector mode
        
        if mode == "train" and lambda_phys > 1e-8:
            # Collocation: compute PIL-based importance weights
            alpha = self.get_collocation_alpha()
            N = coords_flat.shape[0]
            
            # Use PIL mask if provided, otherwise uniform sampling
            if pil_mask is not None:
                # PIL mask is [H, W] tensor (may be on GPU)
                # Ensure pil_mask is a tensor on the correct device
                if isinstance(pil_mask, np.ndarray):
                    pil_mask_tensor = torch.from_numpy(pil_mask).to(device).float()
                else:
                    pil_mask_tensor = pil_mask.to(device).float()
                
                H, W = pil_mask_tensor.shape
                
                # We already have coords sampled, so compute importance weights from PIL mask
                # Extract spatial coords (x,y) from coords_flat
                xy_coords = coords_flat[:, :2]  # [N, 2] in [-1, 1]
                
                # Convert to pixel coordinates (keep as tensors on device)
                x_px = ((xy_coords[:, 0] + 1.0) * 0.5 * (W - 1)).long().clamp(0, W-1)
                y_px = ((xy_coords[:, 1] + 1.0) * 0.5 * (H - 1)).long().clamp(0, H-1)
                
                # Get PIL mask values at sampled points (tensor indexing, no numpy)
                pil_values = pil_mask_tensor[y_px, x_px]
                
                # Compute mixture probability: p = alpha * p_pil + (1-alpha) * p_uniform
                p_uniform_xy = 0.25  # uniform over [-1,1]^2 has area 4
                p_uniform_t = 0.5    # uniform over [-1,1] has length 2
                p_uniform = p_uniform_xy * p_uniform_t  # = 0.125
                
                # For PIL regions: mask value indicates importance
                pil_sum = pil_mask_tensor.sum()
                pil_frac = float(pil_sum / (H * W)) if pil_sum > 0 else 1.0
                p_pil_xy = 1.0 / (4.0 * pil_frac) if pil_frac > 0 else p_uniform_xy
                p_pil = p_pil_xy * p_uniform_t
                
                # Mixture: p(x,y,t) = alpha * p_pil * I_pil(x,y) + (1-alpha) * p_uniform
                p_spatial = alpha * p_pil * pil_values + (1 - alpha) * p_uniform
                p_spatial = p_spatial.clamp_min(1e-8)  # Avoid division by zero
                
                imp_weights, _ = clip_and_renorm_importance(p_spatial.unsqueeze(-1), self.cfg.collocation.impw_clip_quantile)
            else:
                # Fallback: uniform weights
                p_uniform = torch.full((N, 1), 0.125, device=device)
                imp_weights, _ = clip_and_renorm_importance(p_uniform, self.cfg.collocation.impw_clip_quantile)
            
            # ESS (Effective Sample Size) - measure of sampling efficiency
            # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
            w_sq_sum = (imp_weights ** 2).sum()
            with torch.no_grad():
                w_sum_val = float(imp_weights.sum().detach().cpu())
                w_sq_sum_val = float(w_sq_sum.detach().cpu())
            ess_val = (w_sum_val ** 2) / (w_sq_sum_val + 1e-8)
            
            # Compute physics residual
            eta_mode = "field" if self.cfg.model.learn_eta else "scalar"
            loss_phys_raw, phys_info = self.physics(
                self.backbone,
                coords_flat,
                imp_weights,
                eta_mode=eta_mode,
                eta_scalar=self.cfg.model.eta_scalar
            )
            loss_phys = lambda_phys * loss_phys_raw
        
        # 4. Optional curl consistency loss (B_perp from Az vs direct prediction)
        loss_curl = (A_z * 0).sum()  # Default: zero but graph-connected
        if mode == "train" and self.cfg.loss_weights.curl_consistency > 0 and B_x is not None:
            # Compute B_perp from vector potential and compare with direct prediction
            # B = curl(A) => Bx = -dAz/dy, By = dAz/dx
            if A_z.requires_grad and coords_flat.requires_grad:
                B_x_curl, B_y_curl = B_perp_from_Az(A_z, coords_flat)
                loss_curl = (
                    (B_x - B_x_curl).abs().mean() + 
                    (B_y - B_y_curl).abs().mean()
                )
        
        # Total loss
        loss_total = (
            self.cfg.loss_weights.cls * loss_cls +
            self.cfg.loss_weights.data * loss_data +
            loss_phys +
            self.cfg.loss_weights.curl_consistency * loss_curl
        )
        
        # Safety check: ensure loss_total has a gradient connection to model params
        # This can fail if all loss components are zero-valued leaf tensors
        if not loss_total.requires_grad:
            import warnings
            warnings.warn(
                f"loss_total does not require grad! "
                f"loss_cls.requires_grad={loss_cls.requires_grad}, "
                f"loss_data.requires_grad={loss_data.requires_grad}, "
                f"loss_phys.requires_grad={loss_phys.requires_grad}, "
                f"loss_curl.requires_grad={loss_curl.requires_grad}"
            )
            # Reconnect by adding a zero-valued term that IS connected to the computation graph
            # Use logits since it's always computed and connected to model parameters
            loss_total = loss_total + (logits * 0).sum()
            # Double-check the reconnection worked
            if not loss_total.requires_grad:
                # Last resort: connect via a model parameter directly
                first_param = next(iter(self.parameters()))
                loss_total = loss_total + (first_param * 0.0).sum()
        
        # Convert phys_info to VectorPhysicsResidualInfo if it's the legacy type
        vector_phys_info = None
        if phys_info is not None and self.vector_B:
            # In vector mode, physics returns VectorPhysicsResidualInfo directly
            # via the parent class, but wrapped in legacy PhysicsResidualInfo
            # The actual VectorPhysicsResidualInfo is available in vector mode
            vector_phys_info = phys_info if isinstance(phys_info, VectorPhysicsResidualInfo) else None
        
        return PINNOutput(
            logits=logits,
            probs=probs,
            A_z=A_z,
            B_z=B_z,
            u_x=u_x,
            u_y=u_y,
            B_x=B_x,
            B_y=B_y,
            B=B_vec if self.vector_B else None,
            u=u_vec if self.vector_B else None,
            loss_cls=loss_cls,
            loss_data=loss_data,
            loss_phys=loss_phys,
            loss_total=loss_total,
            ess=ess_val,
            lambda_phys=lambda_phys,
            # FIXED: Avoid .item() which can hang on MPS
            fourier_alpha=float(self.backbone.ff._alpha.detach()),
            physics_info=vector_phys_info,
        )
    
    @torch.no_grad()
    def predict(
        self,
        coords: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Inference-only forward pass.
        
        Args:
            coords: [T, P, 3] evaluation coordinates
            observed_mask: [T] which frames are observed
        
        Returns:
            probs: [1, n_horizons] flare probabilities
            fields: dict of field values at coords
        """
        self.eval()
        out = self.forward(coords, observed_mask=observed_mask, mode="eval")
        fields = {
            "A_z": out.A_z,
            "B_z": out.B_z,
            "u_x": out.u_x,
            "u_y": out.u_y,
            "B_x": out.B_x,
            "B_y": out.B_y,
        }
        
        # Include packed vectors if in vector mode
        if self.vector_B:
            fields["B"] = out.B
            fields["u"] = out.u
        
        return out.probs, fields


