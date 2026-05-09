"""
Strong-form 2.5D resistive induction equation PINN baseline.

This implements the pointwise (strong-form) PDE residuals:
    r_Bx = ∂t Bx - ∂y Φ + η ∂y Jz
    r_By = ∂t By + ∂x Φ - η ∂x Jz  
    r_Bz = ∂t Bz + ∇⊥·(Bz u⊥) - η ∇²⊥ Bz

where:
    Φ = ux*By - uy*Bx (shear flux)
    Jz = ∂x By - ∂y Bx (vertical current density)

Unlike weak-form, this requires second derivatives which amplify noise.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class StrongFormConfig:
    """Configuration for strong-form physics."""
    eta: float = 0.01  # Magnetic diffusivity
    w_div: float = 1.0  # Divergence penalty weight
    use_huber: bool = True  # Use pseudo-Huber loss for stability
    huber_delta: float = 1.0  # Huber delta parameter
    chunk_size: int = 256  # Chunk size for gradient computation (memory)
    use_resistive: bool = True  # Include resistive terms
    edge_margin: int = 0  # Pixels to exclude from edges (for noisy boundaries)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of outputs w.r.t. inputs via autograd.
    
    Args:
        outputs: Tensor of shape (N,) or (N, 1)
        inputs: Tensor of shape (N, 3) with requires_grad=True
        
    Returns:
        Gradients of shape (N, 3) containing [∂/∂x, ∂/∂y, ∂/∂t]
    """
    if outputs.dim() == 2:
        outputs = outputs.squeeze(-1)
    
    # Check if outputs has grad_fn (i.e., is connected to computation graph)
    if outputs.grad_fn is None:
        return torch.zeros(inputs.shape[0], 3, device=inputs.device, dtype=inputs.dtype)
    
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,  # Handle case where outputs doesn't depend on all inputs
    )[0]
    
    # If grad is None (unused inputs), return zeros
    if grad is None:
        return torch.zeros(inputs.shape[0], 3, device=inputs.device, dtype=inputs.dtype)
    
    return grad  # (N, 3)


def pseudo_huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Pseudo-Huber loss: smooth L1-ish penalty for stability.
    
    Behaves like L2 near zero, L1 for large values.
    Prevents gradient explosion from outlier residuals.
    """
    return delta**2 * (torch.sqrt(1 + (x / delta)**2) - 1)


class StrongFormInduction2p5D(nn.Module):
    """
    Strong-form 2.5D resistive induction equation physics module.
    
    This is a baseline comparison for the weak-form approach.
    It computes pointwise PDE residuals requiring second derivatives.
    
    Args:
        config: StrongFormConfig with hyperparameters
        
    Note:
        Strong-form is typically more sensitive to noise and requires
        careful tuning of physics loss weight (λ_phys) to avoid NaNs.
    """
    
    def __init__(self, config: Optional[StrongFormConfig] = None):
        super().__init__()
        self.config = config or StrongFormConfig()
        
        # Track NaN occurrences for diagnostics
        self.register_buffer('nan_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_count', torch.tensor(0, dtype=torch.long))
    
    def compute_residuals(
        self,
        coords: torch.Tensor,
        Bx: torch.Tensor,
        By: torch.Tensor,
        Bz: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute strong-form PDE residuals at collocation points.
        
        Args:
            coords: (N, 3) tensor of (x, y, t) with requires_grad=True
            Bx, By, Bz: (N,) or (N, 1) magnetic field components
            ux, uy: (N,) or (N, 1) velocity components
            
        Returns:
            Dictionary with residual tensors:
                - r_Bx: Bx induction residual
                - r_By: By induction residual  
                - r_Bz: Bz induction residual
                - r_div: Divergence constraint residual
        """
        eta = self.config.eta
        
        # Ensure proper shapes
        if Bx.dim() == 2:
            Bx, By, Bz = Bx.squeeze(-1), By.squeeze(-1), Bz.squeeze(-1)
        if ux.dim() == 2:
            ux, uy = ux.squeeze(-1), uy.squeeze(-1)
        
        # First derivatives of B
        gBx = _grad(Bx, coords)
        dBx_dx, dBx_dy, dBx_dt = gBx[:, 0], gBx[:, 1], gBx[:, 2]
        
        gBy = _grad(By, coords)
        dBy_dx, dBy_dy, dBy_dt = gBy[:, 0], gBy[:, 1], gBy[:, 2]
        
        gBz = _grad(Bz, coords)
        dBz_dx, dBz_dy, dBz_dt = gBz[:, 0], gBz[:, 1], gBz[:, 2]
        
        # Current density: Jz = ∂By/∂x - ∂Bx/∂y
        Jz = dBy_dx - dBx_dy
        
        # Second derivatives for Jz (needed for resistive terms)
        if self.config.use_resistive:
            gJz = _grad(Jz, coords)
            dJz_dx, dJz_dy = gJz[:, 0], gJz[:, 1]
        else:
            dJz_dx = dJz_dy = torch.zeros_like(Jz)
        
        # Shear flux: Φ = ux*By - uy*Bx
        Phi = ux * By - uy * Bx
        gPhi = _grad(Phi, coords)
        dPhi_dx, dPhi_dy = gPhi[:, 0], gPhi[:, 1]
        
        # Advection terms for Bz: ∂(Bz*ux)/∂x + ∂(Bz*uy)/∂y
        Fx = Bz * ux
        Fy = Bz * uy
        gFx = _grad(Fx, coords)
        gFy = _grad(Fy, coords)
        dFx_dx, dFy_dy = gFx[:, 0], gFy[:, 1]
        
        # Diffusion for Bz: η(∂²Bz/∂x² + ∂²Bz/∂y²)
        if self.config.use_resistive:
            gdBz_dx = _grad(dBz_dx, coords)
            gdBz_dy = _grad(dBz_dy, coords)
            d2Bz_dx2 = gdBz_dx[:, 0]
            d2Bz_dy2 = gdBz_dy[:, 1]
        else:
            d2Bz_dx2 = d2Bz_dy2 = torch.zeros_like(Bz)
        
        # Residuals (constant η form)
        r_Bx = dBx_dt - dPhi_dy + eta * dJz_dy
        r_By = dBy_dt + dPhi_dx - eta * dJz_dx
        r_Bz = dBz_dt + dFx_dx + dFy_dy - eta * (d2Bz_dx2 + d2Bz_dy2)
        
        # Divergence constraint: ∇·B⊥ = 0
        r_div = dBx_dx + dBy_dy
        
        return {
            'r_Bx': r_Bx,
            'r_By': r_By,
            'r_Bz': r_Bz,
            'r_div': r_div,
            'Jz': Jz,  # For diagnostics
        }
    
    def compute_loss(
        self,
        residuals: dict[str, torch.Tensor],
        importance_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute physics loss from residuals.
        
        Args:
            residuals: Dictionary from compute_residuals()
            importance_weights: Optional (N,) weights for PIL-biased sampling
            
        Returns:
            loss: Scalar physics loss
            stats: Dictionary of component statistics
        """
        r_Bx = residuals['r_Bx']
        r_By = residuals['r_By']
        r_Bz = residuals['r_Bz']
        r_div = residuals['r_div']
        
        # Check for NaNs
        has_nan = (torch.isnan(r_Bx).any() or torch.isnan(r_By).any() or 
                   torch.isnan(r_Bz).any() or torch.isnan(r_div).any())
        
        if has_nan:
            self.nan_count += 1
            # Return safe loss to avoid breaking training
            return torch.tensor(0.0, device=r_Bx.device, requires_grad=True), {
                'r_Bx_median': float('nan'),
                'r_By_median': float('nan'),
                'r_Bz_median': float('nan'),
                'r_div_median': float('nan'),
                'has_nan': True,
            }
        
        self.total_count += 1
        
        # Apply importance weights if provided
        if importance_weights is not None:
            w = importance_weights / importance_weights.sum()
        else:
            w = torch.ones_like(r_Bx) / len(r_Bx)
        
        # Compute loss (pseudo-Huber or MSE)
        if self.config.use_huber:
            delta = self.config.huber_delta
            loss_Bx = (w * pseudo_huber(r_Bx, delta)).sum()
            loss_By = (w * pseudo_huber(r_By, delta)).sum()
            loss_Bz = (w * pseudo_huber(r_Bz, delta)).sum()
            loss_div = self.config.w_div * (w * pseudo_huber(r_div, delta)).sum()
        else:
            loss_Bx = (w * r_Bx**2).sum()
            loss_By = (w * r_By**2).sum()
            loss_Bz = (w * r_Bz**2).sum()
            loss_div = self.config.w_div * (w * r_div**2).sum()
        
        total_loss = loss_Bx + loss_By + loss_Bz + loss_div
        
        # Statistics for logging
        stats = {
            'r_Bx_median': r_Bx.abs().median().item(),
            'r_By_median': r_By.abs().median().item(),
            'r_Bz_median': r_Bz.abs().median().item(),
            'r_div_median': r_div.abs().median().item(),
            'r_Bx_p90': r_Bx.abs().quantile(0.9).item(),
            'r_By_p90': r_By.abs().quantile(0.9).item(),
            'r_Bz_p90': r_Bz.abs().quantile(0.9).item(),
            'loss_Bx': loss_Bx.item(),
            'loss_By': loss_By.item(),
            'loss_Bz': loss_Bz.item(),
            'loss_div': loss_div.item(),
            'has_nan': False,
        }
        
        return total_loss, stats
    
    def forward(
        self,
        model_fn: Callable,
        coords: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        domain_size: int = 128,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute strong-form physics loss.
        
        Args:
            model_fn: Function that takes coords and returns dict with B, u fields
            coords: (N, 3) collocation points (x, y, t) 
            importance_weights: Optional (N,) weights
            domain_size: Domain size in pixels (for edge masking)
            
        Returns:
            loss: Scalar physics loss
            stats: Dictionary of statistics
        """
        # Apply edge masking if configured
        if self.config.edge_margin > 0:
            interior_mask = self.create_edge_mask(coords, domain_size)
            coords = coords[interior_mask]
            if importance_weights is not None:
                importance_weights = importance_weights[interior_mask]
        
        # Ensure gradients are enabled for coords
        coords = coords.detach().clone().requires_grad_(True)
        
        # Query model for field values
        out = model_fn(coords)
        
        # Extract field components
        if 'B' in out:
            B = out['B']
            Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
        else:
            Bx = out['B_x'].squeeze(-1)
            By = out['B_y'].squeeze(-1)
            Bz = out['B_z'].squeeze(-1)
        
        
        if 'u' in out:
            u = out['u']
            ux, uy = u[..., 0], u[..., 1]
        else:
            ux = out['u_x'].squeeze(-1)
            uy = out['u_y'].squeeze(-1)
        
        # Compute residuals in chunks to save memory
        chunk_size = self.config.chunk_size
        N = coords.shape[0]
        
        if N <= chunk_size:
            residuals = self.compute_residuals(coords, Bx, By, Bz, ux, uy)
            return self.compute_loss(residuals, importance_weights)
        
        # Chunked computation
        all_residuals = {'r_Bx': [], 'r_By': [], 'r_Bz': [], 'r_div': []}
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk_coords = coords[i:end]
            chunk_Bx = Bx[i:end]
            chunk_By = By[i:end]
            chunk_Bz = Bz[i:end]
            chunk_ux = ux[i:end]
            chunk_uy = uy[i:end]
            
            chunk_residuals = self.compute_residuals(
                chunk_coords, chunk_Bx, chunk_By, chunk_Bz, chunk_ux, chunk_uy
            )
            
            for key in all_residuals:
                all_residuals[key].append(chunk_residuals[key])
        
        # Concatenate chunks
        residuals = {k: torch.cat(v, dim=0) for k, v in all_residuals.items()}
        
        return self.compute_loss(residuals, importance_weights)
    
    def get_nan_rate(self) -> float:
        """Return fraction of forward passes that produced NaNs."""
        if self.total_count == 0:
            return 0.0
        return (self.nan_count / self.total_count).item()
    
    def create_edge_mask(self, coords: torch.Tensor, domain_size: int = 128) -> torch.Tensor:
        """
        Create mask excluding edge points (noisy SHARP boundaries).
        
        Args:
            coords: (N, 3) normalized coords in [-1, 1]
            domain_size: Original domain size in pixels
            
        Returns:
            Boolean mask (N,) - True for interior points
        """
        margin = self.config.edge_margin
        if margin <= 0:
            return torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        
        # Convert margin pixels to normalized coords
        # domain_size pixels maps to [-1, 1], so each pixel = 2/domain_size
        margin_norm = margin * (2.0 / domain_size)
        
        x, y = coords[:, 0], coords[:, 1]
        interior = (
            (x > -1 + margin_norm) & (x < 1 - margin_norm) &
            (y > -1 + margin_norm) & (y < 1 - margin_norm)
        )
        
        return interior


# Convenience function matching weak-form interface
def compute_strong_form_phys_loss(
    model_fn: Callable,
    coords: torch.Tensor,
    importance_weights: Optional[torch.Tensor] = None,
    eta: float = 0.01,
    w_div: float = 1.0,
    use_huber: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Convenience function to compute strong-form physics loss.
    
    Args:
        model_fn: Function mapping coords -> {B, u} tensors
        coords: (N, 3) collocation points
        importance_weights: Optional (N,) PIL weights
        eta: Magnetic diffusivity
        w_div: Divergence penalty weight
        use_huber: Use pseudo-Huber for stability
        
    Returns:
        loss: Scalar physics loss
        stats: Component statistics dict
    """
    config = StrongFormConfig(eta=eta, w_div=w_div, use_huber=use_huber)
    module = StrongFormInduction2p5D(config)
    module = module.to(coords.device)
    
    return module(model_fn, coords, importance_weights)

