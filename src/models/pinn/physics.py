# src/models/pinn/physics.py
"""
Physics-Informed Loss Module for 2.5D MHD Induction Equation.

Implements weak-form residuals for the FULL VECTOR induction equation
with multi-scale test functions for robust physics enforcement.

Mathematical Foundation (2.5D Vector Induction):
=================================================
The resistive MHD induction equation:
    ∂B/∂t = ∇×(u×B) - ∇×(η∇×B)

Under 2.5D assumptions:
- All quantities depend on (x, y, t) only: ∂_z(·) = 0
- B = (Bx, By, Bz) - all three components exist
- u = (ux, uy, 0) - in-plane velocity only

The component equations become:
    ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η(∂x·By - ∂y·Bx)]
    ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η(∂x·By - ∂y·Bx)]
    ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η∇⊥Bz)

Solenoidal constraint (div B = 0):
    ∂x·Bx + ∂y·By = 0 (since ∂z·Bz = 0)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from .config import EtaConfig, PhysicsConfig


class VectorPhysicsResidualInfo(BaseModel):
    """Diagnostic information from vector physics residual computation."""
    model_config = ConfigDict(frozen=True)
    
    # Component-wise induction residuals
    loss_induction_Bx: float
    loss_induction_By: float
    loss_induction_Bz: float
    loss_induction_total: float
    
    # Solenoidal constraint (div B = 0)
    loss_divergence_B: float
    
    # Velocity divergence (incompressibility)
    loss_divergence_u: float
    
    # Regularization
    loss_tv_eta: float
    
    # Summary statistics
    residual_Bx_mean: float
    residual_By_mean: float
    residual_Bz_mean: float
    residual_Bx_max: float
    residual_By_max: float
    residual_Bz_max: float


# Legacy dataclass for backward compatibility
@dataclass
class PhysicsResidualInfo:
    """Legacy diagnostic information (for backward compatibility)."""
    loss_induction: float
    loss_divergence: float
    loss_tv_eta: float
    residual_mean: float
    residual_max: float


class MultiScaleTestFunction(nn.Module):
    """
    Test functions for weak-form PDE residuals on Ω=[-1,1]².
    
    Combines Fourier sine basis + polynomial basis:
    
    1. Fourier modes (high-frequency, k=1,...,4):
       - ψ^(x)_k = sin(kπx)       [vanishes at x=±1]
       - ψ^(y)_k = sin(kπy)       [vanishes at y=±1]
       - ψ^(xy)_k = sin(kπx)sin(kπy)  [vanishes on all ∂Ω]
    
    2. Polynomial modes (low-frequency):
       - (1-x²), x(1-x²): vanish at x=±1
       - (1-y²), y(1-y²): vanish at y=±1
       - (1-x²)(1-y²): vanishes on all ∂Ω
    
    Component-specific filtering for mathematically correct weak forms:
    - Bx equation (∂Bx/∂t = ∂y[...]) → test functions vanishing at y=±1
      Valid: sin(kπy), sin(kπx)sin(kπy), (1-y²), y(1-y²), (1-x²)(1-y²)
      Count: 4 + 4 + 1 + 1 + 1 = 11 test functions
    
    - By equation (∂By/∂t = -∂x[...]) → test functions vanishing at x=±1
      Valid: sin(kπx), sin(kπx)sin(kπy), (1-x²), x(1-x²), (1-x²)(1-y²)
      Count: 4 + 4 + 1 + 1 + 1 = 11 test functions
    
    - Bz equation (∂Bz/∂t = -∇⊥·[...]) → test functions vanishing on all ∂Ω
      Valid: sin(kπx)sin(kπy), (1-x²)(1-y²)
      Count: 4 + 1 = 5 test functions
    
    Boundary handling:
    - We omit boundary flux integrals in the weak form (justified: SHARP patches
      have artificial edges with noisy data; boundary fluxes are not physically meaningful).
    - For component-matched test functions, the omitted boundary term is identically
      zero → exact weak form.
    - This filtering ensures mathematical correctness without "auxiliary regularizers".
    """
    def __init__(self, n_fourier_modes: int = 4, n_random: int = 4):  # n_random kept for backward compat
        super().__init__()
        self.n_fourier = n_fourier_modes
        # n_random is unused - kept for checkpoint loading compatibility
        
    def forward(
        self, coords: torch.Tensor, component: str = "all"
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate test functions and their spatial gradients at collocation points.
        
        Args:
            coords: [N, 3] coordinates (x, y, t) in [-1, 1]^3
            component: Which component to get test functions for:
                      "Bx", "By", "Bz", or "all" (returns all test functions)
            
        Returns:
            phis: List of test function values [N, 1]
            dphi_dx: List of x-derivatives [N, 1]
            dphi_dy: List of y-derivatives [N, 1]
        """
        x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
        device = coords.device
        
        # Generate ALL test functions first, then filter by component
        all_test_functions = []
        
        # Fourier test functions
        for k in range(1, self.n_fourier + 1):
            freq = math.pi * k
            
            # sin(k*π*x) modes (vanish at x = ±1) → valid for By
            phi_sin_x = torch.sin(freq * x)
            dphi_sin_x = freq * torch.cos(freq * x)
            all_test_functions.append(("sin_x", phi_sin_x, dphi_sin_x, torch.zeros_like(y)))
            
            # sin(k*π*y) modes (vanish at y = ±1) → valid for Bx
            phi_sin_y = torch.sin(freq * y)
            dphi_sin_y = freq * torch.cos(freq * y)
            all_test_functions.append(("sin_y", phi_sin_y, torch.zeros_like(x), dphi_sin_y))
            
            # Cross terms: sin(k*π*x)*sin(k*π*y) (vanish on all boundaries) → valid for ALL
            phi_cross = torch.sin(freq * x) * torch.sin(freq * y)
            dphi_cross_x = freq * torch.cos(freq * x) * torch.sin(freq * y)
            dphi_cross_y = freq * torch.sin(freq * x) * torch.cos(freq * y)
            all_test_functions.append(("cross", phi_cross, dphi_cross_x, dphi_cross_y))
        
        # Polynomial test functions
        # φ = (1-x²): vanishes at x = ±1 → valid for By
        phi_poly_x = 1 - x**2
        dphi_poly_x_dx = -2 * x
        all_test_functions.append(("poly_x", phi_poly_x, dphi_poly_x_dx, torch.zeros_like(y)))
        
        # φ = (1-y²): vanishes at y = ±1 → valid for Bx
        phi_poly_y = 1 - y**2
        dphi_poly_y_dy = -2 * y
        all_test_functions.append(("poly_y", phi_poly_y, torch.zeros_like(x), dphi_poly_y_dy))
        
        # φ = (1-x²)(1-y²): vanishes on all boundaries → valid for ALL
        phi_poly_cross = (1 - x**2) * (1 - y**2)
        dphi_poly_cross_dx = -2 * x * (1 - y**2)
        dphi_poly_cross_dy = -2 * y * (1 - x**2)
        all_test_functions.append(("poly_cross", phi_poly_cross, dphi_poly_cross_dx, dphi_poly_cross_dy))
        
        # φ = x(1-x²): vanishes at x = ±1 → valid for By
        phi_poly_linear_x = x * (1 - x**2)
        dphi_poly_linear_x_dx = (1 - x**2) - 2 * x**2  # = 1 - 3x²
        all_test_functions.append(("poly_linear_x", phi_poly_linear_x, dphi_poly_linear_x_dx, torch.zeros_like(y)))
        
        # φ = y(1-y²): vanishes at y = ±1 → valid for Bx
        phi_poly_linear_y = y * (1 - y**2)
        dphi_poly_linear_y_dy = (1 - y**2) - 2 * y**2  # = 1 - 3y²
        all_test_functions.append(("poly_linear_y", phi_poly_linear_y, torch.zeros_like(x), dphi_poly_linear_y_dy))
        
        # Filter based on component
        # Bx equation uses ∂y (so test functions must vanish at y=±1)
        # By equation uses ∂x (so test functions must vanish at x=±1)
        # Bz equation uses full ∇⊥ (so test functions must vanish on all ∂Ω)
        
        if component == "Bx":
            # Valid: sin_y, cross, poly_y, poly_cross, poly_linear_y
            valid_types = {"sin_y", "cross", "poly_y", "poly_cross", "poly_linear_y"}
        elif component == "By":
            # Valid: sin_x, cross, poly_x, poly_cross, poly_linear_x
            valid_types = {"sin_x", "cross", "poly_x", "poly_cross", "poly_linear_x"}
        elif component == "Bz":
            # Valid: cross, poly_cross (only functions that vanish on ALL boundaries)
            valid_types = {"cross", "poly_cross"}
        else:  # component == "all"
            # Return all test functions (backward compatibility)
            valid_types = {"sin_x", "sin_y", "cross", "poly_x", "poly_y", "poly_cross", 
                          "poly_linear_x", "poly_linear_y"}
        
        # Extract filtered test functions
        phis = []
        dphi_dx_list = []
        dphi_dy_list = []
        
        for func_type, phi, dphi_dx, dphi_dy in all_test_functions:
            if func_type in valid_types:
                phis.append(phi)
                dphi_dx_list.append(dphi_dx)
                dphi_dy_list.append(dphi_dy)
        
        return phis, dphi_dx_list, dphi_dy_list


class VectorInduction2p5D(nn.Module):
    """
    Weak-form of 2.5D MHD Vector Induction Equation.
    
    Physics equations (component form under ∂_z = 0):
    ================================================
    
    For Bx:
        ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η·Jz]
        where Jz = ∂x·By - ∂y·Bx (z-component of current density)
        
    For By:
        ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η·Jz]
        
    For Bz:
        ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
        
    Weak form (multiply by test φ and integrate by parts):
    ======================================================
    
    Bx equation:
        ∫ φ·∂Bx/∂t dx = -∫ (∂yφ)·(ux·By - uy·Bx) dx + ∫ (∂yφ)·η·Jz dx
        
    By equation:
        ∫ φ·∂By/∂t dx = ∫ (∂xφ)·(ux·By - uy·Bx) dx - ∫ (∂xφ)·η·Jz dx
        
    Bz equation (same as scalar):
        ∫ φ·∂Bz/∂t dx = ∫ ∇⊥φ·(Bz·u) dx - ∫ ∇⊥φ·(η·∇⊥Bz) dx
        
    Solenoidal constraint:
        ∇·B = ∂x·Bx + ∂y·By = 0 (soft penalty)
    
    This formulation:
        1. Requires only first derivatives (not second)
        2. Naturally handles discontinuities
        3. Works well with importance-weighted Monte Carlo integration
        4. Enforces physics for ALL three components of B
    """
    
    def __init__(
        self, 
        physics_cfg: PhysicsConfig,
        eta_cfg: EtaConfig,
        n_fourier_modes: int = 3,
        n_random_tests: int = 2,
        residual_normalization: str = "adaptive",
    ):
        """
        Args:
            physics_cfg: Physics configuration object
            eta_cfg: Resistivity configuration object
            n_fourier_modes: Number of Fourier modes in test functions
            n_random_tests: Number of random linear test functions
            residual_normalization: "fixed", "adaptive", or "per_scale"
        """
        super().__init__()
        self.eta_min = eta_cfg.min
        self.eta_max = eta_cfg.max
        self.tv_eta = eta_cfg.tv_weight
        
        self.use_resistive = physics_cfg.resistive
        self.include_boundary = physics_cfg.boundary_terms
        
        # Vector physics options
        self.enforce_div_free_u = physics_cfg.enforce_div_free_u
        self.enforce_div_free_B = True
        self.div_B_weight = physics_cfg.div_B_weight
        self.component_weights = physics_cfg.component_weights
        
        self.normalization = residual_normalization
        
        # Advanced options
        #  FIX: Disable uncertainty weighting if GradNorm is active (redundant mechanisms)
        use_gradnorm = getattr(physics_cfg, 'use_gradnorm', False)
        self.use_uncertainty_weighting = physics_cfg.use_uncertainty_weighting and not use_gradnorm
        self.enforce_force_free = physics_cfg.enforce_force_free
        self.force_free_weight = physics_cfg.force_free_weight
        
        self.use_causal_weighting = physics_cfg.use_causal_weighting
        self.causal_tol = physics_cfg.causal_tol
        
        self.enable_gradient_clamping = physics_cfg.enable_gradient_clamping
        self.gradient_clamp_value = physics_cfg.gradient_clamp_value
        
        # MPS optimization: disable create_graph for ~10x speedup
        self.mps_fast_physics = getattr(physics_cfg, 'mps_fast_physics', True)
        
        self.test_fn = MultiScaleTestFunction(n_fourier_modes, n_random_tests)
        
        # Learnable scaling factors for each loss component
        self.register_buffer("_loss_scale", torch.tensor(1.0))
        
        # Running estimates for normalization (more stable than per-batch percentiles)
        self.register_buffer("_B_scale_ema", torch.tensor(1.0))
        self.register_buffer("_u_scale_ema", torch.tensor(0.1))
        self._ema_momentum = 0.99
        
        # SOTA: Learned uncertainty weights for automatic loss balancing (Kendall et al. 2018)
        # log(σ²) for each component - initialized to balance roughly equal losses
        if self.use_uncertainty_weighting:
            self.log_var_Bx = nn.Parameter(torch.tensor(0.0))
            self.log_var_By = nn.Parameter(torch.tensor(0.0))
            self.log_var_Bz = nn.Parameter(torch.tensor(0.0))
            self.log_var_div = nn.Parameter(torch.tensor(-1.0))  # Higher precision for div constraint
        
        # Per-component EMA for adaptive scaling (more granular than combined)
        self.register_buffer("_Bx_scale_ema", torch.tensor(1.0))
        self.register_buffer("_By_scale_ema", torch.tensor(1.0))
        self.register_buffer("_Bz_scale_ema", torch.tensor(1.0))
    
    @staticmethod
    def _get_output_field(out: object, name: str) -> Optional[torch.Tensor]:
        """
        Safely fetch a field from mixed object/dict model outputs.
        
        Handles:
            - attribute-style access (dataclass or SimpleNamespace outputs)
            - dict-style access (legacy code paths)
        """
        if hasattr(out, name):
            value = getattr(out, name)
            if value is not None:
                return value
        if isinstance(out, dict):
            return out.get(name)
        return None
    
    def _compute_spatial_gradients(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        field_name: str = "field"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spatial and temporal gradients of a field.
        
        Args:
            field: [N, 1] field values
            coords: [N, 3] coordinates with requires_grad=True
            field_name: Name for error messages
            
        Returns:
            df_dx, df_dy, df_dt: [N, 1] gradient components
        """
        if not coords.requires_grad:
            zeros = torch.zeros_like(field)
            return zeros, zeros, zeros
        
        with torch.no_grad():
            has_invalid = torch.isnan(field).any() or torch.isinf(field).any()
        if has_invalid:
            field = torch.nan_to_num(field, nan=0.0, posinf=1.0, neginf=-1.0)
        
        ones = torch.ones_like(field)
        is_mps = field.device.type == "mps"
        # CRITICAL FIX: Physics training REQUIRES create_graph=True to compute 
        # d(Loss)/d(Params) via d(Residual)/d(Params).
        # The residual depends on spatial derivatives (dB/dx, etc).
        # d(Residual)/d(Params) requires differentiating through these spatial derivatives.
        # This needs double backward (create_graph=True).
        # Without this, the physics loss has NO gradient w.r.t. model parameters!
        create_graph = self.training
        
        try:
            grads = torch.autograd.grad(
                field, coords,
                grad_outputs=ones,
                create_graph=create_graph,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )[0]
        except RuntimeError as e:
            warnings.warn(f"Gradient computation failed for {field_name}: {e}")
            zeros = torch.zeros_like(field)
            return zeros, zeros, zeros
        
        if grads is None:
            zeros = torch.zeros_like(field)
            return zeros, zeros, zeros
        
        with torch.no_grad():
            has_invalid_grads = torch.isnan(grads).any() or torch.isinf(grads).any()
        if has_invalid_grads:
            #  SAFETY: Replace NaN/Inf with zero gradients
            grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.enable_gradient_clamping:
            scale = self.gradient_clamp_value
            #  SAFETY: Use hard clamp instead of tanh for simpler 2nd derivatives
            grads = grads.clamp(-scale, scale)
        
        df_dx = grads[..., 0:1]
        df_dy = grads[..., 1:2]
        df_dt = grads[..., 2:3]
        
        return df_dx, df_dy, df_dt
    
    def _compute_spatial_gradients_batched(
        self,
        fields: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        OPTIMIZED: Compute gradients for multiple fields at once.
        
        Args:
            fields: [N, C] stacked field values (e.g., Bx, By, Bz stacked)
            coords: [N, 3] coordinates with requires_grad=True
            
        Returns:
            df_dx, df_dy, df_dt: [N, C] gradient components for all fields
        """
        if not coords.requires_grad:
            zeros = torch.zeros_like(fields)
            return zeros, zeros, zeros
        
        with torch.no_grad():
            has_invalid = torch.isnan(fields).any() or torch.isinf(fields).any()
        if has_invalid:
            fields = torch.nan_to_num(fields, nan=0.0, posinf=1.0, neginf=-1.0)
        
        N, C = fields.shape
        is_mps = fields.device.type == "mps"
        # CRITICAL FIX: Physics training REQUIRES create_graph=True
        create_graph = self.training
        
        # Compute gradients for all channels at once by summing
        # This works because grad(sum(f_i), x) = sum(grad(f_i, x))
        # We'll compute per-channel gradients by doing C gradient calls
        # but with better memory access patterns
        
        all_dx = []
        all_dy = []
        all_dt = []
        
        for c in range(C):
            field_c = fields[:, c:c+1]
            ones = torch.ones_like(field_c)
            
            try:
                grads = torch.autograd.grad(
                    field_c, coords,
                    grad_outputs=ones,
                    create_graph=create_graph,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                grads = torch.zeros(N, 3, device=fields.device)
            
            if grads is None:
                grads = torch.zeros(N, 3, device=fields.device)
            
            with torch.no_grad():
                if torch.isnan(grads).any() or torch.isinf(grads).any():
                    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.enable_gradient_clamping:
                scale = self.gradient_clamp_value
                #  SAFETY: Use hard clamp instead of tanh for simpler 2nd derivatives
                grads = grads.clamp(-scale, scale)
            
            all_dx.append(grads[:, 0:1])
            all_dy.append(grads[:, 1:2])
            all_dt.append(grads[:, 2:3])
        
        df_dx = torch.cat(all_dx, dim=1)  # [N, C]
        df_dy = torch.cat(all_dy, dim=1)  # [N, C]
        df_dt = torch.cat(all_dt, dim=1)  # [N, C]
        
        return df_dx, df_dy, df_dt
    
    def _compute_residual_Bz(
        self,
        Bz: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        eta: torch.Tensor,
        dBz_dx: torch.Tensor,
        dBz_dy: torch.Tensor,
        dBz_dt: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for Bz component.
        
        Equation: ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
        
        Weak form:
            ∫ φ·∂Bz/∂t dx = ∫ ∇⊥φ·(Bz·u) dx - ∫ ∇⊥φ·(η·∇⊥Bz) dx
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂Bz/∂t
            term_time = phi * dBz_dt
            
            # Transport term (after integration by parts): ∫ ∇⊥φ·(Bz·u)
            # = ∫ (∂φ/∂x·Bz·ux + ∂φ/∂y·Bz·uy)
            term_transport = dphi_dx * (Bz * ux) + dphi_dy * (Bz * uy)
            
            # Resistive diffusion term: -∫ ∇⊥φ·(η·∇⊥Bz)
            # = -∫ (∂φ/∂x·η·∂Bz/∂x + ∂φ/∂y·η·∂Bz/∂y)
            if self.use_resistive:
                term_resistive = -(dphi_dx * (eta * dBz_dx) + dphi_dy * (eta * dBz_dy))
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂Bz/∂t - ∫∇φ·(Bz·u) + ∫∇φ·(η∇Bz) = 0
            # term_resistive already has negative sign from the IBP
            # So: term_time - term_transport - term_resistive = 0
            residual = term_time - term_transport - term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _compute_residual_Bx(
        self,
        dBx_dt: torch.Tensor,
        F_conv: torch.Tensor,
        Jz: torch.Tensor,
        eta: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for Bx component.
        
        Equation: ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η·Jz]
        where Jz = ∂x·By - ∂y·Bx, F_conv = ux·By - uy·Bx
        
        Weak form (after integration by parts):
            ∫φ·∂Bx/∂t + ∫(∂yφ)·F_conv - ∫(∂yφ)·η·Jz = 0
        
        Args:
            dBx_dt: Time derivative of Bx
            F_conv: Pre-computed convective flux (ux·By - uy·Bx)
            Jz: Pre-computed current density z-component
            eta: Resistivity field
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂Bx/∂t
            term_time = phi * dBx_dt
            
            # Advection term (after IBP): ∫(∂yφ)·F_conv
            term_advection = dphi_dy * F_conv
            
            # Resistive diffusion term (after IBP): ∫(∂yφ)·η·Jz
            if self.use_resistive:
                term_resistive = dphi_dy * (eta * Jz)
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂Bx/∂t + ∫(∂yφ)·F_conv - ∫(∂yφ)·η·Jz = 0
            residual = term_time + term_advection - term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _compute_residual_By(
        self,
        dBy_dt: torch.Tensor,
        F_conv: torch.Tensor,
        Jz: torch.Tensor,
        eta: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for By component.
        
        Equation: ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η·Jz]
        where Jz = ∂x·By - ∂y·Bx, F_conv = ux·By - uy·Bx
        
        Weak form (after integration by parts):
            ∫φ·∂By/∂t - ∫(∂xφ)·F_conv + ∫(∂xφ)·η·Jz = 0
        
        Args:
            dBy_dt: Time derivative of By
            F_conv: Pre-computed convective flux (ux·By - uy·Bx)
            Jz: Pre-computed current density z-component
            eta: Resistivity field
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂By/∂t
            term_time = phi * dBy_dt
            
            # Advection term (after IBP): ∫(∂xφ)·F_conv
            term_advection = dphi_dx * F_conv
            
            # Resistive diffusion term (after IBP): ∫(∂xφ)·η·Jz
            if self.use_resistive:
                term_resistive = dphi_dx * (eta * Jz)
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂By/∂t - ∫(∂xφ)·F_conv + ∫(∂xφ)·η·Jz = 0
            residual = term_time - term_advection + term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _apply_causal_weighting(self, residuals: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply causal weighting to residuals (Wang et al. 2022).
        
        Weights residuals at time t based on cumulative loss at time < t.
        w(t) = exp(-epsilon * sum_{tau < t} L(tau))
        
        Args:
            residuals: [N, 1] or [N, C] residuals
            coords: [N, 3] coordinates (t is index 2)
            
        Returns:
            Weighted residuals
        """
        if not self.use_causal_weighting:
            return residuals
            
        # Extract time coordinate
        t = coords[..., 2]
        
        # Sort by time
        t_sorted, indices = torch.sort(t)
        
        # Compute cumulative loss (L1 norm of residuals)
        # We use the mean of absolute residuals at each time step?
        # Since coordinates are continuous, we can just use cumulative sum on sorted residuals
        
        # Gather residuals in time order
        res_sorted = residuals[indices]
        res_abs = res_sorted.abs().mean(dim=-1) # [N]
        
        # Cumulative sum of losses
        # We want to penalize current time if previous times are not solved
        # weight_i = exp(-tol * sum_{j<i} loss_j)
        # This forces the model to solve early times first
        
        # To make this stable, we bucket by time or use a running sum
        cumulative_loss = torch.cumsum(res_abs, dim=0)
        
        # Offset: w[i] depends on sum_{j<i}, so shift
        cumulative_loss = torch.roll(cumulative_loss, 1, 0)
        cumulative_loss[0] = 0.0
        
        # Compute weights
        # Normalize cumulative loss to avoid vanishing gradients too fast?
        # Wang et al. use a specific epsilon. Here we use causal_tol
        weights = torch.exp(-self.causal_tol * cumulative_loss)
        
        # Apply weights (shape [N, 1])
        weights = weights.unsqueeze(-1)
        res_weighted = res_sorted * weights
        
        # Scatter back to original order? 
        # Actually, since we are summing the loss anyway, we can just return the weighted sorted residuals
        # But for consistency with other returns (like info), we might want original order.
        # The loss computation takes mean(). sum(res_weighted^2) is the same sorted or unsorted.
        # BUT, we return 'info' with max/mean stats. Those should be on the weighted or unweighted?
        # Usually we want to minimize Weighted Loss.
        
        # Let's unsort to preserve correspondence with inputs
        res_unsorted = torch.empty_like(residuals)
        res_unsorted.scatter_(0, indices.unsqueeze(-1).expand_as(residuals), res_weighted)
        
        return res_unsorted

    def _normalize_residuals(
        self,
        residuals: torch.Tensor,
        B_abs: torch.Tensor,
        u_abs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive normalization to residuals.
        
        Uses exponential moving average of field scales for stability,
        avoiding noisy per-batch percentile estimates.
        
        FIX: Properly handle first iteration by using batch stats directly
        when EMA hasn't been initialized yet.
        """
        if self.normalization == "adaptive":
            # Compute batch statistics
            with torch.no_grad():
                # FIXED: Handle edge case where B_abs or u_abs are all zeros
                B_flat = B_abs.flatten()
                u_flat = u_abs.flatten()
                
                # FIXED: Avoid torch.quantile on MPS - use sort-based exact calculation
                if B_flat.numel() > 0 and (B_flat > 0).any():
                    B_positive = B_flat[B_flat > 0]
                    B_sorted = torch.sort(B_positive).values
                    idx = int(0.75 * (B_sorted.numel() - 1))
                    B_q75 = B_sorted[idx].clamp(min=0.05)
                else:
                    B_q75 = torch.tensor(0.1, device=B_abs.device)
                    
                if u_flat.numel() > 0 and (u_flat > 0).any():
                    u_positive = u_flat[u_flat > 0]
                    u_sorted = torch.sort(u_positive).values
                    idx = int(0.75 * (u_sorted.numel() - 1))
                    u_q75 = u_sorted[idx].clamp(min=0.02)
                else:
                    u_q75 = torch.tensor(0.05, device=u_abs.device)
                
                # Update EMA if training (more stable than raw percentiles)
                if self.training:
                    # FIXED: Wrap EMA updates in no_grad to prevent graph accumulation on MPS
                    with torch.no_grad():
                        # FIXED: More robust check for first iteration
                        # Use absolute comparison with small tolerance
                        is_initial = torch.abs(self._B_scale_ema - 1.0) < 1e-6
                        if is_initial:
                            # First update: initialize directly to batch stats
                            # FIXED: Avoid .item() which can hang on MPS
                            self._B_scale_ema.copy_(B_q75.detach())
                            self._u_scale_ema.copy_(u_q75.detach())
                        else:
                            self._B_scale_ema.lerp_(B_q75.detach(), 1 - self._ema_momentum)
                            self._u_scale_ema.lerp_(u_q75.detach(), 1 - self._ema_momentum)
                
                # Use EMA for normalization - add safety clamps
                B_scale = self._B_scale_ema.clamp(min=0.05, max=100.0)
                u_scale = self._u_scale_ema.clamp(min=0.02, max=10.0)
            
            # FIXED: Clamp norm_factor more aggressively and add larger epsilon
            norm_factor = (B_scale * u_scale).clamp(min=1e-4).sqrt().clamp(min=0.1, max=10.0)
            return residuals / (norm_factor.detach() + 1e-6)
        elif self.normalization == "per_scale":
            with torch.no_grad():
                residual_scales = residuals.abs().mean(dim=0, keepdim=True).clamp(min=1e-6)
            return residuals / residual_scales
        return residuals  # "fixed": no normalization
    
    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, VectorPhysicsResidualInfo]:
        """
        Compute physics loss from weak-form residuals for full vector B field.
        
        Args:
            model: Neural field model that maps coords -> {"B": [N,3], "u": [N,2], "eta_raw": [N,1]}
                   B = (Bx, By, Bz), u = (ux, uy)
            coords: [N, 3] collocation points with requires_grad=True
            imp_weights: [N, 1] importance weights (normalized, mean=1)
            eta_mode: "scalar" (fixed) or "field" (learned spatially-varying)
            eta_scalar: Value of eta when eta_mode="scalar"
            
        Returns:
            loss: Scalar physics loss
            info: VectorPhysicsResidualInfo with component-wise diagnostics
        """
        # Forward through model to get fields
        out = model(coords)
        get_field = lambda name: self._get_output_field(out, name)
        
        # Handle both old (scalar) and new (vector) output formats
        # PINNBackboneOutput supports dict-style access or attribute access
        B = get_field("B")
        if B is not None:
            # New vector format: B is [N, 3]
            Bx, By, Bz = B[..., 0:1], B[..., 1:2], B[..., 2:3]
        else:
            # Legacy format: individual components
            Bx = get_field("B_x")
            By = get_field("B_y")
            Bz = get_field("B_z")
            
            # If Bx, By not available, fall back to scalar-only physics
            if Bx is None or By is None:
                return self._forward_scalar_fallback(
                    model, coords, imp_weights, out, eta_mode, eta_scalar
                )
        
        if Bz is None:
            raise ValueError("Model output missing required magnetic field component B_z.")
        
        # Get velocity components
        u = get_field("u")
        if u is not None:
            # New vector format: u is [N, 2]
            ux, uy = u[..., 0:1], u[..., 1:2]
        else:
            # Legacy format
            ux = get_field("u_x")
            uy = get_field("u_y")
            if ux is None or uy is None:
                raise ValueError("Model output missing required velocity fields (u_x, u_y).")
        
        # Get resistivity
        eta_raw = get_field("eta_raw")
             
        if eta_raw is not None and eta_mode == "field":
            eta = torch.sigmoid(eta_raw)
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            eta_l2_reg = self.tv_eta * (eta ** 2).mean() if self.tv_eta > 0 else (Bz * 0).sum()
            tv_reg = eta_l2_reg
        else:
            eta = Bz.new_full(Bz.shape, float(eta_scalar))
            tv_reg = (Bz * 0).sum()  # Connected to computation graph
        
        # Compute all required gradients
        dBx_dx, dBx_dy, dBx_dt = self._compute_spatial_gradients(Bx, coords, "Bx")
        dBy_dx, dBy_dy, dBy_dt = self._compute_spatial_gradients(By, coords, "By")
        dBz_dx, dBz_dy, dBz_dt = self._compute_spatial_gradients(Bz, coords, "Bz")
        
        # Check for gradient computation failures
        if (dBx_dt.abs().max() == 0 and dBy_dt.abs().max() == 0 and dBz_dt.abs().max() == 0):
            # All gradients are zero - likely a computation failure
            zero_loss = Bz.new_tensor(0.0, requires_grad=True)
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=0.0, loss_induction_By=0.0, loss_induction_Bz=0.0,
                loss_induction_total=0.0, loss_divergence_B=0.0, loss_divergence_u=0.0,
                loss_tv_eta=0.0, residual_Bx_mean=0.0, residual_By_mean=0.0,
                residual_Bz_mean=0.0, residual_Bx_max=0.0, residual_By_max=0.0,
                residual_Bz_max=0.0
            )
            return zero_loss, info
        
        # Pre-compute shared terms (optimization: avoid redundant computation)
        # Current density z-component: Jz = ∂x·By - ∂y·Bx
        Jz = dBy_dx - dBx_dy
        
        # Convective flux: F_conv = ux·By - uy·Bx (shared by Bx and By equations)
        F_conv = ux * By - uy * Bx
        
        # Get component-specific test functions (mathematically correct only)
        phis_Bx, dphi_dx_Bx, dphi_dy_Bx = self.test_fn(coords, component="Bx")
        phis_By, dphi_dx_By, dphi_dy_By = self.test_fn(coords, component="By")
        phis_Bz, dphi_dx_Bz, dphi_dy_Bz = self.test_fn(coords, component="Bz")
        
        # Compute residuals for each component with their matched test functions
        residuals_Bx = self._compute_residual_Bx(
            dBx_dt, F_conv, Jz, eta,
            phis_Bx, dphi_dx_Bx, dphi_dy_Bx
        )
        residuals_By = self._compute_residual_By(
            dBy_dt, F_conv, Jz, eta,
            phis_By, dphi_dx_By, dphi_dy_By
        )
        residuals_Bz = self._compute_residual_Bz(
            Bz, ux, uy, eta, dBz_dx, dBz_dy, dBz_dt,
            phis_Bz, dphi_dx_Bz, dphi_dy_Bz
        )
        
        # Normalize residuals
        B_abs = (Bx.abs() + By.abs() + Bz.abs()) / 3.0
        u_abs = (ux.abs() + uy.abs()) / 2.0
        
        residuals_Bx = self._normalize_residuals(residuals_Bx, B_abs, u_abs)
        residuals_By = self._normalize_residuals(residuals_By, B_abs, u_abs)
        residuals_Bz = self._normalize_residuals(residuals_Bz, B_abs, u_abs)
        
        # Apply causal weighting if enabled
        if self.use_causal_weighting:
            residuals_Bx = self._apply_causal_weighting(residuals_Bx, coords)
            residuals_By = self._apply_causal_weighting(residuals_By, coords)
            residuals_Bz = self._apply_causal_weighting(residuals_Bz, coords)
        
        # Clamp and handle NaN
        def safe_residual(r: torch.Tensor) -> torch.Tensor:
            r = r.clamp(-10.0, 10.0)
            return torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)
        
        residuals_Bx = safe_residual(residuals_Bx)
        residuals_By = safe_residual(residuals_By)
        residuals_Bz = safe_residual(residuals_Bz)
        
        # Compute component losses (importance-weighted MSE)
        imp_weights_safe = torch.nan_to_num(imp_weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        loss_Bx = ((residuals_Bx ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        loss_By = ((residuals_By ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        loss_Bz = ((residuals_Bz ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        
        # Weighted combination of component losses
        w_Bx, w_By, w_Bz = self.component_weights
        
        if self.use_uncertainty_weighting and hasattr(self, 'log_var_Bx'):
            # Learned uncertainty weighting (Kendall et al. 2018)
            # L = (1/2σ²) * loss + log(σ) = (1/2) * exp(-log_var) * loss + 0.5 * log_var
            precision_Bx = torch.exp(-self.log_var_Bx)
            precision_By = torch.exp(-self.log_var_By)
            precision_Bz = torch.exp(-self.log_var_Bz)
            
            loss_induction = (
                0.5 * precision_Bx * loss_Bx + 0.5 * self.log_var_Bx +
                0.5 * precision_By * loss_By + 0.5 * self.log_var_By +
                0.5 * precision_Bz * loss_Bz + 0.5 * self.log_var_Bz
            )
        else:
            # Fixed weights with robust normalization using smooth Huber-like scaling
            # sqrt(x + delta^2) - delta has bounded gradients: x/sqrt(x + delta^2)
            # This approaches 0 as x→0 (stable!) and 1 as x→∞ (linear scaling for large losses)
            delta = 0.1  # Transition point between quadratic and linear regimes
            delta_sq = delta * delta
            
            def smooth_scale(x: torch.Tensor) -> torch.Tensor:
                """Numerically stable loss scaling with bounded gradients."""
                # FIXED: Clamp input to ensure non-negative and add stronger minimum
                x_safe = x.clamp(min=0.0)
                return (x_safe + delta_sq).clamp(min=1e-8).sqrt() - delta
            
            # FIXED: Clamp individual losses before scaling
            loss_induction = (
                w_Bx * smooth_scale(loss_Bx.clamp(max=100.0)) + 
                w_By * smooth_scale(loss_By.clamp(max=100.0)) + 
                w_Bz * smooth_scale(loss_Bz.clamp(max=100.0))
            )
        
        # Solenoidal constraint: ∇·B = ∂x·Bx + ∂y·By = 0
        loss_div_B = (Bz * 0).sum()  # Connected to computation graph
        if self.enforce_div_free_B:
            div_B = dBx_dx + dBy_dy
            loss_div_B = self.div_B_weight * ((div_B ** 2) * imp_weights_safe).mean()
        
        # Optional: Divergence-free velocity constraint
        loss_div_u = (Bz * 0).sum()  # Connected to computation graph
        if self.enforce_div_free_u:
            dux_dx, _, _ = self._compute_spatial_gradients(ux, coords, "ux")
            _, duy_dy, _ = self._compute_spatial_gradients(uy, coords, "uy")
            div_u = dux_dx + duy_dy
            loss_div_u = ((div_u ** 2) * imp_weights_safe).mean()
        
        # SOTA: Force-free constraint (J × B ≈ 0 in low-β corona)
        # This is critical for flare prediction as non-force-free regions indicate
        # magnetic stress that can trigger reconnection.
        # J × B = (Jx*By - Jy*Bx, Jy*Bz - Jz*By, Jz*Bx - Jx*Bz)
        # In 2.5D: Jx = ∂y·Bz, Jy = -∂x·Bz, Jz = ∂x·By - ∂y·Bx
        # Force-free means: α*B = J where α is constant (or slowly varying)
        # Soft constraint: minimize |J × B|² / |B|²
        loss_force_free = (Bz * 0).sum()  # Connected to computation graph
        if self.enforce_force_free:
            Jx = dBz_dy
            Jy = -dBz_dx
            # Jz already computed as current density
            
            # Cross product components (J × B)
            JxB_x = Jx * By - Jy * Bx
            JxB_y = Jy * Bz - Jz * By  
            JxB_z = Jz * Bx - Jx * Bz
            
            JxB_sq = JxB_x**2 + JxB_y**2 + JxB_z**2
            # FIX: Add larger epsilon for division stability
            B_sq = (Bx**2 + By**2 + Bz**2).clamp(min=1e-6)
            
            # Normalized force-free violation
            ff_violation = JxB_sq / B_sq
            loss_force_free = self.force_free_weight * (ff_violation * imp_weights_safe).mean()
        
        # Total physics loss
        total_loss = loss_induction + loss_div_B + 0.1 * loss_div_u + loss_force_free + tv_reg
        
        # Create diagnostic info
        with torch.no_grad():
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=float(loss_Bx.detach()),
                loss_induction_By=float(loss_By.detach()),
                loss_induction_Bz=float(loss_Bz.detach()),
                loss_induction_total=float(loss_induction.detach()),
                loss_divergence_B=float(loss_div_B.detach()),
                loss_divergence_u=float(loss_div_u.detach()),
                loss_tv_eta=float(tv_reg.detach()),
                residual_Bx_mean=float(residuals_Bx.abs().mean().detach()),
                residual_By_mean=float(residuals_By.abs().mean().detach()),
                residual_Bz_mean=float(residuals_Bz.abs().mean().detach()),
                residual_Bx_max=float(residuals_Bx.abs().max().detach()),
                residual_By_max=float(residuals_By.abs().max().detach()),
                residual_Bz_max=float(residuals_Bz.abs().max().detach()),
            )
        
        return total_loss, info
    
    def _forward_scalar_fallback(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        out: dict,
        eta_mode: str,
        eta_scalar: float
    ) -> tuple[torch.Tensor, VectorPhysicsResidualInfo]:
        """
        Fallback to scalar Bz-only physics when Bx, By not available.
        
        This maintains backward compatibility with models that only output Bz.
        """
        Bz = self._get_output_field(out, "B_z")
        ux = self._get_output_field(out, "u_x")
        uy = self._get_output_field(out, "u_y")
        if Bz is None or ux is None or uy is None:
            raise ValueError("Model output missing required scalar physics fields (B_z, u_x, u_y).")
        
        # Get resistivity
        eta_raw = self._get_output_field(out, "eta_raw")
        if eta_raw is not None and eta_mode == "field":
            eta = torch.sigmoid(eta_raw)
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            tv_reg = self.tv_eta * (eta ** 2).mean() if self.tv_eta > 0 else (Bz * 0).sum()
        else:
            eta = Bz.new_full(Bz.shape, float(eta_scalar))
            tv_reg = (Bz * 0).sum()  # Connected to computation graph
        
        # Compute gradients
        dBz_dx, dBz_dy, dBz_dt = self._compute_spatial_gradients(Bz, coords, "Bz")
        
        # Get test functions for Bz component only
        phis, dphi_dx_list, dphi_dy_list = self.test_fn(coords, component="Bz")
        
        # Compute Bz residuals only
        residuals_Bz = self._compute_residual_Bz(
            Bz, ux, uy, eta, dBz_dx, dBz_dy, dBz_dt,
            phis, dphi_dx_list, dphi_dy_list
        )
        
        # Normalize
        B_abs = Bz.abs()
        u_abs = (ux.abs() + uy.abs()) / 2.0
        residuals_Bz = self._normalize_residuals(residuals_Bz, B_abs, u_abs)
        residuals_Bz = torch.nan_to_num(residuals_Bz.clamp(-10.0, 10.0), nan=0.0)
        
        # Loss
        imp_weights_safe = torch.nan_to_num(imp_weights, nan=1.0, posinf=1.0, neginf=1.0)
        loss_Bz = ((residuals_Bz ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        
        total_loss = loss_Bz + tv_reg
        
        with torch.no_grad():
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=0.0,
                loss_induction_By=0.0,
                loss_induction_Bz=float(loss_Bz.detach()),
                loss_induction_total=float(loss_Bz.detach()),
                loss_divergence_B=0.0,
                loss_divergence_u=0.0,
                loss_tv_eta=float(tv_reg.detach()),
                residual_Bx_mean=0.0,
                residual_By_mean=0.0,
                residual_Bz_mean=float(residuals_Bz.abs().mean().detach()),
                residual_Bx_max=0.0,
                residual_By_max=0.0,
                residual_Bz_max=float(residuals_Bz.abs().max().detach()),
            )
        
        return total_loss, info


# Backward compatibility alias
class WeakFormInduction2p5D(VectorInduction2p5D):
    """
    Legacy alias for backward compatibility.
    
    This class wraps VectorInduction2p5D but returns the legacy
    PhysicsResidualInfo format for existing training code.
    """
    
    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, PhysicsResidualInfo]:
        """Forward with legacy return type."""
        loss, vector_info = super().forward(model, coords, imp_weights, eta_mode, eta_scalar)
        
        # Convert to legacy format
        legacy_info = PhysicsResidualInfo(
            loss_induction=vector_info.loss_induction_total,
            loss_divergence=vector_info.loss_divergence_B + vector_info.loss_divergence_u,
            loss_tv_eta=vector_info.loss_tv_eta,
            residual_mean=(
                vector_info.residual_Bx_mean + 
                vector_info.residual_By_mean + 
                vector_info.residual_Bz_mean
            ) / 3.0,
            residual_max=max(
                vector_info.residual_Bx_max,
                vector_info.residual_By_max,
                vector_info.residual_Bz_max
            ),
        )
        
        return loss, legacy_info


class FreeEnergyProxy(nn.Module):
    """
    Compute free magnetic energy proxy from PINN outputs.
    
    Free energy E_free ∝ ∫ (B - B_potential)² dV
    
    This is a key predictor of flare potential that can be computed
    from the PINN's magnetic field representation.
    
    Enhanced for vector field inputs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        B_z: torch.Tensor,
        B_x: Optional[torch.Tensor],
        B_y: Optional[torch.Tensor],
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute free energy proxy.
        
        For PINN outputs, we can approximate this by measuring
        the non-potentiality of the field configuration.
        
        Args:
            B_z: [N, 1] vertical field
            B_x, B_y: [N, 1] horizontal field components (optional)
            coords: [N, 3] coordinates
            
        Returns:
            free_energy: [1] scalar proxy value
        """
        if B_x is None or B_y is None:
            # Fallback: use Bz variance as proxy for complexity
            return B_z.var()
        
        # Compute |∇Bz| as proxy for shear
        grads = torch.autograd.grad(
            B_z, coords,
            grad_outputs=torch.ones_like(B_z),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_Bz_mag = (grads[..., 0:1]**2 + grads[..., 1:2]**2).sqrt()
        
        # Current helicity proxy: Jz * Bz where Jz = dBy/dx - dBx/dy
        if coords.requires_grad and B_x.requires_grad and B_y.requires_grad:
            grads_Bx = torch.autograd.grad(
                B_x, coords,
                grad_outputs=torch.ones_like(B_x),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grads_By = torch.autograd.grad(
                B_y, coords,
                grad_outputs=torch.ones_like(B_y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            Jz = grads_By[..., 0:1] - grads_Bx[..., 1:2]
            helicity_proxy = (Jz * B_z).abs().mean()
            
            # Combined metric
            return grad_Bz_mag.mean() + 0.5 * helicity_proxy
        
        return grad_Bz_mag.mean()


class CurrentHelicityProxy(nn.Module):
    """
    Compute current helicity proxy from PINN outputs.
    
    Current helicity H_c = ∫ B · J dV
    
    where J = ∇×B is the current density.
    This measures the twist and complexity of the magnetic field.
    
    Enhanced for vector field inputs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        A_z: torch.Tensor,
        B_z: torch.Tensor,
        coords: torch.Tensor,
        B_x: Optional[torch.Tensor] = None,
        B_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute current helicity proxy.
        
        For 2.5D, J = ∇×B has components:
            Jx = ∂y Bz (since ∂z By = 0)
            Jy = -∂x Bz (since ∂z Bx = 0)  
            Jz = ∂x By - ∂y Bx
            
        Helicity ≈ Bz * Jz + Bx * Jx + By * Jy
        
        Args:
            A_z: [N, 1] vector potential
            B_z: [N, 1] vertical field
            coords: [N, 3] coordinates
            B_x, B_y: [N, 1] horizontal field components (optional)
            
        Returns:
            helicity: [1] scalar proxy value
        """
        # Compute ∇Bz for Jx, Jy
        grad_Bz = torch.autograd.grad(
            B_z, coords,
            grad_outputs=torch.ones_like(B_z),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        Jx = grad_Bz[..., 1:2]   # ∂y Bz
        Jy = -grad_Bz[..., 0:1]  # -∂x Bz
        
        if B_x is not None and B_y is not None:
            # Full helicity: B · J = Bx*Jx + By*Jy + Bz*Jz
            
            # Compute Jz = ∂x By - ∂y Bx
            grad_Bx = torch.autograd.grad(
                B_x, coords,
                grad_outputs=torch.ones_like(B_x),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grad_By = torch.autograd.grad(
                B_y, coords,
                grad_outputs=torch.ones_like(B_y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            Jz = grad_By[..., 0:1] - grad_Bx[..., 1:2]
            
            helicity = (B_x * Jx + B_y * Jy + B_z * Jz).abs().mean()
        else:
            # Fallback: use |Bz * Jz| where Jz ≈ ∇²Az
            # Compute Laplacian of Az
            grad_Az = torch.autograd.grad(
                A_z, coords,
                grad_outputs=torch.ones_like(A_z),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            dAz_dx = grad_Az[..., 0:1]
            dAz_dy = grad_Az[..., 1:2]
            
            d2Az_dx2 = torch.autograd.grad(
                dAz_dx.sum(), coords,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0][..., 0:1]
            
            d2Az_dy2 = torch.autograd.grad(
                dAz_dy.sum(), coords,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0][..., 1:2]
            
            Jz = d2Az_dx2 + d2Az_dy2
            helicity = (B_z * Jz).abs().mean()
        
        return helicity
