"""
Strong-form PINN baseline for 2.5D resistive induction equation.

This module provides a strong-form (pointwise PDE residual) implementation
as a baseline comparison against the weak-form approach. The strong-form
requires computing second derivatives via autograd, which can be numerically
stiff when applied to noisy magnetogram observations.

Usage:
    from src.baselines.strong_form import StrongFormInduction2p5D, StrongFormConfig
"""

from .physics import StrongFormInduction2p5D, StrongFormConfig, compute_strong_form_phys_loss

__all__ = ["StrongFormInduction2p5D", "StrongFormConfig", "compute_strong_form_phys_loss"]

