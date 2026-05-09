# src/models/pinn/film.py
"""
Feature-wise Linear Modulation (FiLM) for conditioning neural networks.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation.
    
    Applies affine transformation to hidden activations:
        FiLM(h; gamma, beta) = gamma ⊙ h + beta
    
    where gamma and beta are predicted from conditioning code.
    
    References:
        Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
    """
    def __init__(self, hidden_dim: int, code_dim: int):
        super().__init__()
        # Predict both scale (gamma) and shift (beta)
        self.mlp = nn.Linear(code_dim, hidden_dim * 2)
        
        # FIXED: Initialize with small random weights (NOT zero!) for gradient flow
        # and correct biases for identity modulation at initialization
        with torch.no_grad():
            # Small random weights so gradients can flow through conditioning
            nn.init.normal_(self.mlp.weight, mean=0.0, std=0.02)
            # gamma_raw = 0 → tanh(0)*2+1 = 1.0 (identity scale)
            self.mlp.bias[:hidden_dim].fill_(0.0)
            # beta = 0 (no shift)
            self.mlp.bias[hidden_dim:].fill_(0.0)
    
    def forward(self, h: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [N, hidden_dim] - Hidden activations
            code: [N, code_dim] - Conditioning code (e.g., global features)
        
        Returns:
            modulated: [N, hidden_dim] - FiLM-modulated activations
        """
        # Handle NaN/Inf in inputs - use any() to avoid MPS sync issues
        with torch.no_grad():
            if torch.isnan(code).any() or torch.isinf(code).any():
                code = torch.nan_to_num(code, nan=0.0, posinf=1.0, neginf=-1.0)
        
        gam_beta = self.mlp(code)  # [N, 2*hidden_dim]
        gamma_raw, beta = gam_beta.chunk(2, dim=-1)
        
        # FIXED: Softer gamma transformation for better gradient flow
        # tanh(x) * 0.5 + 1.0 gives range [0.5, 1.5] centered at 1.0
        # This prevents extreme scaling while allowing modulation
        gamma = torch.tanh(gamma_raw) * 0.5 + 1.0
        
        # Clamp beta to prevent extreme shifts (but less aggressive)
        beta = beta.clamp(-5.0, 5.0)
        
        result = gamma * h + beta
        
        # Softer output clamping (let gradients flow better)
        return result.clamp(-100.0, 100.0)

