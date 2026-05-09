# src/utils/memory_optimization.py
"""
Memory optimization utilities for 16GB MPS training.

Key strategies:
1. Gradient accumulation for effective larger batches
2. Gradient checkpointing for memory-compute tradeoff
3. Chunked forward passes for large inputs
4. Memory-efficient attention alternatives
5. Dynamic batch sizing based on sequence length
"""
from __future__ import annotations

import gc
import math
from contextlib import contextmanager
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Memory Monitoring for MPS
# ============================================================================

def get_mps_memory_info() -> dict:
    """
    Get memory info for MPS device.
    
    Note: MPS doesn't expose detailed memory stats like CUDA.
    We estimate based on Python's gc and system monitoring.
    """
    if not torch.backends.mps.is_available():
        return {"device": "not_mps"}
    
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    return {
        "device": "mps",
        "gc_collections": gc.get_count(),
        "gc_objects": len(gc.get_objects()),
    }


def mps_train_interval_cleanup() -> None:
    """
    Hot-path MPS cleanup while stepping: return pooled allocations to the device allocator.

    Intentionally does *not* call mps.synchronize() — a full sync every N train steps
    stalls the pipeline and can make steady-state step time look much worse than
    the actual matmul/physics work. Use aggressive_memory_cleanup() after validation
    or checkpointing when a sync is acceptable.
    """
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def aggressive_memory_cleanup():
    """
    Aggressive memory cleanup for MPS/CPU.
    Call this between training and validation.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


@contextmanager
def low_memory_mode():
    """
    Context manager for low memory operations.
    Reduces fragmentation during memory-intensive operations.
    """
    try:
        # Force garbage collection before memory-intensive work
        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        yield
    finally:
        # Cleanup after
        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


# ============================================================================
# Gradient Checkpointing Utilities
# ============================================================================

class CheckpointedSequential(nn.Module):
    """
    Sequential module with gradient checkpointing.
    
    Trades compute for memory by recomputing activations during backward.
    Critical for fitting larger models in 16GB.
    """
    
    def __init__(self, *modules, checkpoint_segments: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.checkpoint_segments = checkpoint_segments
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and len(self.layers) > self.checkpoint_segments:
            # Split layers into segments and checkpoint each
            segment_size = len(self.layers) // self.checkpoint_segments
            
            for i in range(0, len(self.layers), max(1, segment_size)):
                segment = self.layers[i:i + segment_size]
                
                def run_segment(inp, seg=segment):
                    for layer in seg:
                        inp = layer(inp)
                    return inp
                
                x = torch.utils.checkpoint.checkpoint(
                    run_segment, x, use_reentrant=False
                )
            return x
        else:
            # No checkpointing during eval
            for layer in self.layers:
                x = layer(x)
            return x


def checkpoint_forward(
    func: Callable,
    *args,
    use_checkpoint: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper to optionally apply gradient checkpointing to a function.
    
    Args:
        func: Function to potentially checkpoint
        use_checkpoint: Whether to apply checkpointing (during training)
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Output of func
    """
    if use_checkpoint and torch.is_grad_enabled():
        # Checkpoint expects the first arg to be tensor that requires grad
        # For functions with kwargs, we need a wrapper
        def wrapper(*a):
            return func(*a, **kwargs)
        return torch.utils.checkpoint.checkpoint(wrapper, *args, use_reentrant=False)
    return func(*args, **kwargs)


# ============================================================================
# Chunked Processing for Large Inputs
# ============================================================================

def chunked_forward(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 2048,
    dim: int = 0
) -> torch.Tensor:
    """
    Process input in chunks to reduce peak memory usage.
    
    Useful for:
    - Large collocation point sets
    - Inference on high-resolution magnetograms
    
    Args:
        model: Module to apply
        x: Input tensor
        chunk_size: Maximum chunk size
        dim: Dimension to chunk along
    
    Returns:
        Concatenated outputs
    """
    if x.shape[dim] <= chunk_size:
        return model(x)
    
    outputs = []
    for i in range(0, x.shape[dim], chunk_size):
        chunk = x.narrow(dim, i, min(chunk_size, x.shape[dim] - i))
        out = model(chunk)
        outputs.append(out)
    
    return torch.cat(outputs, dim=dim)


def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 256,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Memory-efficient attention by processing query chunks.
    
    Reduces peak memory from O(N²) to O(N * chunk_size).
    Critical for MPS where memory is limited.
    
    Args:
        query: [B, H, N, D] query tensor
        key: [B, H, M, D] key tensor
        value: [B, H, M, D] value tensor
        chunk_size: Number of queries to process at once
        mask: Optional attention mask [B, 1, N, M] or [B, H, N, M]
    
    Returns:
        Attention output [B, H, N, D]
    """
    B, H, N, D = query.shape
    _, _, M, _ = key.shape
    
    if N <= chunk_size:
        # Standard attention for small sequences
        scale = D ** -0.5
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, value)
    
    # Chunked attention
    outputs = []
    for i in range(0, N, chunk_size):
        q_chunk = query[:, :, i:i + chunk_size, :]
        
        scale = D ** -0.5
        attn = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale
        
        if mask is not None:
            mask_chunk = mask[:, :, i:i + chunk_size, :]
            attn = attn.masked_fill(~mask_chunk, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out_chunk = torch.matmul(attn, value)
        outputs.append(out_chunk)
    
    return torch.cat(outputs, dim=2)


# ============================================================================
# Gradient Accumulation with Memory Management
# ============================================================================

class MPSGradientAccumulator:
    """
    Gradient accumulation optimized for MPS.
    
    Features:
    - Automatic gradient scaling for accumulation
    - Memory cleanup between accumulation steps
    - Support for mixed precision (though MPS has quirks)
    
    Usage:
        accumulator = MPSGradientAccumulator(model, optimizer, accum_steps=4)
        
        for step, batch in enumerate(dataloader):
            loss = compute_loss(batch)
            did_step = accumulator.step(loss, step)
            if did_step:
                scheduler.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accum_steps: int = 4,
        grad_clip: float = 1.0,
        cleanup_every: int = 10
    ):
        self.model = model
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.cleanup_every = cleanup_every
        
        self._accum_count = 0
        self._global_step = 0
        self.last_grad_norm = 0.0  # Store actual gradient norm for logging
    
    def step(self, loss: torch.Tensor, step: int) -> bool:
        """
        Accumulate gradients and step optimizer when ready.
        
        Args:
            loss: Current loss (will be scaled by 1/accum_steps)
            step: Global training step (for logging)
        
        Returns:
            True if optimizer stepped
        """
        #  SAFETY: Check for NaN/Inf in loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            return False

        # Scale loss
        scaled_loss = loss / self.accum_steps
        
        # Backward
        scaled_loss.backward()
        
        #  SAFETY: Check for NaN/Inf gradients in micro-batch
        has_nan = False
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_nan = True
                    break
        
        if has_nan:
            # Discard this micro-batch accumulation
            self.zero_grad()
            return False
        
        self._accum_count += 1
        
        # Step optimizer when accumulated enough
        if self._accum_count >= self.accum_steps:
            # Clip gradients and store actual norm
            if self.grad_clip > 0:
                self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip
                ).item()
            else:
                # Compute norm without clipping for logging
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                self.last_grad_norm = total_norm ** 0.5
            
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            self._accum_count = 0
            self._global_step += 1
            
            # Periodic memory cleanup to prevent fragmentation
            if self._global_step % self.cleanup_every == 0:
                gc.collect()
            
            return True
        
        return False
    
    def zero_grad(self):
        """Reset gradients."""
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_count = 0
    
    @property
    def is_accumulating(self) -> bool:
        """Check if currently accumulating (not yet stepped)."""
        return self._accum_count > 0


# ============================================================================
# Dynamic Batch Sizing
# ============================================================================

def estimate_batch_memory(
    model: nn.Module,
    sample_input: torch.Tensor,
    target_memory_gb: float = 10.0
) -> int:
    """
    Estimate maximum batch size for target memory budget.
    
    For MPS with 16GB unified memory, recommend target_memory_gb=10
    to leave headroom for system and Metal overhead.
    
    Args:
        model: The model to estimate for
        sample_input: A single sample input tensor
        target_memory_gb: Target memory budget in GB
    
    Returns:
        Estimated maximum batch size
    """
    # Get model parameter memory
    param_memory = sum(
        p.numel() * p.element_size() 
        for p in model.parameters()
    )
    
    # Estimate activation memory (rough: 2x params for forward, 3x for backward)
    activation_factor = 4.0  # Forward + backward + optimizer states
    
    # Memory per sample (very rough estimate)
    sample_memory = sample_input.numel() * sample_input.element_size()
    
    total_model_memory = param_memory * activation_factor
    remaining_memory = target_memory_gb * 1e9 - total_model_memory
    
    # Estimate batch size with safety margin
    estimated_batch = int(remaining_memory / (sample_memory * 10))  # 10x factor for safety
    
    return max(1, min(32, estimated_batch))


# ============================================================================
# Memory-Efficient Attention Variants
# ============================================================================

class EfficientSelfAttention(nn.Module):
    """
    Memory-efficient self-attention for MPS.
    
    Uses chunked computation to reduce peak memory.
    Alternative to standard attention when sequence is long.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        chunk_size: int = 256
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input features
            mask: [B, N] boolean mask (True = valid)
        
        Returns:
            [B, N, C] output features
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]  # [B, 1, 1, N]
        
        # Use chunked attention for memory efficiency
        out = chunked_attention(
            q, k, v,
            chunk_size=self.chunk_size,
            mask=attn_mask
        )
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class LinearAttention(nn.Module):
    """
    Linear attention approximation with O(N) complexity.
    
    Uses kernel feature maps to approximate softmax attention.
    Much more memory efficient for long sequences.
    
    Based on "Transformers are RNNs" (Katharopoulos et al., 2020)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        feature_map: str = "elu",  # "elu" or "softmax"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.feature_map = feature_map
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map for kernel approximation."""
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        elif self.feature_map == "softmax":
            return F.softmax(x, dim=-1)
        else:
            return x
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input features
            mask: [B, N] boolean mask (True = valid)
        
        Returns:
            [B, N, C] output features
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)
        
        # Apply mask to k, v
        if mask is not None:
            mask_expanded = mask[:, None, :, None]  # [B, 1, N, 1]
            k = k * mask_expanded.float()
            v = v * mask_expanded.float()
        
        # Linear attention: O(N) instead of O(N²)
        # attn = Q @ (K^T @ V) instead of (Q @ K^T) @ V
        kv = torch.einsum('bhnd,bhnc->bhdc', k, v)  # [B, H, D, D]
        qkv_out = torch.einsum('bhnd,bhdc->bhnc', q, kv)  # [B, H, N, D]
        
        # Normalization
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).squeeze(-1)  # [B, H, N]
        normalizer = normalizer.clamp(min=1e-6)
        
        out = qkv_out / normalizer.unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


# ============================================================================
# Activation Checkpointing Wrapper
# ============================================================================

class ActivationCheckpointWrapper(nn.Module):
    """
    Wrapper that applies activation checkpointing to any module.
    
    Usage:
        layer = ActivationCheckpointWrapper(heavy_layer)
    """
    
    def __init__(self, module: nn.Module, enabled: bool = True):
        super().__init__()
        self.module = module
        self.enabled = enabled
    
    def forward(self, *args, **kwargs):
        if self.enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                **kwargs
            )
        return self.module(*args, **kwargs)


# ============================================================================
# Memory Budget Manager
# ============================================================================

class MemoryBudgetManager:
    """
    Manages memory budget during training.
    
    Dynamically adjusts settings based on memory pressure:
    - Collocation points
    - Batch processing chunks
    - Checkpointing
    """
    
    def __init__(
        self,
        target_memory_gb: float = 12.0,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95
    ):
        self.target_memory_gb = target_memory_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._consecutive_warnings = 0
        self._reduction_factor = 1.0
    
    def check_and_adjust(self) -> dict:
        """
        Check memory and return recommended adjustments.
        
        Returns:
            Dictionary with recommended settings
        """
        # For MPS, we can't directly query memory, so track indirectly
        gc.collect()
        
        recommendations = {
            'should_reduce_collocation': self._consecutive_warnings > 3,
            'reduction_factor': self._reduction_factor,
            'cleanup_recommended': self._consecutive_warnings > 0,
        }
        
        return recommendations
    
    def report_oom(self):
        """Report an OOM event."""
        self._consecutive_warnings += 1
        self._reduction_factor *= 0.8  # Reduce by 20%
    
    def report_success(self):
        """Report successful step."""
        if self._consecutive_warnings > 0:
            self._consecutive_warnings -= 1
        if self._reduction_factor < 1.0:
            self._reduction_factor = min(1.0, self._reduction_factor * 1.05)

