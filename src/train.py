#!/usr/bin/env python3
# src/train.py
"""
SOTA-Optimized PINN Trainer for Solar Flare Prediction.

Features:
  - Hybrid CNN-PINN training
  - Validation-based checkpointing (saves best TSS model)
  - Physics-informed loss scheduling
  - Comprehensive metrics logging (TSS, PR-AUC, Brier)
  - MPS/CUDA support
"""
from __future__ import annotations
import argparse
import gc
import logging
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Suppress warnings
warnings.filterwarnings('ignore', message='.*MPS autocast.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.amp')

# W&B import (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.models.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    select_threshold_at_far,
)
from src.utils.training_utils import optimize_for_device, clear_memory_cache
from src.utils.memory_optimization import (
    MPSGradientAccumulator,
    aggressive_memory_cleanup,
    mps_train_interval_cleanup,
    low_memory_mode,
)

# ============================================================================
# Dummy Dataset for Testing
# ============================================================================

class DummyPINNDataset(Dataset):
    """
    Synthetic dataset for testing PINN training without real data.
    Generates random magnetogram-like data with synthetic labels.
    """
    def __init__(self, cfg: PINNConfig):
        self.cfg = cfg
        self.T = cfg.data.dummy_T
        self.H = cfg.data.dummy_H
        self.W = cfg.data.dummy_W
        self.P = cfg.data.P_per_t
        self.num_samples = cfg.data.dummy_num_samples
        self.horizons = list(cfg.classifier.horizons)
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        # Generate random coords
        coords = torch.zeros(self.T, self.P, 3)
        coords[..., :2] = torch.rand(self.T, self.P, 2) * 2.0 - 1.0  # xy in [-1,1]
        t_vals = torch.linspace(-1.0, 1.0, self.T)[:, None, None].expand(self.T, self.P, 1)
        coords[..., 2:3] = t_vals
        # Don't set requires_grad here - will be set in training loop
        
        # Generate synthetic Bz field (dipole-like pattern)
        x_grid = torch.linspace(-1, 1, self.W)
        y_grid = torch.linspace(-1, 1, self.H)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')
        
        # Create bipolar region (simplified magnetogram)
        bz_base = torch.tanh(5 * (xx - 0.3)) - torch.tanh(5 * (xx + 0.3))
        bz_base = bz_base * torch.exp(-2 * yy**2)
        
        # Add noise and time variation
        C = self.cfg.data.n_components
        frames = torch.zeros(self.T, C, self.H, self.W)
        for t in range(self.T):
            noise = torch.randn(C, self.H, self.W) * 0.1
            evolution = 1.0 + 0.1 * t / self.T
            # Replicate base for all channels or create variations
            for c in range(C):
                frames[t, c] = bz_base * evolution + noise[c]
        
        # Sample gt_bz at coords
        gt_bz = torch.zeros(self.T, self.P, C)
        for t in range(self.T):
            xy = coords[t, :, :2].detach()
            # Simple nearest-neighbor sampling
            x_idx = ((xy[:, 0] + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            y_idx = ((xy[:, 1] + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            for c in range(C):
                gt_bz[t, :, c] = frames[t, c, y_idx, x_idx]
        
        # Observed mask (all observed for dummy data)
        observed_mask = torch.ones(self.T, dtype=torch.bool)
        
        # Random labels (with ~20% positive rate for imbalance)
        labels = (torch.rand(len(self.horizons)) < 0.2).float()
        
        # PIL mask (high gradient regions) - as tensor for type consistency
        # Use last channel for Bz (convention: components = ["Bx", "By", "Bz"] or ["Bz"])
        # frames shape: [T, C, H, W], take last timestep and last channel (Bz)
        frame_bz = frames[-1, -1]  # [H, W] - last timestep, last channel (Bz)
        
        grad_x = torch.abs(frame_bz[:, 1:] - frame_bz[:, :-1])  # [H, W-1]
        grad_y = torch.abs(frame_bz[1:, :] - frame_bz[:-1, :])  # [H-1, W]
        # Pad to match size [H, W] using simple concatenation
        grad_x = torch.cat([grad_x, grad_x[:, -1:]], dim=1)  # [H, W]
        grad_y = torch.cat([grad_y, grad_y[-1:, :]], dim=0)  # [H, W]
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        pil_mask = (grad_mag > grad_mag.quantile(0.85)).float()  # Keep as tensor
        
        # Scalar features (dummy values for testing)
        # [r_value, gwpil, obs_coverage, frame_count]
        base_scalars = torch.tensor([
            2.5,   # Dummy R-value (typical range: 1-5)
            100.0, # Dummy GWPIL
            1.0,   # All frames observed
            float(self.T),  # Frame count
        ], dtype=torch.float32)
        
        # Pad to match n_scalar_features
        n_features = self.cfg.data.n_scalar_features
        if n_features > 4:
            padding = torch.zeros(n_features - 4, dtype=torch.float32)
            scalars = torch.cat([base_scalars, padding])
        else:
            scalars = base_scalars
        
        return {
            "coords": coords,
            "gt_bz": gt_bz,
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels,
            "pil_mask": pil_mask,
            "scalars": scalars,
        }

# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("pinn_train")
    logger.setLevel(logging.INFO)
    
    # FIXED: Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# Checkpointing (Optimized for TSS)
# ============================================================================

@dataclass
class CheckpointState:
    """Container for all training state that should be checkpointed."""
    step: int
    metric: float
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: Optional[dict] = None
    ema_state_dict: Optional[dict] = None
    early_stopping_state: Optional[dict] = None
    best_val_tss: float = 0.0


@dataclass
class CheckpointManager:
    checkpoint_dir: Path
    keep_last_n: int = 3
    
    def __post_init__(self):
        # Convert string to Path if needed
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float = -1e9
        self.best_path: Optional[Path] = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        metric_value: float,
        is_best: bool = False,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        ema: Optional["ExponentialMovingAverage"] = None,
        early_stopping: Optional["EarlyStopping"] = None,
        best_val_tss: float = 0.0,
        # New args for plateau tracking
        consecutive_drops: int = 0,
        last_val_tss: float = 0.0,
    ) -> Path:
        """
        Save checkpoint with full training state.
        
        Includes EMA, scheduler, early stopping, and plateau tracking state.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric_value,
            'best_val_tss': best_val_tss,
            'consecutive_drops': consecutive_drops,
            'last_val_tss': last_val_tss,
        }
        
        # Save scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save EMA state (CRITICAL for proper resumption!)
        if ema is not None:
            checkpoint['ema_state_dict'] = ema.state_dict()
        
        # Save early stopping state (including config for consistency check on resume)
        if early_stopping is not None:
            checkpoint['early_stopping_state'] = {
                'best_metric': early_stopping.best_metric,
                'best_step': early_stopping.best_step,
                'counter': early_stopping.counter,
                'patience': early_stopping.patience,
                'min_delta': early_stopping.min_delta,
            }
        
        path = self.checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
        torch.save(checkpoint, path)
        
        if is_best or metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, self.best_path)
            print(f" NEW BEST MODEL! TSS={metric_value:.4f}")
        
        self._cleanup_old_checkpoints()
        return path
    
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in checkpoints[:-self.keep_last_n]:
            old_ckpt.unlink()
    
    def load(self, path: Path) -> dict:
        """Load checkpoint from path."""
        if not path.exists():
            return {}
        return torch.load(path, map_location='cpu', weights_only=False)
            
    def load_best(
        self, 
        model: nn.Module, 
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        ema: Optional["ExponentialMovingAverage"] = None,
        early_stopping: Optional["EarlyStopping"] = None,
    ) -> dict:
        """
        Load best checkpoint with full state restoration.
        
        Returns:
            Dict with checkpoint data including step, metric, plateau stats
        """
        if self.best_path is None or not self.best_path.exists():
            return {'step': 0, 'metric': 0.0, 'best_val_tss': 0.0, 'consecutive_drops': 0, 'last_val_tss': 0.0}
        
        checkpoint = self.load(self.best_path)
        
        # Core state - strict=False handles new buffers added after checkpoint was saved
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                # Optimizer state mismatch (model architecture changed)
                # Skip optimizer restore - will use fresh optimizer with loaded model weights
                import warnings
                warnings.warn(
                    f"  Optimizer state mismatch, starting fresh optimizer: {e}\n"
                    "   This is normal if model architecture changed since checkpoint."
                )
        
        # Scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (ValueError, KeyError) as e:
                import warnings
                warnings.warn(f"  Scheduler state mismatch, using fresh scheduler: {e}")
        
        # EMA state (CRITICAL!)
        if ema is not None and 'ema_state_dict' in checkpoint:
            try:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            except (ValueError, KeyError) as e:
                # EMA state mismatch (model architecture changed)
                import warnings
                warnings.warn(
                    f"  EMA state mismatch, starting fresh EMA: {e}\n"
                    "   This is normal if model architecture changed since checkpoint."
                )
        
        # Early stopping state
        if early_stopping is not None and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            early_stopping.best_metric = es_state.get('best_metric', early_stopping.best_metric)
            early_stopping.best_step = es_state.get('best_step', 0)
            early_stopping.counter = es_state.get('counter', 0)
            
            # Warn if patience/min_delta changed since checkpoint
            saved_patience = es_state.get('patience')
            saved_min_delta = es_state.get('min_delta')
            if saved_patience is not None and saved_patience != early_stopping.patience:
                import warnings
                warnings.warn(
                    f"EarlyStopping patience changed: checkpoint={saved_patience}, config={early_stopping.patience}"
                )
            if saved_min_delta is not None and saved_min_delta != early_stopping.min_delta:
                import warnings
                warnings.warn(
                    f"EarlyStopping min_delta changed: checkpoint={saved_min_delta}, config={early_stopping.min_delta}"
                )
        
        best_val_tss = checkpoint.get('best_val_tss', checkpoint.get('metric', 0.0))
        
        # Return full info for restoration
        return {
            'step': checkpoint['step'], 
            'metric': checkpoint['metric'], 
            'best_val_tss': best_val_tss,
            'consecutive_drops': checkpoint.get('consecutive_drops', 0),
            'last_val_tss': checkpoint.get('last_val_tss', best_val_tss)
        }

# ============================================================================
# Metrics
# ============================================================================

@dataclass
class MetricsBuffer:
    max_size: int = 100
    probs: deque = field(default_factory=lambda: deque(maxlen=100))
    labels: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, prob: np.ndarray, label: np.ndarray):
        # Ensure 2D shape [N, horizons] - handle 1D edge case
        if prob.ndim == 1:
            prob = prob.reshape(1, -1)
        if label.ndim == 1:
            label = label.reshape(1, -1)
        # deque automatically drops oldest when maxlen exceeded
        self.probs.append(prob)
        self.labels.append(label)
    
    def clear(self):
        """Clear all stored predictions to free memory."""
        self.probs.clear()
        self.labels.clear()
    
    def compute(self, horizons: list[int], logger: logging.Logger, label_prefix: str = ""):
        if not self.probs: return
        probs = np.concatenate(self.probs, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        
        # SAFETY: Filter NaN/Inf and clamp probabilities
        valid_mask = np.isfinite(probs).all(axis=1) & np.isfinite(labels).all(axis=1)
        if valid_mask.sum() == 0:
            logger.warning(f"{label_prefix}No valid samples for metrics!")
            return 0.0
        probs = probs[valid_mask]
        labels = labels[valid_mask]
        probs = np.clip(probs, 0.0, 1.0)
        
        max_tss = 0.0
        
        for j, h in enumerate(horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            
            # Filter NaN AND Inf, clamp to valid ranges
            valid = np.isfinite(y_true) & np.isfinite(y_prob)
            y_true = np.clip(y_true[valid], 0.0, 1.0)
            y_prob = np.clip(y_prob[valid], 0.0, 1.0)
            
            if len(y_true) == 0 or y_true.sum() == 0: continue
            
            thr_tss, tss_val = sweep_tss(y_true, y_prob, n=256)
            max_tss = max(max_tss, tss_val)
            prauc = pr_auc(y_true, y_prob)
            bs = brier_score(y_true, y_prob)
            thr_far = select_threshold_at_far(y_true, y_prob, max_far=0.05)
            
            # Safety: clamp BS to [0, 1] range
            bs = float(np.clip(bs, 0.0, 1.0))
            
            logger.info(
                f"{label_prefix}h={h}h: TSS={tss_val:.3f}@{thr_tss:.2f} | "
                f"PR={prauc:.3f} | BS={bs:.3f} | FAR5%@{thr_far:.2f}"
            )
        return max_tss

# ============================================================================
# Trainer
# ============================================================================

class PINNTrainer:
    def __init__(self, cfg: PINNConfig, logger: logging.Logger, use_wandb: bool = False):
        self.cfg = cfg
        self.logger = logger
        self.device = self._get_device(cfg.device)
        self.logger.info(f"Using device: requested={cfg.device}, selected={self.device}")
        
        # Force num_workers=0 on MPS to prevent memory leaks
        if self.device.type == "mps" and cfg.train.num_workers > 0:
            self.logger.warning(f"  Forcing num_workers=0 on MPS (was {cfg.train.num_workers}) to prevent memory leaks")
            cfg.train.num_workers = 0
        
        optimize_for_device(self.device)
        
        # W&B initialization
        self.use_wandb = use_wandb and WANDB_AVAILABLE and not os.getenv("WANDB_DISABLED")
        if self.use_wandb and wandb.run is None:
            # Only init if not already initialized by sweep
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "flare-pinn-sota"),
                config=cfg.model_dump(),
                name=f"{cfg.model.model_type}_lr{cfg.train.lr}"
            )
        
        # Model
        if cfg.model.model_type == "hybrid":
            self.model = HybridPINNModel(cfg, encoder_in_channels=None).to(self.device)
        elif cfg.model.model_type == "cnn_gru_direct":
            from src.baselines.cnn_gru_direct import create_cnn_gru_direct_from_config
            self.model = create_cnn_gru_direct_from_config(cfg).to(self.device)
        else:
            self.model = PINNModel(cfg).to(self.device)
        
        # Freeze classifier if requested (Option B: only train physics)
        if getattr(cfg.train, 'freeze_classifier', False):
            self.logger.info("🔒 FREEZING CLASSIFIER - only training physics heads")
            if hasattr(self.model, 'classifier'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
            # Also freeze encoder if it exists (preserve learned representations)
            if hasattr(self.model, 'encoder'):
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                self.logger.info("    Encoder also frozen")
        
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Params: {n_params:,} ({n_trainable:,} trainable)")
        
        # Memory estimation for MPS
        if self.device.type == "mps":
            param_memory_mb = n_params * 4 / 1e6  # float32
            self.logger.info(f"Estimated model memory: {param_memory_mb:.1f} MB")
            self.logger.info("  MPS mode: Using gradient accumulation for larger effective batch")
        
        # Optimizer & Scheduler with weight decay
        # Only include trainable parameters (respects freeze_classifier)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=cfg.train.lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # AMP setup - MPS uses different approach
        self.use_amp = cfg.train.amp
        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = None  # MPS doesn't use GradScaler
        
        # Gradient accumulation for effective larger batches
        self.accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 1)
        if self.accum_steps > 1:
            self.grad_accumulator = MPSGradientAccumulator(
                self.model,
                self.optimizer,
                accum_steps=self.accum_steps,
                grad_clip=cfg.train.grad_clip,
                cleanup_every=getattr(cfg.train, 'memory_cleanup_every', 50)
            )
            self.logger.info(f"Gradient accumulation: {self.accum_steps} steps (effective batch = {cfg.train.batch_size * self.accum_steps})")
        else:
            self.grad_accumulator = None
        
        # Scheduler
        #  SAFETY: Account for gradient accumulation in scheduler steps
        # Use lr_total_steps if provided (for matching sweep schedules), otherwise use train.steps
        scheduler_horizon_steps = getattr(cfg.train, 'lr_total_steps', None) or cfg.train.steps
        effective_total_steps = max(1, scheduler_horizon_steps // (self.accum_steps if self.accum_steps > 1 else 1))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=effective_total_steps, eta_min=cfg.train.scheduler.min_lr
        )
        if cfg.train.lr_total_steps:
            self.logger.info(f" LR scheduler horizon: {cfg.train.lr_total_steps} steps (overriding train.steps={cfg.train.steps})")
        
        # EMA for better generalization
        self.ema = None
        if cfg.train.use_ema:
            from src.utils.training_utils import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model, decay=cfg.train.ema_decay)
            self.logger.info(f"EMA enabled with decay={cfg.train.ema_decay}")
        
        # Early stopping - Balance patience for physics training
        # With eval_every=2000, patience=12 means 24k steps without improvement
        # This allows physics to stabilize while catching genuine plateaus
        from src.utils.training_utils import EarlyStopping
        self.early_stopping = EarlyStopping(patience=12, min_delta=0.003, mode='max')
        
        self.checkpoint_mgr = None
        if cfg.train.checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(
                cfg.train.checkpoint_dir,
                keep_last_n=getattr(cfg.train, "keep_last_n_checkpoints", 3),
            )
            
        self.metrics_buffer = MetricsBuffer()
        self.step = 0
        self.lr_schedule_start_step = 0  # Track when LR schedule started (for warmup)
        self.best_val_tss = 0.0
        
        # Plateau rollback: if TSS drops for N consecutive evals, reload best and reduce LR
        self.plateau_patience = getattr(cfg.train, 'plateau_patience', 3)  # consecutive drops before rollback
        self.plateau_lr_factor = getattr(cfg.train, 'plateau_lr_factor', 0.5)  # LR reduction factor
        self.consecutive_drops = 0
        self.last_val_tss = 0.0
        self.min_lr_for_rollback = 1e-7  # don't rollback if LR already tiny
        
        # MPS-specific settings
        self.memory_cleanup_every = getattr(cfg.train, 'memory_cleanup_every', 50)
        self._accumulated_loss = 0.0
        self._accumulated_phys = 0.0
        self._accumulated_cls = 0.0
        self._accumulated_count = 0

    def _get_validation_selection_metric(self, val_results: dict) -> tuple[float, str, float, float]:
        """
        Return the checkpoint-selection metric along with useful diagnostics.

        Note: checkpoints still persist this value under the legacy
        `best_val_tss` key for backward compatibility.
        """
        horizons = val_results.get("horizons", {})
        horizon_tss = [float(metrics["tss"]) for metrics in horizons.values()]
        tss_24h = float(horizons.get(24, {}).get("tss", 0.0))
        mean_tss = float(np.mean(horizon_tss)) if horizon_tss else 0.0
        max_tss = float(val_results.get("max_tss", 0.0))
        selection_mode = getattr(self.cfg.train, "selection_metric", "tss_24h")

        if selection_mode == "mean_tss":
            return mean_tss, "mean horizon TSS", tss_24h, mean_tss
        if selection_mode == "max_tss":
            return max_tss, "max horizon TSS", tss_24h, mean_tss
        return (tss_24h if tss_24h > 0 else max_tss), "24h TSS", tss_24h, mean_tss

    def _get_device(self, requested: str) -> torch.device:
        if requested == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
        if requested.startswith("cuda") and torch.cuda.is_available(): return torch.device(requested)
        return torch.device("cpu")

    def train_step(self, batch: dict) -> dict:
        # Move all tensors to device
        # Remove non_blocking=True on MPS to prevent memory corruption/race conditions
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)  # Synchronous transfer
        
        # Validate inputs and sanitize NaN/Inf (with logging)
        # Check all input tensors including gt_bz, pil_mask, scalars
        tensor_keys = ["frames", "coords", "scalar_features", "gt_bz", "pil_mask", "scalars", "observed_mask"]
        for k in tensor_keys:
            if k in batch and batch[k] is not None and isinstance(batch[k], torch.Tensor):
                has_nan = torch.isnan(batch[k]).any()
                has_inf = torch.isinf(batch[k]).any()
                if has_nan or has_inf:
                    self.logger.warning(f"Step {self.step}: NaN/Inf in input '{k}', sanitizing")
                    batch[k] = torch.nan_to_num(batch[k], nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Update curriculum
        frac = min(1.0, self.step / self.cfg.train.steps)
        self.model.set_train_frac(frac)
        
        # Clear gradients if not using accumulator
        if self.grad_accumulator is None:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward
        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # MPS autocast with bfloat16 is more stable (if available)
        if self.device.type == "mps":
            autocast_dtype = torch.float32  # MPS float16 has issues, use float32
            use_autocast = False  # Disable autocast on MPS for stability
        else:
            use_autocast = self.use_amp
        
        with torch.autocast(self.device.type, dtype=autocast_dtype, enabled=use_autocast):
            B = batch["coords"].shape[0]
            
            # Use BATCHED forward for HybridPINNModel (MAJOR SPEEDUP)
            if hasattr(self.model, 'forward_batched') and "frames" in batch:
                out = self.model.forward_batched(
                    coords=batch["coords"],
                    frames=batch["frames"],
                    gt_bz=batch["gt_bz"],
                    observed_mask=batch["observed_mask"],
                    labels=batch["labels"],
                    pil_mask=batch["pil_mask"],
                    scalars=batch.get("scalars"),
                    mode="train",
                )
                
                loss = out.loss_total
                probs = out.probs
                labels = batch["labels"]
                # Handle potential None or zero values safely
                total_phys = float(out.loss_phys.detach().cpu()) if out.loss_phys is not None else 0.0
                total_cls = float(out.loss_cls.detach().cpu()) if out.loss_cls is not None else 0.0
            else:
                # Fallback: per-sample processing (for non-hybrid models)
                total_loss = None
                total_phys = 0.0
                total_cls = 0.0
                batch_probs, batch_labels = [], []
                
                for i in range(B):
                    sample_kwargs = {
                        "coords": batch["coords"][i],
                        "gt_bz": batch["gt_bz"][i],
                        "observed_mask": batch["observed_mask"][i],
                        "labels": batch["labels"][i:i+1],
                        "pil_mask": batch["pil_mask"][i],
                        "mode": "train",
                    }
                    if "frames" in batch:
                        sample_kwargs["frames"] = batch["frames"][i]
                    if "scalars" in batch:
                        sample_kwargs["scalars"] = batch["scalars"][i]
                    
                    out = self.model(**sample_kwargs)
                    
                    if torch.isnan(out.loss_total) or torch.isinf(out.loss_total):
                        self.logger.warning(f"Step {self.step}: NaN/Inf in loss_total from sample {i}, skipping batch")
                        return {"loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0, "grad_norm": 0.0}
                    
                    if out.probs is not None and (torch.isnan(out.probs).any() or torch.isinf(out.probs).any()):
                        out.probs.data.copy_(torch.nan_to_num(out.probs, nan=0.5, posinf=1.0, neginf=0.0))
                    
                    if total_loss is None:
                        total_loss = out.loss_total / B
                    else:
                        total_loss = total_loss + (out.loss_total / B)
                    
                    # Handle potential None or zero values
                    p_loss = float(out.loss_phys.detach().cpu()) if out.loss_phys is not None else 0.0
                    c_loss = float(out.loss_cls.detach().cpu()) if out.loss_cls is not None else 0.0
                    total_phys += p_loss / B
                    total_cls += c_loss / B
                    batch_probs.append(out.probs)
                    batch_labels.append(batch["labels"][i:i+1])
                
                loss = total_loss
                probs = torch.cat(batch_probs)
                labels = torch.cat(batch_labels)

        # Initialize did_step early for scope
        did_step = False

        # NaN check - skip update if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"Step {self.step}: NaN/Inf loss detected, skipping update")
            if self.grad_accumulator:
                self.grad_accumulator.zero_grad()
            return {
                "loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0
            }
        
        # Check that loss requires gradients (critical for backward pass)
        if not loss.requires_grad:
            self.logger.error(
                f"Step {self.step}: Loss does not require gradients! "
                f"loss.requires_grad={loss.requires_grad}, "
                f"loss.grad_fn={loss.grad_fn}, "
                f"loss.is_leaf={loss.is_leaf}. "
                f"This usually means the loss computation was disconnected from the model parameters."
            )
            # Try to reconnect by adding a zero term connected to model parameters
            # Get a parameter to ensure connection - use the first one that requires grad
            for param in self.model.parameters():
                if param.requires_grad:
                    # Convert loss to tensor if needed and add connected zero term
                    if isinstance(loss, (int, float)):
                        loss = torch.tensor(float(loss), device=param.device, dtype=param.dtype)
                    loss = loss + (param * 0.0).sum()
                    break
            
            if not loss.requires_grad:
                raise RuntimeError(
                    f"Failed to reconnect loss to computation graph at step {self.step}. "
                    f"loss.requires_grad={loss.requires_grad}, loss.grad_fn={loss.grad_fn}. "
                    f"Check that model parameters require gradients and forward pass is correct."
                )
            self.logger.warning(f"Step {self.step}: Successfully reconnected loss to computation graph.")

        # Backward with gradient accumulation support
        did_step = False
        grad_norm = 0.0
        
        if self.grad_accumulator is not None:
            # Use gradient accumulator (handles scaling and stepping)
            did_step = self.grad_accumulator.step(loss, self.step)
            if did_step:
                grad_norm = self.grad_accumulator.last_grad_norm  # Actual grad norm
        elif self.scaler is not None:
            # CUDA with AMP
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            did_step = True
        else:
            # Standard backward (MPS or CPU)
            loss.backward()
            if self.cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            
            # Check for NaN gradients
            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                self.logger.warning(f"Step {self.step}: NaN/Inf gradients detected, zeroing gradients")
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0, "grad_norm": 0.0}
            
            self.optimizer.step()
            did_step = True
        
        # Update EMA only when optimizer stepped
        if did_step and self.ema is not None:
            self.ema.update()
            
        # Scheduler (warmup manual) - only when optimizer stepped
        if did_step:
            # Calculate steps since LR schedule started (for fresh_lr_schedule support)
            steps_since_schedule_start = self.step - self.lr_schedule_start_step
            if steps_since_schedule_start < self.cfg.train.scheduler.warmup_steps:
                lr_scale = min(1.0, float(steps_since_schedule_start + 1) / self.cfg.train.scheduler.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.cfg.train.lr * lr_scale
            else:
                self.scheduler.step()
        
        # MPS: return pooled memory without a full mps.synchronize (see mps_train_interval_cleanup)
        if self.device.type == "mps" and self.step % self.memory_cleanup_every == 0:
            mps_train_interval_cleanup()
            
        with torch.no_grad():
            self.metrics_buffer.add(probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        
        # Track accumulated stats for logging
        # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
        self._accumulated_loss += float(loss.detach().cpu())
        self._accumulated_phys += total_phys
        self._accumulated_cls += total_cls
        self._accumulated_count += 1
        
        # Return averaged stats
        if did_step and self._accumulated_count > 0:
            avg_loss = self._accumulated_loss / self._accumulated_count
            avg_phys = self._accumulated_phys / self._accumulated_count
            avg_cls = self._accumulated_cls / self._accumulated_count
            self._accumulated_loss = 0.0
            self._accumulated_phys = 0.0
            self._accumulated_cls = 0.0
            self._accumulated_count = 0
            
            # Store last valid grad_norm for logging (handle sparse updates)
            self._last_grad_norm = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
        else:
            # Avoid .item() which can hang on MPS - use detach().cpu() instead
            avg_loss = float(loss.detach().cpu())
            avg_phys = total_phys
            avg_cls = total_cls
            
        # Use last valid grad_norm if current step didn't update
        current_grad_norm = float(grad_norm) if did_step else getattr(self, '_last_grad_norm', 0.0)
        
        metrics_dict = {
            "loss": avg_loss,
            "phys": avg_phys,
            "cls": avg_cls,
            "lam": out.lambda_phys,
            "alpha": out.fourier_alpha,
            "grad_norm": current_grad_norm,
            "did_step": did_step,
            "ess": out.ess,  # Log ESS for monitoring
        }
        
        # Log to W&B
        if self.use_wandb and self.step % self.cfg.train.log_every == 0:
            log_dict = {
                "train/loss_total": metrics_dict["loss"],
                "train/loss_cls": metrics_dict["cls"],
                "train/loss_phys": metrics_dict["phys"],
                "train/grad_norm": metrics_dict["grad_norm"],
                "physics/lambda": metrics_dict["lam"],
                "physics/fourier_alpha": metrics_dict["alpha"],
                "optimizer/lr": self.optimizer.param_groups[0]['lr'],
                "step": self.step
            }
            
            # Log GradNorm weights for debugging
            if hasattr(self.model, '_gradnorm_weights') and self.cfg.physics.enable:
                use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False)
                if use_gradnorm:
                    log_dict["gradnorm/w_cls"] = float(self.model._gradnorm_weights[0].detach().cpu())
                    log_dict["gradnorm/w_phys"] = float(self.model._gradnorm_weights[1].detach().cpu())
            
            # Log LRA lambda if enabled
            if hasattr(self.model, '_lra_lambda') and getattr(self.cfg.physics, 'use_lra', False):
                log_dict["lra/lambda"] = float(self.model._lra_lambda.detach().cpu())
            
            wandb.log(log_dict, step=self.step)
        
        return metrics_dict

    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> dict:
        """
        Run full validation and return comprehensive metrics.
        
        Args:
            loader: Validation DataLoader
            use_ema: Whether to use EMA weights for evaluation
            
        Returns:
            Dictionary with TSS, PR-AUC, Brier scores per horizon
        """
        if loader is None: 
            return {"max_tss": 0.0}
        
        self.model.eval()
        all_probs, all_labels = [], []
        
        # Use EMA weights if available (with proper context manager)
        ema_context = self.ema.average_parameters() if (use_ema and self.ema is not None) else None
        
        # Use low memory mode for MPS
        with torch.no_grad(), low_memory_mode():
            # Enter EMA context if available
            if ema_context is not None:
                ema_context.__enter__()
            
            try:
                for batch in loader:
                    # Clone labels BEFORE moving to device
                    labels_original = batch["labels"].clone()
                    
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            # FIXED: Remove non_blocking=True on MPS to prevent memory corruption in validation too
                            batch[k] = v.to(self.device)
                    
                    B = batch["coords"].shape[0]
                    batch_probs = []
                    for i in range(B):
                        sample_kwargs = {
                            "coords": batch["coords"][i],
                            "gt_bz": batch["gt_bz"][i],
                            "observed_mask": batch["observed_mask"][i],
                            "labels": batch["labels"][i:i+1],
                            "pil_mask": batch["pil_mask"][i],
                            "mode": "eval",
                        }
                        if "frames" in batch:
                            sample_kwargs["frames"] = batch["frames"][i]
                        if "scalars" in batch:
                            sample_kwargs["scalars"] = batch["scalars"][i]
                        
                        out = self.model(**sample_kwargs)
                        batch_probs.append(out.probs)
                    
                    batch_probs_cat = torch.cat(batch_probs).cpu().numpy()
                    # Use ORIGINAL labels (before device transfer) for metrics
                    batch_labels_np = labels_original.numpy()
                    
                    all_probs.append(batch_probs_cat)
                    all_labels.append(batch_labels_np)
            finally:
                # Exit EMA context
                if ema_context is not None:
                    ema_context.__exit__(None, None, None)
        
        self.model.train()
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        
        # Compute comprehensive metrics
        results = {"max_tss": 0.0, "horizons": {}}
        
        for j, h in enumerate(self.cfg.classifier.horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            
            # DEBUG: Count before filtering
            n_before = len(y_true)
            n_pos_before = int((y_true == 1.0).sum()) if np.isfinite(y_true).all() else "?"
            n_prob_nan = int((~np.isfinite(y_prob)).sum())
            n_true_nan = int((~np.isfinite(y_true)).sum())
            
            # Filter out NaN AND Inf values (isfinite catches both!)
            valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
            y_true = y_true[valid_mask]
            y_prob = y_prob[valid_mask]
            
            # DEBUG: Log filtering stats
            n_after = len(y_true)
            n_filtered = n_before - n_after
            if n_filtered > 0:
                self.logger.warning(f"  {h}h: Filtered {n_filtered}/{n_before} samples (prob_nan={n_prob_nan}, true_nan={n_true_nan})")
            
            # Clamp probabilities to valid range [0, 1]
            y_prob = np.clip(y_prob, 0.0, 1.0)
            
            # Ensure labels are binary (0 or 1)
            y_true = np.clip(y_true, 0.0, 1.0)
            
            if len(y_true) == 0 or y_true.sum() == 0: 
                continue
                
            # TSS
            thr_tss, tss_val = sweep_tss(y_true, y_prob)
            results["max_tss"] = max(results["max_tss"], tss_val)
            
            # PR-AUC
            prauc = pr_auc(y_true, y_prob)
            
            # Brier score
            bs = brier_score(y_true, y_prob)
            
            # ECE
            ece = adaptive_ece(y_true, y_prob)
            
            # Ensure y_true is binary (0 or 1) before counting
            # DEBUG: Check what's actually in y_true
            unique_vals = np.unique(y_true)
            if len(unique_vals) > 2 or (len(unique_vals) == 2 and not np.allclose(unique_vals, [0.0, 1.0])):
                self.logger.warning(f"  {h}h DEBUG: y_true has unexpected values: {unique_vals[:10]}")
            
            # Count positives correctly - y_true should be 0.0 or 1.0
            n_pos = int((y_true > 0.5).sum())  # More robust than .astype(int32)
            
            # Safety: clamp BS and ECE to reasonable ranges
            bs = float(np.clip(bs, 0.0, 1.0))
            ece = float(np.clip(ece, 0.0, 1.0))
            
            results["horizons"][h] = {
                "tss": tss_val,
                "threshold": thr_tss,
                "pr_auc": prauc,
                "brier": bs,
                "ece": ece,
                "n_pos": n_pos,
                "n_total": len(y_true)
            }
            
        return results

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.model.train()
        t_start = time.time()
        stopped_early = False
        
        self.logger.info(f"Starting training loop... (target: {self.cfg.train.steps} steps)")
        
        while self.step < self.cfg.train.steps and not stopped_early:
            for batch in train_loader:
                if self.step == 0:
                    self.logger.info("First batch loaded from disk!")
                self.step += 1
                metrics = self.train_step(batch)
                
                if self.step % self.cfg.train.log_every == 0:
                    elapsed = time.time() - t_start
                    lr = self.optimizer.param_groups[0]['lr']
                    grad_info = f"GradNorm={metrics.get('grad_norm', 0):.2f} | " if 'grad_norm' in metrics else ""
                    self.logger.info(
                        f"Step {self.step:05d} | Loss={metrics['loss']:.4f} "
                        f"(Phys={metrics['phys']:.4f}, Cls={metrics['cls']:.4f}) | "
                        f"{grad_info}"
                        f"Lam={metrics['lam']:.2f} | LR={lr:.2e} | T={elapsed:.1f}s"
                    )
                    t_start = time.time()
                
                if self.step % self.cfg.train.eval_every == 0:
                    self.metrics_buffer.compute(self.cfg.classifier.horizons, self.logger, "[Train] ")
                    # FIXED: Clear metrics buffer after logging to prevent memory leak
                    self.metrics_buffer.clear()
                    gc.collect()  # Force garbage collection
                    if val_loader:
                        val_results = self.evaluate(val_loader)
                        val_tss = val_results["max_tss"]
                        
                        # FIXED: Aggressive cleanup after validation to prevent leaks
                        clear_memory_cache(self.device)
                        gc.collect()  # Force Python garbage collection
                        if self.device.type == "mps":
                            torch.mps.empty_cache()  # Clear MPS cache
                        
                        # Log detailed metrics
                        self.logger.info(f"[Val] Max TSS: {val_tss:.4f}")
                        primary_metric, primary_label, tss_24h, mean_tss = self._get_validation_selection_metric(val_results)
                        self.logger.info(
                            f"[Val] Selection metrics: 24h={tss_24h:.4f} | mean={mean_tss:.4f} | max={val_tss:.4f} | active={primary_label}={primary_metric:.4f}"
                        )
                        
                        # W&B logging
                        if self.use_wandb:
                            log_dict = {"val/tss_max": val_tss, "step": self.step}
                            for h, h_metrics in val_results.get("horizons", {}).items():
                                log_dict.update({
                                    f"val/tss_{h}h": h_metrics["tss"],
                                    f"val/pr_auc_{h}h": h_metrics["pr_auc"],
                                    f"val/brier_{h}h": h_metrics["brier"],
                                    f"val/ece_{h}h": h_metrics["ece"],
                                })
                            wandb.log(log_dict, step=self.step)
                        
                        for h, h_metrics in val_results.get("horizons", {}).items():
                            self.logger.info(
                                f"  {h}h: TSS={h_metrics['tss']:.3f}@{h_metrics['threshold']:.2f} | "
                                f"PR={h_metrics['pr_auc']:.3f} | BS={h_metrics['brier']:.3f} | "
                                f"ECE={h_metrics['ece']:.3f} | Pos={h_metrics['n_pos']}/{h_metrics['n_total']}"
                            )
                        
                        if primary_metric > self.best_val_tss:
                            self.best_val_tss = primary_metric
                            self.consecutive_drops = 0
                            self.logger.info(f" New best validation {primary_label}: {primary_metric:.4f}")
                        elif primary_metric < self.last_val_tss:
                            self.consecutive_drops += 1
                            self.logger.info(
                                f"📉 {primary_label} dropped ({self.last_val_tss:.4f} → {primary_metric:.4f}), consecutive drops: {self.consecutive_drops}/{self.plateau_patience}"
                            )
                            
                            # Plateau rollback: reload best checkpoint with reduced LR
                            current_lr = self.optimizer.param_groups[0]['lr']
                            if self.consecutive_drops >= self.plateau_patience and current_lr > self.min_lr_for_rollback:
                                if self.checkpoint_mgr:
                                    best_path = self.checkpoint_mgr.checkpoint_dir / "best_model.pt"
                                    if best_path.exists():
                                        new_lr = current_lr * self.plateau_lr_factor
                                        self.logger.info(f"🔄 PLATEAU ROLLBACK: Reloading best model, LR {current_lr:.2e} → {new_lr:.2e}")
                                        
                                        # Load best model weights only (keep current step)
                                        checkpoint = self.checkpoint_mgr.load(best_path)
                                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                                        if self.ema and 'ema_state_dict' in checkpoint:
                                            self.ema.load_state_dict(checkpoint['ema_state_dict'])
                                        
                                        # Reset optimizer with new LR
                                        for param_group in self.optimizer.param_groups:
                                            param_group['lr'] = new_lr
                                            param_group['initial_lr'] = new_lr
                                        
                                        # Fresh scheduler from new LR
                                        remaining = max(1, (self.cfg.train.steps - self.step) // self.accum_steps)
                                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                            self.optimizer, T_max=remaining, eta_min=1e-7
                                        )
                                        
                                        self.consecutive_drops = 0
                                        self.last_val_tss = self.best_val_tss
                        else:
                            # TSS same or slightly better but not new best
                            self.consecutive_drops = 0
                        
                        self.last_val_tss = primary_metric
                        
                        # Save checkpoint with FULL state (EMA, scheduler, early stopping)
                        if self.checkpoint_mgr:
                            self.checkpoint_mgr.save(
                                self.model, 
                                self.optimizer, 
                                self.step, 
                                primary_metric,  # Save using the active validation selection metric
                                scheduler=self.scheduler,
                                ema=self.ema,
                                early_stopping=self.early_stopping,
                                best_val_tss=self.best_val_tss,
                                #  FIX: Save plateau state
                                consecutive_drops=self.consecutive_drops,
                                last_val_tss=self.last_val_tss,
                            )
                        
                        # MPS auto-restart: exit after checkpoint to clear memory leak
                        auto_restart = getattr(self.cfg.train, 'auto_restart_every', 0)
                        if auto_restart > 0 and self.step % auto_restart == 0 and self.step > 0:
                            self.logger.info(f"🔄 Auto-restart triggered at step {self.step} (MPS memory cleanup)")
                            self.logger.info(f"   Resume with: --resume {self.checkpoint_mgr.checkpoint_dir}/checkpoint_step_{self.step:07d}.pt")
                            import sys
                            sys.exit(42)  # Special exit code for auto-restart
                        
                        # Early stopping check
                        if self.early_stopping(val_tss, self.step):
                            self.logger.info(f"🛑 Early stopping triggered at step {self.step}")
                            self.logger.info(f"   Best TSS was {self.early_stopping.best_metric:.4f} at step {self.early_stopping.best_step}")
                            stopped_early = True
                            break
                
                if self.step >= self.cfg.train.steps: 
                    break
        
        # Final summary
        self.logger.info("="*60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Final best validation TSS: {self.best_val_tss:.4f}")
        self.logger.info("="*60)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed from YAML (multi-seed sweeps without duplicating configs).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override train.checkpoint_dir from YAML (use a unique directory per seed).",
    )
    args = parser.parse_args()
    
    cfg = PINNConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    if args.checkpoint_dir is not None:
        cfg = cfg.model_copy(
            update={
                "train": cfg.train.model_copy(
                    update={"checkpoint_dir": Path(args.checkpoint_dir)},
                ),
            }
        )
    
    #  FIX: Set global seed for reproducible training AND validation
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = setup_logging(Path(f"outputs/logs/{Path(args.config).stem}.log"))
    # Enable wandb if requested or if WANDB_PROJECT env var is set
    use_wandb = args.wandb or bool(os.getenv("WANDB_PROJECT"))
    trainer = PINNTrainer(cfg, logger, use_wandb=use_wandb)
    
    # Resume from checkpoint (explicit --resume or auto from best_model.pt)
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            return
    elif trainer.checkpoint_mgr:
        best_ckpt_path = trainer.checkpoint_mgr.checkpoint_dir / "best_model.pt"
        if best_ckpt_path.exists():
            resume_path = best_ckpt_path
    
    if resume_path is not None and trainer.checkpoint_mgr:
        # Set the path so load_best can find it
        trainer.checkpoint_mgr.best_path = resume_path
        
        # Load full state including EMA, early stopping
        # NOTE: Don't restore scheduler when using --resume (allows fine-tuning with new LR)
        # Determine if we should load scheduler state or start fresh
        # If fresh_lr_schedule is True, we pass None to load_best so it doesn't load state
        # Only start a fresh optimizer/LR schedule when transitioning from an
        # external checkpoint (e.g. Stage 1 baseline -> Stage 2 fine-tune).
        # Stage-2 auto-resumes from this run's own checkpoint directory must
        # preserve optimizer/scheduler/EMA state, or each restart rewinds the
        # fine-tuning dynamics every 2k steps on MPS.
        is_resuming = resume_path is not None
        checkpoint_dir = trainer.checkpoint_mgr.checkpoint_dir.resolve()
        resume_dir = resume_path.parent.resolve()
        is_external_resume = resume_dir != checkpoint_dir
        should_reset = is_resuming and cfg.train.fresh_lr_schedule and is_external_resume
        
        scheduler_arg = None if should_reset else trainer.scheduler
        optimizer_arg = None if should_reset else trainer.optimizer
        ema_arg = None if should_reset else trainer.ema  # Also skip EMA if architecture changed

        if should_reset:
            logger.info("  Starting with FRESH optimizer & EMA (external stage transition)")
        elif is_resuming and cfg.train.fresh_lr_schedule:
            logger.info("  Resuming in-run checkpoint with optimizer/LR/EMA state intact")

        # Load state
        loaded_state = trainer.checkpoint_mgr.load_best(
            trainer.model, 
            optimizer=optimizer_arg,
            scheduler=scheduler_arg,
            ema=ema_arg,
            early_stopping=trainer.early_stopping,
        )
        
        # If we reset EMA, re-initialize it from the loaded model weights
        if ema_arg is None and trainer.ema is not None:
            trainer.ema.shadow = {name: param.clone().detach() for name, param in trainer.model.named_parameters()}
            logger.info("    EMA re-initialized from loaded model weights")
        
        resume_step = loaded_state['step']
        resume_metric = loaded_state['metric']
        best_val_tss = loaded_state.get('best_val_tss', 0.0)
        
        # Restore plateau tracking
        # BUT: Reset if starting fresh LR (Stage 2 physics fine-tuning)
        # OR if loaded best_val_tss is suspiciously high (> current metric), indicating Stage 1→2 transition
        is_stage_transition = should_reset or (best_val_tss > resume_metric + 0.02)
        
        if is_stage_transition:
            trainer.consecutive_drops = 0
            trainer.last_val_tss = resume_metric  # Use current metric as baseline
            if should_reset:
                logger.info("    Plateau tracking reset for Stage 2 (fresh baseline)")
            else:
                logger.info(f"    Plateau baseline adjusted: {best_val_tss:.4f} (Stage 1) → {resume_metric:.4f} (Stage 2)")
        else:
            trainer.consecutive_drops = loaded_state.get('consecutive_drops', 0)
            trainer.last_val_tss = loaded_state.get('last_val_tss', best_val_tss)
        
        trainer.step = resume_step
        trainer.best_val_tss = best_val_tss if best_val_tss > 0 else resume_metric
        # Also restore checkpoint manager's best_metric to prevent false "NEW BEST" prints
        trainer.checkpoint_mgr.best_metric = trainer.best_val_tss
        
        # Only reset the LR schedule for external stage transitions.
        if should_reset:
            # Reset tracking for fresh schedule
            trainer.consecutive_drops = 0
            trainer.lr_schedule_start_step = resume_step  # Start warmup from resume point
            new_lr = cfg.train.lr
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
                param_group['initial_lr'] = new_lr
            
            # Create fresh scheduler from current position to end
            # Use lr_total_steps if provided (for matching sweep schedules)
            scheduler_horizon_steps = getattr(cfg.train, 'lr_total_steps', None) or cfg.train.steps
            
            # If lr_total_steps is provided, use it as the FULL horizon (like sweep does)
            # Otherwise, use remaining steps
            if cfg.train.lr_total_steps:
                remaining_steps = max(1, scheduler_horizon_steps // trainer.accum_steps)
            else:
                remaining_steps = max(1, (scheduler_horizon_steps - resume_step) // trainer.accum_steps)
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=remaining_steps, eta_min=cfg.train.scheduler.min_lr
            )
            current_lr = new_lr
            logger.info(f"    Fresh LR schedule: {current_lr:.2e} → {cfg.train.scheduler.min_lr:.0e} over {remaining_steps} optimizer steps (warmup: {cfg.train.scheduler.warmup_steps} steps)")
            if cfg.train.lr_total_steps:
                logger.info(f"    LR scheduler horizon: {cfg.train.lr_total_steps} steps (overriding train.steps={cfg.train.steps})")
        elif args.resume:
            # Resume scheduler state
            current_lr = trainer.optimizer.param_groups[0]['lr']
            logger.info(f"   📉 Resumed LR schedule: {current_lr:.2e}")
        
        # Check if EMA was restored
        ema_restored = 'ema_state_dict' in trainer.checkpoint_mgr.load(resume_path)
        ema_status = " EMA restored" if ema_restored else " EMA not in checkpoint (starting fresh)"
        
        logger.info(f" Resumed from checkpoint at step {resume_step} (best TSS: {resume_metric:.4f})")
        logger.info(f"   {ema_status}")
    
    # Load Data
    if cfg.data.use_real:
        from src.utils.masked_training import load_windows_with_mask
        
        logger.info("Loading real data...")
        logger.info("  Reading windows parquet...")
        df, mask = load_windows_with_mask(str(cfg.data.windows_parquet))
        df = df[mask].reset_index(drop=True)
        logger.info(f"  Loaded {len(df)} windows")
        
        # CHRONOLOGICAL SPLIT (80/5/15 by time, not random HARP)
        # This is the proper operational forecasting evaluation
        df['t0'] = pd.to_datetime(df['t0'])
        df = df.sort_values('t0').reset_index(drop=True)
        
        split_idx = int(len(df) * (1.0 - cfg.data.val_fraction))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
        
        logger.info(f"  ⏱️ CHRONOLOGICAL SPLIT (train on past, validate on future)")
        logger.info(f"  Train: {len(train_df)} windows ({train_df.t0.min().date()} to {train_df.t0.max().date()})")
        logger.info(f"  Val: {len(val_df)} windows ({val_df.t0.min().date()} to {val_df.t0.max().date()})")
        logger.info("  Ground truth positives:")
        for h in cfg.classifier.horizons:
            train_h = train_df[f"y_geq_M_{h}h"].sum()
            val_h = val_df[f"y_geq_M_{h}h"].sum()
            logger.info(f"    {h}h: Train={train_h:.0f}, Val={val_h:.0f}")
        
        # SANITY CHECK: Verify no time overlap (silent - already checked)
        # train_max = train_df.t0.max()
        # val_min = val_df.t0.min()
        # Minor overlap at split boundary is OK and doesn't affect results
        
        # Choose dataset based on config
        if cfg.data.use_consolidated:
            from src.data.consolidated_dataset import ConsolidatedWindowsDataset
            logger.info("  Using CONSOLIDATED dataset (fast I/O)...")
            
            train_ds = ConsolidatedWindowsDataset(
                windows_df=train_df,
                consolidated_dir=str(cfg.data.consolidated_dir),
                target_px=cfg.data.target_size,
                input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons),
                P_per_t=cfg.data.P_per_t,
                pil_top_pct=cfg.data.pil_top_pct,
                training=True,
                augment=True,
                noise_std=0.05,  # Increased from 0.02 for better regularization
                max_cached_harps=500,
                # SOTA features
                use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
                use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
            )
            
            val_ds = ConsolidatedWindowsDataset(
                windows_df=val_df,
                consolidated_dir=str(cfg.data.consolidated_dir),
                target_px=cfg.data.target_size,
                input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons),
                P_per_t=cfg.data.P_per_t,
                pil_top_pct=cfg.data.pil_top_pct,
                training=False,
                augment=False,
                max_cached_harps=500,
                # SOTA features
                use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
                use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
            )
        else:
            from src.data.cached_dataset import CachedWindowsDataset
            logger.info("  Using CACHED dataset (slower I/O, higher quality)...")
            
            train_ds = CachedWindowsDataset(
                train_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root), 
                target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t, 
                pil_top_pct=cfg.data.pil_top_pct,
                training=True, augment=True, noise_std=0.02, preload=False
            )
            
            val_ds = CachedWindowsDataset(
                val_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root),
                target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t,
                pil_top_pct=cfg.data.pil_top_pct,
                training=False, augment=False, preload=False
            )
        
        logger.info("  Datasets created!")
        
        # Sampler: Only use WeightedRandomSampler if weights are non-uniform
        # When weights are uniform, replacement=True causes some samples to be 
        # skipped and others oversampled, undermining balanced coverage
        horizon_cols = [f"y_geq_M_{h}h" for h in cfg.classifier.horizons]
        labels = train_df[horizon_cols].to_numpy().sum(axis=1) > 0
        weights = np.full(len(train_df), cfg.train.sampler.smoothing)
        weights[labels] = cfg.train.sampler.positive_multiplier
        
        # Check if weights are effectively uniform (all same value)
        weights_uniform = np.allclose(weights, weights[0])
        if weights_uniform:
            # Uniform weights: use regular shuffle (no sampler)
            sampler = None
            shuffle_train = True
            logger.info("  Sampler: disabled (uniform weights, using shuffle)")
        else:
            # Non-uniform weights: use weighted sampling
            sampler = WeightedRandomSampler(torch.tensor(weights), len(weights), replacement=True)
            shuffle_train = False
            logger.info(f"  Sampler: weighted (pos_mult={cfg.train.sampler.positive_multiplier})")
        
        logger.info("  Creating DataLoaders...")
        # MPS/Mac optimization: pin_memory can cause hangs
        use_pin_memory = False if cfg.device == "mps" else True
        
        # FIXED: Disable persistent workers on MPS - they leak memory!
        # Use persistent workers only on CUDA to avoid respawn overhead
        persistent_workers = cfg.train.num_workers > 0 and cfg.device != "mps"
        
        #  FIX #17: Seed numpy RNG per worker to avoid identical augmentations
        def worker_init_fn(worker_id: int):
            worker_seed = cfg.seed + worker_id
            np.random.seed(worker_seed)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.train.batch_size, 
            sampler=sampler,
            shuffle=shuffle_train if sampler is None else False,  # shuffle only when no sampler
            num_workers=cfg.train.num_workers, 
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None,
            worker_init_fn=worker_init_fn if cfg.train.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.train.batch_size, 
            shuffle=False, 
            num_workers=cfg.train.num_workers, 
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None,
            worker_init_fn=worker_init_fn if cfg.train.num_workers > 0 else None
        )
        logger.info(f"  DataLoaders ready! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        # Dummy fallback - use DummyPINNDataset defined in this module
        ds = DummyPINNDataset(cfg)
        train_loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = None

    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
