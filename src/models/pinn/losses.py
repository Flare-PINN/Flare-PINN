# src/models/pinn/losses.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

def interp_schedule(schedule: list[list[float]], frac: float) -> float:
    """
    schedule = [[x0,y0], [x1,y1], ..., [1.0,y_end]]; piecewise-linear.
    """
    pts = sorted(schedule, key=lambda p: p[0])
    if frac <= pts[0][0]: return float(pts[0][1])
    for i in range(1, len(pts)):
        x0,y0 = pts[i-1]; x1,y1 = pts[i]
        if frac <= x1:
            t = (frac - x0) / max(1e-8, (x1 - x0))
            return float(y0*(1-t) + y1*t)
    return float(pts[-1][1])

def bce_logits(y_hat: torch.Tensor, y: torch.Tensor, pos_weight: Optional[float] = None) -> torch.Tensor:
    """
    Binary cross-entropy with logits. 
    pos_weight: weight for positive class (typically N_neg/N_pos, e.g. 5-20 for solar flares)
    """
    # Safety checks
    if torch.isnan(y).any() or torch.isinf(y).any():
        return torch.zeros(1, device=y.device, requires_grad=True).squeeze()
    
    if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Clamp logits and labels for stability
    y_hat = y_hat.clamp(-100.0, 100.0)
    y = y.clamp(0.0, 1.0)
    
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=y_hat.device, dtype=y_hat.dtype)
    loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pw)
    
    # Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.zeros(1, device=y_hat.device, requires_grad=True).squeeze()
    
    return loss

def focal_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss (Lin et al. 2017) for class imbalance.
    Downweights easy examples, focuses on hard negatives/positives.
    
    Args:
        logits: raw logits [N, C] (NOT probabilities!)
        y: targets {0,1} [N, C]
        alpha: weight for positive class (0.25 = emphasize positives)
        gamma: focusing parameter (2.0 standard, higher = more focus on hard examples)
    
    Common in solar flare prediction papers for rare event detection.
    
    Note: Uses numerically stable implementation with logsigmoid.
    """
    # Safety checks for numerical stability
    if torch.isnan(y).any() or torch.isinf(y).any():
        # Invalid labels - return zero loss (NOT using broken logits)
        return torch.zeros(1, device=y.device, requires_grad=True).squeeze()
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        # Invalid logits - clamp to valid range first
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Clamp logits to prevent overflow in sigmoid/exp
    logits = logits.clamp(-100.0, 100.0)
    
    # Ensure labels are in valid range [0, 1]
    y = y.clamp(0.0, 1.0)
    
    # Use numerically stable BCE with logits
    bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    
    # Check if BCE loss is valid
    if torch.isnan(bce_loss).any() or torch.isinf(bce_loss).any():
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    
    # Compute probabilities for focal weight (clamped for stability)
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    
    # p_t = p if y=1, else 1-p
    p_t = probs * y + (1 - probs) * (1 - y)
    
    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    
    # focal weight: (1 - p_t)^gamma (clamped to avoid extreme values)
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    
    loss = alpha_t * focal_weight * bce_loss
    
    # Final safety check
    final_loss = loss.mean()
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    
    return final_loss

def l1_data(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # FIXED: Handle NaN/Inf in inputs
    if not torch.isfinite(pred).all():
        pred = torch.nan_to_num(pred, nan=0.0, posinf=3.0, neginf=-3.0)
    if not torch.isfinite(target).all():
        target = torch.nan_to_num(target, nan=0.0, posinf=3.0, neginf=-3.0)
    
    if mask is not None:
        w = mask.float()
        valid_count = w.sum()
        
        # Return connected zero if no valid data (all masked)
        if valid_count < 0.5:
            return (pred * 0).sum()  # Connected to computation graph
        
        # FIXED: Use where() to avoid 0 * Inf = NaN issues
        diff = (pred - target).abs()
        # Clamp diff to prevent extreme values
        diff = diff.clamp(max=100.0)
        masked_diff = torch.where(mask.bool(), diff, torch.zeros_like(diff))
        return masked_diff.sum() / valid_count.clamp(min=1.0)
    
    diff = (pred - target).abs().clamp(max=100.0)
    return diff.mean()

def focal_loss_with_label_smoothing(
    logits: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.25, 
    gamma: float = 2.0,
    smoothing: float = 0.05
) -> torch.Tensor:
    """
    Focal Loss with label smoothing for improved calibration.
    
    Label smoothing prevents overconfident predictions and
    can improve calibration metrics (ECE, Brier score).
    
    Args:
        logits: raw logits [N, C]
        y: targets {0,1} [N, C]
        alpha: weight for positive class
        gamma: focusing parameter
        smoothing: label smoothing factor (0.0 to 0.1 typical)
    """
    # FIXED: Input validation
    if torch.isnan(y).any() or torch.isinf(y).any():
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # FIXED: Clamp logits and labels for stability
    logits = logits.clamp(-100.0, 100.0)
    y = y.clamp(0.0, 1.0)
    
    # Apply label smoothing: y_smooth = (1 - smoothing) * y + smoothing * 0.5
    y_smooth = (1 - smoothing) * y + smoothing * 0.5
    
    # Standard focal loss computation with smoothed labels
    bce_loss = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction='none')
    
    # FIXED: Check if BCE produced NaN
    if torch.isnan(bce_loss).any() or torch.isinf(bce_loss).any():
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    p_t = probs * y_smooth + (1 - probs) * (1 - y_smooth)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)  # Use original y for alpha weighting
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    
    loss = alpha_t * focal_weight * bce_loss
    
    # FIXED: Final safety check
    final_loss = loss.mean()
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    
    return final_loss


def curl_consistency_l1(
    B_perp_from_Az_fn,
    A_z_points: torch.Tensor,
    coords_points: torch.Tensor,
    Bx_obs: Optional[torch.Tensor] = None,
    By_obs: Optional[torch.Tensor] = None,
    weight: float = 0.1
) -> torch.Tensor:
    if Bx_obs is None or By_obs is None or weight <= 0: 
        return (A_z_points * 0).sum()  # Connected to computation graph
    Bx, By = B_perp_from_Az_fn(A_z_points, coords_points)
    return weight * (Bx - Bx_obs).abs().mean() + weight * (By - By_obs).abs().mean()


# ============================================================================
# SOTA IMPROVEMENTS
# ============================================================================

def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation (Zhang et al., 2018).
    
    Particularly effective for imbalanced classification.
    Creates virtual training examples by interpolating between samples.
    
    Args:
        x: Input features [B, ...]
        y: Labels [B, C]
        alpha: Beta distribution parameter (0.2-0.4 typical)
        
    Returns:
        mixed_x: Interpolated features
        y_a: Original labels
        y_b: Mixed labels  
        lam: Interpolation coefficient
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Mixup loss: weighted combination of losses for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05
) -> torch.Tensor:
    """
    Asymmetric Focal Loss (Ben-Baruch et al., 2020).
    
    Key insight: For imbalanced data, we want different focusing for
    positive vs negative examples. More aggressive focusing on negatives
    helps with class imbalance.
    
    Args:
        logits: Raw predictions [N, C]
        y: Binary targets [N, C]
        gamma_pos: Focusing for positives (0 = no focusing)
        gamma_neg: Focusing for negatives (4 = strong focusing)
        clip: Probability clipping for negatives
        
    Returns:
        loss: Scalar loss
    """
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    
    # Positive focal term
    pos_term = y * ((1 - probs) ** gamma_pos) * torch.log(probs)
    
    # Negative focal term with probability clipping
    p_neg = (probs * (1 - y)).clamp(max=1 - clip)
    neg_term = (1 - y) * (p_neg ** gamma_neg) * torch.log(1 - probs)
    
    loss = -(pos_term + neg_term)
    return loss.mean()


class TemperatureScaling(torch.nn.Module):
    """
    Temperature scaling for calibration (Guo et al., 2017).
    
    Post-hoc calibration method that learns a single temperature
    parameter to scale logits. Apply after training on validation set.
    
    Usage:
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(val_logits, val_labels)
        calibrated_probs = temp_scaler(test_logits)
    """
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and return probabilities."""
        return torch.sigmoid(logits / self.temperature.clamp(min=0.01))
    
    def fit(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fit temperature on validation set using NLL minimization.
        
        Returns:
            Optimal temperature value
        """
        self.temperature.data.fill_(1.0)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return float(self.temperature.item())


def gradient_penalty(
    model: torch.nn.Module,
    coords: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    weight: float = 0.1
) -> torch.Tensor:
    """
    Gradient penalty for Lipschitz regularization.
    
    Encourages the model to have bounded gradients, which:
    - Improves generalization
    - Stabilizes physics loss training
    - Reduces overfitting to training distribution
    
    Args:
        model: The neural network
        coords: Input coordinates [N, 3]
        outputs: Model outputs dict with B_z, u_x, u_y
        weight: Regularization weight
        
    Returns:
        Gradient penalty loss
    """
    if weight <= 0:
        return (coords * 0).sum()  # Connected to computation graph
    
    # Compute gradient norm of B_z w.r.t inputs
    B_z = outputs.get("B_z")
    if B_z is None or not coords.requires_grad:
        return (coords * 0).sum()  # Connected to computation graph
    
    grads = torch.autograd.grad(
        B_z, coords,
        grad_outputs=torch.ones_like(B_z),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Penalize gradient norm exceeding 1 (1-Lipschitz)
    grad_norm = grads.norm(2, dim=-1)
    penalty = ((grad_norm - 1).clamp(min=0) ** 2).mean()
    
    return weight * penalty


def class_balanced_focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    samples_per_class: tuple[int, int] = (1000, 50),
    beta: float = 0.9999,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Class-Balanced Focal Loss (Cui et al., 2019).
    
    Combines class-balanced re-weighting with focal loss for extreme imbalance.
    Much more effective than standard focal loss for 1:50 imbalance.
    
    The effective number of samples is: E_n = (1 - β^n) / (1 - β)
    This captures the diminishing returns of more samples for majority class.
    
    Args:
        logits: Raw predictions [N, C]
        y: Binary targets [N, C]  
        samples_per_class: (n_negative, n_positive) sample counts
        beta: Hyperparameter in [0, 1). Higher = more weighting. 0.9999 for extreme imbalance.
        gamma: Focal loss focusing parameter
        
    Returns:
        loss: Scalar class-balanced focal loss
    """
    n_neg, n_pos = samples_per_class
    
    # Effective number of samples
    def effective_num(n: int, beta: float) -> float:
        if beta == 1.0:
            return float(n)
        return (1.0 - beta ** n) / (1.0 - beta)
    
    E_neg = effective_num(n_neg, beta)
    E_pos = effective_num(n_pos, beta)
    
    # Class-balanced weights (inversely proportional to effective number)
    w_neg = 1.0 / E_neg
    w_pos = 1.0 / E_pos
    
    # Normalize so weights sum to 2 (for binary classification)
    total = w_neg + w_pos
    w_neg = 2.0 * w_neg / total
    w_pos = 2.0 * w_pos / total
    
    # Compute focal loss with class-balanced weights
    # ±15 logit clamp: sigmoid(15)≈0.999999, sigmoid(-15)≈0.000001
    # Beyond ±15: BCE gradients vanish, focal weights overflow, no meaningful learning
    logits_clamped = logits.clamp(-15.0, 15.0)
    bce_loss = F.binary_cross_entropy_with_logits(logits_clamped, y, reduction='none')
    
    probs = torch.sigmoid(logits_clamped).clamp(1e-7, 1 - 1e-7)
    p_t = probs * y + (1 - probs) * (1 - y)
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    
    # Apply class-balanced weights
    cb_weight = w_pos * y + w_neg * (1 - y)
    
    loss = cb_weight * focal_weight * bce_loss
    
    # Final safety: clamp loss and handle NaN
    loss = loss.clamp(max=100.0)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)
    
    return loss.mean()


def poly_focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1.0,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    PolyLoss + Focal Loss (Leng et al., 2022).
    
    Adds polynomial correction term that helps with calibration
    and improves performance on imbalanced datasets.
    
    L_poly = L_focal + ε * (1 - p_t)
    
    Args:
        logits: Raw predictions [N, C]
        y: Binary targets [N, C]
        epsilon: Polynomial coefficient (1.0 works well)
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        
    Returns:
        loss: Scalar polyloss
    """
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    
    # Standard focal loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    p_t = probs * y + (1 - probs) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    focal_loss = alpha_t * focal_weight * bce_loss
    
    # Polynomial correction
    poly_term = epsilon * (1 - p_t)
    
    loss = focal_loss + poly_term
    return loss.mean()


def confidence_penalty(
    probs: torch.Tensor,
    weight: float = 0.1
) -> torch.Tensor:
    """
    Confidence penalty to prevent overconfident predictions.
    
    Adds entropy regularization to encourage calibrated predictions,
    especially important for rare event prediction like flares.
    
     FIXED: Now returns a bounded value that cannot make total loss negative.
    Uses inverse entropy (low entropy = high penalty) instead of negative entropy.
    
    Args:
        probs: Predicted probabilities [N, C]
        weight: Regularization weight
        
    Returns:
        Entropy penalty loss (to be added to main loss) - always >= 0
    """
    if weight <= 0:
        return (probs * 0).sum()  # Connected to computation graph
    
    probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
    
    # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    # Max entropy at p=0.5: H(0.5) = ln(2) ≈ 0.693
    entropy = -(probs_clamped * torch.log(probs_clamped) + 
                (1 - probs_clamped) * torch.log(1 - probs_clamped))
    
    #  FIX: Penalize LOW entropy (overconfidence) instead of using negative entropy
    # max_entropy = ln(2) ≈ 0.693 for binary classification
    # penalty = weight * (max_entropy - actual_entropy)
    # This is always >= 0 and penalizes confident predictions
    max_entropy = 0.693  # ln(2)
    penalty = weight * (max_entropy - entropy.mean()).clamp(min=0.0)
    
    return penalty

