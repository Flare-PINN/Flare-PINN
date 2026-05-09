from __future__ import annotations
import numpy as np
from typing import Tuple

from pydantic import BaseModel, ConfigDict

def _safe_eps(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    
    # Filter out NaN/Inf values
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return 0.0
    
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)  # Clamp to valid probability range
    
    return float(np.mean((y_prob - y_true) ** 2))

def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> tuple[int,int,int,int]:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn

def tpr_fpr(tp:int, fp:int, fn:int, tn:int) -> Tuple[float,float]:
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return tpr, fpr


class ThresholdClassificationMetrics(BaseModel):
    """Skill scores at a fixed probability threshold (e.g. validation D2C applied on test)."""

    model_config = ConfigDict(frozen=True)

    tss: float
    pod: float
    fpr: float
    far: float
    csi: float
    tp: int
    fp: int
    fn: int
    tn: int


def threshold_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float
) -> ThresholdClassificationMetrics:
    """
    POD = TPR; FPR = FP/(FP+TN); FAR = FP/(TP+FP) (meteorological false-alarm ratio); CSI = TP/(TP+FN+FP).
    """
    tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, thr)
    tpr, fpr = tpr_fpr(tp, fp, fn, tn)
    far = fp / max(1, tp + fp)
    csi = tp / max(1, tp + fn + fp)
    tss = float(tpr - fpr)
    return ThresholdClassificationMetrics(
        tss=tss,
        pod=float(tpr),
        fpr=float(fpr),
        far=float(far),
        csi=float(csi),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )

def tss_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, thr)
    tpr, fpr = tpr_fpr(tp, fp, fn, tn)
    return float(tpr - fpr)

def distance_to_corner(y_true: np.ndarray, y_prob: np.ndarray, n: int = 1024) -> float:
    """
    Find threshold that minimizes Euclidean distance to perfect ROC corner (TPR=1, FPR=0).
    
    This geometric approach is more robust than pointwise maximum TSS,
    as it optimizes for global ROC curve quality rather than a single point.
    
    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        n: Number of thresholds to evaluate
    
    Returns:
        Optimal threshold based on distance-to-corner criterion
    """
    # Fixed seed for reproducibility
    rng = np.random.RandomState(42)
    n = max(n, 4)
    
    # Filter NaN/Inf
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return 0.5
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    
    # Get thresholds
    unique_probs = np.unique(y_prob)
    if len(unique_probs) <= n:
        thrs = unique_probs
    else:
        n_half = max(n // 2, 2)
        linspace_thrs = np.linspace(0, 1, n_half)
        sampled_unique = rng.choice(unique_probs, size=min(n_half, len(unique_probs)), replace=False)
        thrs = np.unique(np.concatenate([linspace_thrs, sampled_unique]))
    
    # Find threshold minimizing distance to (TPR=1, FPR=0)
    best_dist = float('inf')
    best_thr = 0.5
    
    for thr in thrs:
        tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, float(thr))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        
        # Euclidean distance to perfect corner (1, 0)
        dist = np.sqrt((1.0 - tpr)**2 + (0.0 - fpr)**2)
        
        if dist < best_dist:
            best_dist = dist
            best_thr = float(thr)
    
    return best_thr

def sweep_tss(y_true: np.ndarray, y_prob: np.ndarray, n: int = 1024, robust: bool = False, robust_pct: float = 0.98, robust_method: str = 'min') -> Tuple[float, float]:
    """
    Find optimal TSS threshold via exhaustive search.
    
    NOTE: This is used for CHECKPOINT SELECTION during training only.
    For final test evaluation, use distance_to_corner() instead.
    
    Uses unique probability values when dataset is small enough,
    otherwise uses linspace + random unique probability samples.
    
    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        n: Number of thresholds to evaluate
        robust: If True, use robust threshold selection within robust_pct% of max TSS
        robust_pct: Percentage of max TSS to consider (e.g., 0.98 = within 98% of max)
        robust_method: How to select from valid thresholds ('min', 'mean', 'median')
            - 'min': Most conservative (lowest threshold)
            - 'mean': Average of top thresholds
            - 'median': Median of top thresholds
    
    NOTE: Uses fixed seed for reproducibility.
    """
    # Fixed seed for reproducibility
    rng = np.random.RandomState(42)
    
    #  FIX #13: Guard against n < 2
    n = max(n, 4)  # Minimum 4 thresholds for meaningful sweep
    
    # Filter NaN/Inf
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return 0.5, 0.0
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    
    # Always include unique probability values for better precision
    unique_probs = np.unique(y_prob)
    
    if len(unique_probs) <= n:
        thrs = unique_probs
    else:
        # Combine linspace with sampled unique probabilities
        n_half = max(n // 2, 2)  # Ensure at least 2 points each
        linspace_thrs = np.linspace(0, 1, n_half)
        sampled_unique = rng.choice(unique_probs, size=min(n_half, len(unique_probs)), replace=False)
        thrs = np.unique(np.concatenate([linspace_thrs, sampled_unique]))
    
    # Compute TSS for all thresholds
    tss_scores = []
    for thr in thrs:
        s = tss_at_threshold(y_true, y_prob, float(thr))
        tss_scores.append(s)
    
    tss_scores = np.array(tss_scores)
    best_tss = tss_scores.max()
    
    if not robust:
        # Original behavior: return threshold with maximum TSS
        best_idx = tss_scores.argmax()
        return float(thrs[best_idx]), float(best_tss)
    else:
        # Robust behavior: select from thresholds within robust_pct% of max TSS
        threshold_for_robust = best_tss * robust_pct
        valid_indices = np.where(tss_scores >= threshold_for_robust)[0]
        
        if len(valid_indices) == 0:
            # Fallback to original behavior
            best_idx = tss_scores.argmax()
            return float(thrs[best_idx]), float(best_tss)
        
        valid_thresholds = thrs[valid_indices]
        valid_tss_scores = tss_scores[valid_indices]
        
        # Select threshold based on method
        if robust_method == 'min':
            # Most conservative: lowest threshold
            robust_idx = valid_indices[np.argmin(valid_thresholds)]
        elif robust_method == 'mean':
            # Average of top thresholds
            robust_thr = float(np.mean(valid_thresholds))
            robust_tss = tss_at_threshold(y_true, y_prob, robust_thr)
            return robust_thr, float(robust_tss)
        elif robust_method == 'median':
            # Median of top thresholds
            robust_thr = float(np.median(valid_thresholds))
            robust_tss = tss_at_threshold(y_true, y_prob, robust_thr)
            return robust_thr, float(robust_tss)
        else:
            raise ValueError(f"Unknown robust_method: {robust_method}")
        
        robust_thr = float(thrs[robust_idx])
        robust_tss = float(tss_scores[robust_idx])
        
        return robust_thr, robust_tss

def select_threshold_at_far(y_true: np.ndarray, y_prob: np.ndarray, max_far: float = 0.05, n: int = 2048) -> float:
    thrs = np.linspace(0, 1, n)
    chosen = 1.0
    for thr in thrs:
        _, fp, _, tn = confusion_at_threshold(y_true, y_prob, float(thr))
        fpr = fp / max(1, fp + tn)
        if fpr <= max_far:
            chosen = float(thr)
            break
    return chosen

def precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, n: int = 512) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    thrs = np.linspace(0, 1, n)[::-1]
    P, R = [], []
    for thr in thrs:
        tp, fp, fn, _ = confusion_at_threshold(y_true, y_prob, float(thr))
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        P.append(prec); R.append(rec)
    return np.asarray(R), np.asarray(P), thrs

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray, n: int = 512) -> float:
    # Filter NaN/Inf
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return 0.0
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    
    R, P, _ = precision_recall_curve(y_true, y_prob, n=n)
    idx = np.argsort(R)
    R, P = R[idx], P[idx]
    return float(np.trapz(P, R))

def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    
    # Filter out NaN/Inf values
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return 0.0
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 1e-12, 1.0 - 1e-12)
    
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(y_prob, qs)
    edges[0], edges[-1] = 0.0, 1.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + 1e-12
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: 
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
    return float(ece)

def reliability_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float,float]:
    y_true = y_true.astype(np.float64)
    y_prob = _safe_eps(y_prob.astype(np.float64))
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(y_prob, qs); edges[0], edges[-1] = 0.0, 1.0
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + 1e-12
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        xs.append(float(y_prob[m].mean()))
        ys.append(float(y_true[m].mean()))
        ws.append(float(m.sum()))
    X = np.vstack([np.ones(len(xs)), np.asarray(xs)]).T
    y = np.asarray(ys)
    W = np.diag(np.asarray(ws))
    XtWX = X.T @ W @ X
    beta = np.linalg.pinv(XtWX) @ X.T @ W @ y
    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept
