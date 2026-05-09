from __future__ import annotations
import numpy as np
from typing import Callable, Sequence, Tuple

from .metrics import tss_at_threshold


def cluster_bootstrap_tss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cluster_ids: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute cluster bootstrap confidence interval for TSS at a fixed threshold.
    
    Resamples entire clusters (e.g., HARPs) with replacement to account for
    temporal correlation within active regions.
    
    Args:
        y_true: Ground truth labels [N]
        y_prob: Predicted probabilities [N]
        cluster_ids: Cluster identifiers (e.g., HARP numbers) [N]
        threshold: Fixed decision threshold
        n_bootstrap: Number of bootstrap iterations
        confidence_level: CI level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        (lower_bound, upper_bound): Confidence interval
    """
    rng = np.random.default_rng(seed)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    
    # Bootstrap loop
    bootstrap_tss = []
    for _ in range(n_bootstrap):
        # Resample clusters with replacement
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        
        # Gather all samples from sampled clusters
        idx = np.concatenate([np.where(cluster_ids == c)[0] for c in sampled_clusters])
        
        # Compute TSS on bootstrap sample
        tss = tss_at_threshold(y_true[idx], y_prob[idx], threshold)
        bootstrap_tss.append(tss)
    
    bootstrap_tss = np.array(bootstrap_tss)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_tss, 100 * alpha / 2)
    upper = np.percentile(bootstrap_tss, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def paired_block_bootstrap(
    groups: Sequence[int],
    y_true_A: np.ndarray, y_prob_A: np.ndarray,
    y_true_B: np.ndarray, y_prob_B: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = rng or np.random.default_rng(12345)
    def _metric(y_true, y_prob):
        return float(metric_fn(y_true, y_prob))
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sample])
        mA = _metric(y_true_A[idx], y_prob_A[idx])
        mB = _metric(y_true_B[idx], y_prob_B[idx])
        deltas.append(mA - mB)
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_mean": float(deltas.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}
