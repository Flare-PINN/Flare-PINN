from __future__ import annotations
import numpy as np
from typing import Tuple

def temperature_scale(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    from scipy.optimize import minimize
    z = logits.astype(np.float64).reshape(-1)
    y = y_true.astype(np.float64).reshape(-1)
    def nll(logT):
        T = np.exp(logT)
        p = 1.0 / (1.0 + np.exp(-z / T))
        eps = 1e-12
        return -np.mean(y * np.log(p + eps) + (1. - y) * np.log(1. - p + eps))
    res = minimize(nll, x0=0.0, method="L-BFGS-B")
    T = float(np.exp(res.x[0]))
    p = 1.0 / (1.0 + np.exp(-z / T))
    return T, p.reshape(logits.shape)

def isotonic_scale(probs: np.ndarray, y_true: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    p = iso.fit_transform(probs.reshape(-1), y_true.reshape(-1))
    return iso, p.reshape(probs.shape)
