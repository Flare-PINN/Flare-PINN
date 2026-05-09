# src/models/pinn/rf_guidance.py
"""
Random Forest Feature Importance Guidance for PINN.

Computes traditional physics features from magnetograms, trains a Random Forest
to find which features matter most, and provides importance weights to guide
the neural network's attention and feature weighting.

This gives the CNN-PINN a "head start" by telling it what domain experts
already know predicts flares.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
import pickle

# Lazy import sklearn - only needed when training RF
if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler


@dataclass
class RFImportances:
    """Container for RF feature importances."""
    feature_names: list[str]
    importances: np.ndarray  # [n_features]
    normalized_importances: np.ndarray  # [n_features], sum to 1
    rf_accuracy: float
    rf_tss: float
    
    def get_weight(self, feature_name: str) -> float:
        """Get importance weight for a specific feature."""
        if feature_name in self.feature_names:
            idx = self.feature_names.index(feature_name)
            return float(self.normalized_importances[idx])
        return 1.0 / len(self.feature_names)  # Default uniform
    
    def get_weights_dict(self) -> dict[str, float]:
        """Get all importances as a dictionary."""
        return {name: float(imp) for name, imp in 
                zip(self.feature_names, self.normalized_importances)}
    
    def save(self, path: str | Path):
        """Save importances to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str | Path) -> RFImportances:
        """Load importances from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def compute_handcrafted_features(bz_frames: np.ndarray) -> dict[str, float]:
    """
    Compute traditional physics features from Bz magnetogram sequence.
    
    These are features that solar physicists have identified as
    predictive of flares.
    
    Args:
        bz_frames: [T, H, W] sequence of Bz magnetograms
        
    Returns:
        Dictionary of feature name -> value
    """
    # Use last frame for instantaneous features
    bz = bz_frames[-1] if bz_frames.ndim == 3 else bz_frames
    H, W = bz.shape
    
    features = {}
    
    # 1. Total unsigned flux
    features['total_unsigned_flux'] = np.abs(bz).sum()
    
    # 2. Net flux (imbalance)
    features['net_flux'] = np.abs(bz.sum())
    
    # 3. Flux imbalance ratio
    pos_flux = bz[bz > 0].sum() if np.any(bz > 0) else 0
    neg_flux = np.abs(bz[bz < 0].sum()) if np.any(bz < 0) else 0
    total = pos_flux + neg_flux + 1e-10
    features['flux_imbalance'] = np.abs(pos_flux - neg_flux) / total
    
    # 4. Polarity balance (high = complex PIL)
    features['polarity_balance'] = np.sqrt(pos_flux * neg_flux) / (total / 2 + 1e-10)
    
    # 5. Gradient magnitude statistics
    gy, gx = np.gradient(bz)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features['mean_gradient'] = grad_mag.mean()
    features['max_gradient'] = grad_mag.max()
    features['gradient_std'] = grad_mag.std()
    
    # 6. PIL detection (where gradient is high and Bz is near zero)
    near_zero = np.abs(bz) < np.std(bz) * 0.5
    pil_mask = near_zero & (grad_mag > np.percentile(grad_mag, 85))
    features['pil_length'] = pil_mask.sum()
    
    # 7. Gradient-weighted PIL length (GWPIL)
    features['gwpil'] = (grad_mag * pil_mask).sum()
    
    # 8. R-value (Schrijver's unsigned flux near PIL)
    pil_dilated = _dilate_mask(pil_mask, radius=3)
    r_flux = np.abs(bz[pil_dilated]).sum()
    features['r_value'] = np.log10(r_flux + 1)
    
    # 9. Field statistics
    features['bz_mean'] = np.abs(bz).mean()
    features['bz_std'] = bz.std()
    features['bz_max'] = np.abs(bz).max()
    features['bz_skewness'] = _skewness(bz)
    features['bz_kurtosis'] = _kurtosis(bz)
    
    # 10. Spatial complexity (entropy-like)
    bz_norm = (bz - bz.min()) / (bz.max() - bz.min() + 1e-10)
    hist, _ = np.histogram(bz_norm.flatten(), bins=50, density=True)
    hist = hist + 1e-10
    features['spatial_entropy'] = -np.sum(hist * np.log(hist))
    
    # 11. Temporal features (if multiple frames)
    if bz_frames.ndim == 3 and bz_frames.shape[0] > 1:
        # Rate of change
        diff = np.abs(bz_frames[1:] - bz_frames[:-1])
        features['temporal_variation'] = diff.mean()
        features['max_temporal_change'] = diff.max()
        
        # Flux emergence rate
        flux_per_frame = np.abs(bz_frames).sum(axis=(1, 2))
        features['flux_emergence_rate'] = np.gradient(flux_per_frame).mean()
    else:
        features['temporal_variation'] = 0.0
        features['max_temporal_change'] = 0.0
        features['flux_emergence_rate'] = 0.0
    
    return features


def _dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Simple dilation using convolution."""
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        struct = generate_binary_structure(2, 1)
        return binary_dilation(mask, structure=struct, iterations=radius)
    except ImportError:
        # Simple fallback
        return mask


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    mean = x.mean()
    std = x.std() + 1e-10
    return float(((x - mean) ** 3).mean() / (std ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis."""
    mean = x.mean()
    std = x.std() + 1e-10
    return float(((x - mean) ** 4).mean() / (std ** 4) - 3)


def train_rf_for_importances(
    frames_list: list[np.ndarray],
    labels: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42
) -> tuple[RFImportances, 'RandomForestClassifier']:
    """
    Train Random Forest on handcrafted features to get importances.
    
    Args:
        frames_list: List of [T, H, W] magnetogram sequences
        labels: [N] binary flare labels
        n_estimators: Number of trees
        max_depth: Max tree depth (prevent overfitting)
        random_state: Random seed
        
    Returns:
        RFImportances object and trained RF model
        
    Requires:
        scikit-learn: pip install scikit-learn
    """
    # Import sklearn here (lazy import)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for RF guidance. "
            "Install with: pip install scikit-learn"
        )
    
    print("Computing handcrafted features for RF...")
    
    # Extract features for all samples
    all_features = []
    feature_names = None
    
    for i, frames in enumerate(frames_list):
        feats = compute_handcrafted_features(frames)
        if feature_names is None:
            feature_names = list(feats.keys())
        all_features.append([feats[name] for name in feature_names])
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(frames_list)} samples")
    
    X = np.array(all_features)
    y = np.array(labels)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Training Random Forest on {len(feature_names)} features...")
    
    # Train RF with class balancing
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)
    
    # Get predictions for metrics
    y_pred = rf.predict(X_scaled)
    y_prob = rf.predict_proba(X_scaled)[:, 1]
    
    # Compute metrics
    accuracy = (y_pred == y).mean()
    
    # TSS
    tp = ((y_pred == 1) & (y == 1)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    tss = tpr - fpr
    
    print(f"RF Training Accuracy: {accuracy:.3f}")
    print(f"RF Training TSS: {tss:.3f}")
    
    # Get importances
    importances = rf.feature_importances_
    normalized = importances / importances.sum()
    
    # Print top features
    print("\nTop 10 Important Features:")
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    rf_importances = RFImportances(
        feature_names=feature_names,
        importances=importances,
        normalized_importances=normalized,
        rf_accuracy=float(accuracy),
        rf_tss=float(tss)
    )
    
    return rf_importances, rf


class RFGuidedFeatureWeighter:
    """
    Applies RF importance weights to physics features in the classifier.
    
    Usage:
        weighter = RFGuidedFeatureWeighter(rf_importances)
        weighted_feats = weighter(physics_features)
    """
    
    def __init__(self, rf_importances: Optional[RFImportances] = None):
        """
        Args:
            rf_importances: Pre-computed RF importances. If None, uses uniform weights.
        """
        self.rf_importances = rf_importances
        
        # Map physics feature names to RF feature names
        self.feature_mapping = {
            'bz_mean': 'bz_mean',
            'bz_std': 'bz_std', 
            'bz_max': 'bz_max',
            'polarity_balance': 'polarity_balance',
            'u_mean': 'mean_gradient',  # Proxy: flow ~ gradient
            'u_max': 'max_gradient',
            'flux_transport': 'gwpil',  # Proxy
            'temporal_var': 'temporal_variation',
            'az_std': 'spatial_entropy',  # Proxy
        }
    
    def get_weights(self, n_features: int = 9) -> np.ndarray:
        """
        Get importance weights for the 9 physics features.
        
        Returns:
            [n_features] array of weights (sum to n_features for scale preservation)
        """
        if self.rf_importances is None:
            return np.ones(n_features)
        
        weights = []
        physics_features = ['bz_mean', 'bz_std', 'bz_max', 'polarity_balance',
                          'u_mean', 'u_max', 'flux_transport', 'temporal_var', 'az_std']
        
        for pf in physics_features[:n_features]:
            rf_name = self.feature_mapping.get(pf, pf)
            w = self.rf_importances.get_weight(rf_name)
            weights.append(w)
        
        weights = np.array(weights)
        # Scale so weights sum to n_features (preserves feature scale)
        weights = weights / weights.sum() * n_features
        
        return weights
    
    def __call__(self, features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Apply RF importance weights to physics features.
        
        Args:
            features: [B, n_features] physics features
            
        Returns:
            [B, n_features] weighted features
        """
        import torch
        
        weights = self.get_weights(features.shape[-1])
        weights_tensor = torch.tensor(weights, dtype=features.dtype, device=features.device)
        
        return features * weights_tensor


def compute_rf_importances_from_dataset(
    windows_df: 'pd.DataFrame',
    frames_meta_path: str,
    npz_root: str,
    horizons: list[int] = [24],
    save_path: Optional[str] = None
) -> RFImportances:
    """
    Convenience function to compute RF importances from the training dataset.
    
    Args:
        windows_df: DataFrame with window information
        frames_meta_path: Path to frames metadata
        npz_root: Root directory for NPZ files
        horizons: Which horizons to use for labels
        save_path: Optional path to save importances
        
    Returns:
        RFImportances object
    """
    import pandas as pd
    from pathlib import Path
    
    print(f"Loading data from {len(windows_df)} windows...")
    
    meta = pd.read_parquet(frames_meta_path)
    meta["date_obs"] = pd.to_datetime(meta["date_obs"], utc=True)
    
    frames_list = []
    labels = []
    
    for idx, row in windows_df.iterrows():
        if idx % 200 == 0:
            print(f"  Loading window {idx}/{len(windows_df)}")
        
        harpnum = int(row["harpnum"])
        t0 = pd.to_datetime(row["t0"], utc=True)
        
        # Get frames for this window
        g = meta[meta["harpnum"] == harpnum]
        if len(g) == 0:
            continue
        
        # Load last few frames
        g_recent = g[g["date_obs"] <= t0].tail(5)
        if len(g_recent) == 0:
            continue
        
        # Load frames
        bz_frames = []
        for _, frame_row in g_recent.iterrows():
            try:
                npz_path = Path(npz_root) / frame_row["frame_path"]
                data = np.load(npz_path)
                bz = data.get("Bz", data.get("Br", None))
                if bz is not None:
                    bz_frames.append(bz.astype(np.float32))
            except:
                continue
        
        if len(bz_frames) == 0:
            continue
        
        bz_stack = np.stack(bz_frames, axis=0)
        frames_list.append(bz_stack)
        
        # Get label (any horizon)
        label = 0
        for h in horizons:
            col = f"y_geq_M_{h}h"
            if col in row and row[col]:
                label = 1
                break
        labels.append(label)
    
    print(f"Loaded {len(frames_list)} samples ({sum(labels)} positive)")
    
    # Train RF
    rf_importances, rf = train_rf_for_importances(frames_list, np.array(labels))
    
    # Save if requested
    if save_path:
        rf_importances.save(save_path)
        print(f"Saved RF importances to {save_path}")
    
    return rf_importances

