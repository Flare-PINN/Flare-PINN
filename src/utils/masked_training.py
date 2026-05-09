"""
Masked Training Utilities
==========================

Handles training only on windows with valid NOAA AR mappings,
preventing false negatives from unmapped HARPs.
"""

import torch
import pandas as pd
from typing import Tuple


def load_windows_with_mask(windows_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load windows and extract the has_noaa_mapping mask.
    
    Args:
        windows_path: Path to windows parquet file
        
    Returns:
        windows_df: Full DataFrame
        mask: Boolean series indicating which windows have valid labels
    """
    df = pd.read_parquet(windows_path)
    
    if 'has_noaa_mapping' not in df.columns:
        raise ValueError(
            "Windows file missing 'has_noaa_mapping' column. "
            "Regenerate windows with updated create_windows.py"
        )
    
    mask = df['has_noaa_mapping']
    
    # Report coverage
    n_labeled = mask.sum()
    pct = 100.0 * n_labeled / len(mask)
    print(f"Label coverage: {n_labeled}/{len(mask)} ({pct:.1f}%) windows have valid labels")
    
    return df, mask


def filter_to_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only windows with valid labels.
    
    Use this when you want to work with a clean subset.
    """
    if 'has_noaa_mapping' not in df.columns:
        raise ValueError("DataFrame missing 'has_noaa_mapping' column")
    
    return df[df['has_noaa_mapping']].copy()


class MaskedBCELoss(torch.nn.Module):
    """
    Binary Cross Entropy that only computes loss on labeled samples.
    
    Example:
        criterion = MaskedBCELoss()
        loss = criterion(logits, targets, mask)
    """
    
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size] or [batch_size, 1]
            targets: Ground truth labels [batch_size]
            mask: Boolean mask [batch_size] - True for labeled samples
            
        Returns:
            loss: Scalar loss computed only on labeled samples
        """
        # Ensure shapes match
        if logits.dim() == 2:
            logits = logits.squeeze(1)
        
        # Only compute loss where mask is True
        if mask.sum() == 0:
            # No labeled samples in batch - return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_labeled = logits[mask]
        targets_labeled = targets[mask].float()
        
        if self.pos_weight is not None:
            bce = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            bce = torch.nn.BCEWithLogitsLoss()
            
        return bce(logits_labeled, targets_labeled)


class MaskedFocalLoss(torch.nn.Module):
    """
    Focal Loss that only computes on labeled samples.
    Useful for imbalanced datasets.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            logits = logits.squeeze(1)
            
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_labeled = logits[mask]
        targets_labeled = targets[mask].float()
        
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_labeled, targets_labeled, reduction='none'
        )
        
        probs = torch.sigmoid(logits_labeled)
        pt = torch.where(targets_labeled == 1, probs, 1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets_labeled == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


def compute_masked_metrics(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Compute metrics only on labeled samples.
    
    Returns dict with:
        - accuracy, precision, recall, f1
        - n_labeled: number of labeled samples used
        - coverage: fraction of samples that were labeled
    """
    import sklearn.metrics as metrics
    
    if predictions.dim() == 2:
        predictions = predictions.squeeze(1)
    
    # Apply mask
    preds_labeled = predictions[mask].cpu().numpy()
    targets_labeled = targets[mask].cpu().numpy()
    
    # Convert probabilities to binary predictions
    if preds_labeled.min() >= 0 and preds_labeled.max() <= 1:
        # Already probabilities
        preds_binary = (preds_labeled > 0.5).astype(int)
    else:
        # Logits - apply sigmoid
        preds_binary = (torch.sigmoid(torch.tensor(preds_labeled)).numpy() > 0.5).astype(int)
    
    results = {
        'accuracy': metrics.accuracy_score(targets_labeled, preds_binary),
        'precision': metrics.precision_score(targets_labeled, preds_binary, zero_division=0),
        'recall': metrics.recall_score(targets_labeled, preds_binary, zero_division=0),
        'f1': metrics.f1_score(targets_labeled, preds_binary, zero_division=0),
        'n_labeled': int(mask.sum()),
        'coverage': float(mask.float().mean()),
    }
    
    return results


# Example usage in training loop:
"""
from src.utils.masked_training import MaskedBCELoss, load_windows_with_mask

# Load data
windows_df, label_mask = load_windows_with_mask('data/windows_train.parquet')

# Filter to only labeled samples (recommended)
train_df = windows_df[label_mask]

# OR: Use full dataset with masked loss (keeps batch sizes consistent)
criterion = MaskedBCELoss()

for batch in dataloader:
    logits = model(batch['features'])
    targets = batch['labels']
    mask = batch['has_label_mask']  # Include this in your dataset
    
    loss = criterion(logits, targets, mask)
    loss.backward()
    optimizer.step()
"""

