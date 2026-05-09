"""Pydantic configuration for DeFN training — matches original paper/code."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DeFNDataConfig(BaseModel):
    """Data paths and split configuration."""
    n_features: int = Field(
        default=79,
        description="Expected number of features (79 for DeFN 2018).",
    )
    features_path: str = Field(
        default="data/defn/defn_features.parquet",
        description="Path to the combined DeFN feature matrix.",
    )
    train_val_windows: str = Field(
        default="data/windows_train_val_8005.parquet",
        description="Train+val window parquet (for split alignment).",
    )
    test_windows: str = Field(
        default="data/windows_test_15.parquet",
        description="Test window parquet (for split alignment).",
    )
    val_fraction: float = Field(
        default=0.0588,
        description="Fraction of train+val to use as validation (chronological tail).",
    )


class DeFNModelConfig(BaseModel):
    """
    DeFN architecture (from the original TensorFlow code).

    Network: [dim_X, 200, 200, dim_X, 200, 200, dim_X, 200, 2]
    with 2 residual skip connections and batch normalization.
    """
    hidden_dim: int = Field(default=200, description="Hidden layer width.")
    dropout_keep: float = Field(
        default=0.75, description="Dropout keep probability (original pkeep=0.75)."
    )
    n_classes: int = Field(default=2, description="Binary classification.")


class DeFNTrainConfig(BaseModel):
    """
    Training hyperparameters from the original code.

    batch_size=150, Adam lr=0.001, class_weights=[1, 60], max_epoch=16000
    """
    lr: float = Field(default=1e-3, description="Learning rate (Adam).")
    batch_size: int = Field(default=150, description="Batch size (original: 150).")
    epochs: int = Field(
        default=200,
        description="Training epochs. Original used 16000 iterations over "
                    "mini-batches; 200 full-dataset epochs is roughly equivalent.",
    )
    weight_decay: float = Field(default=0.0, description="L2 regularization.")
    pos_class_weight: float = Field(
        default=60.0,
        description="Weight for positive (flare) class in cross-entropy. "
                    "Original: [1, 60] for ≥M-class.",
    )
    seeds: list[int] = Field(
        default=[24, 10, 100, 42, 123],
        description="Random seeds for multi-seed training. Default reproduces the published 5-seed DeFN ensemble.",
    )
    patience: int = Field(
        default=30,
        description="Early stopping patience (epochs without val TSS improvement).",
    )
    horizons: list[str] = Field(
        default=["6h", "12h", "24h"],
        description="Forecast horizons to train.",
    )


class DeFNConfig(BaseModel):
    """Top-level DeFN configuration."""
    data: DeFNDataConfig = Field(default_factory=DeFNDataConfig)
    model: DeFNModelConfig = Field(default_factory=DeFNModelConfig)
    train: DeFNTrainConfig = Field(default_factory=DeFNTrainConfig)


class DeFNRunResult(BaseModel):
    """One training run metrics (written to ``defn_results.csv``)."""
    seed: int
    horizon: str
    iters: int
    argmax_tss: float
    threshold: float
    test_tss: float
    test_pod: float
    test_far: float
    test_csi: float
    tp: int
    fp: int
    fn: int
    tn: int


class DeFNSummaryRow(BaseModel):
    """
    Per-horizon aggregates over seeds (written to ``defn_summary.csv``).

    Std devs use the sample formula (ddof=1), matching typical paper reporting.
    """
    horizon: str
    n_seeds: int
    mean_test_tss: float
    std_test_tss: float
    mean_argmax_tss: float
    std_argmax_tss: float
    mean_pod: float
    std_pod: float
    mean_far: float
    std_far: float
    mean_csi: float
    std_csi: float
