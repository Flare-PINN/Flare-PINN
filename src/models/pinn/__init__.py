# PINN models package
from .config import (
    PINNConfig,
    ModelConfig,
    ClassifierConfig,
    PhysicsConfig,
    EtaConfig,
    LossWeightsConfig,
    CollocationConfig,
    TrainConfig,
    DataConfig,
    FourierConfig,
    SchedulerConfig,
    SamplerConfig,
)
from .core import (
    PINNBackbone, 
    ClassifierHead, 
    B_perp_from_Az, 
    FourierFeatures,
    SpatialAttentionPool,
    TemporalAttentionPool,
    PhysicsFeatureExtractor,
)
from .losses import (
    focal_loss, focal_loss_with_label_smoothing, bce_logits, l1_data, 
    curl_consistency_l1, interp_schedule, asymmetric_focal_loss,
    mixup_data, mixup_criterion, TemperatureScaling, 
    gradient_penalty, confidence_penalty,
    class_balanced_focal_loss, poly_focal_loss,
)
from .physics import (
    WeakFormInduction2p5D,
    VectorInduction2p5D,
    MultiScaleTestFunction,
    PhysicsResidualInfo,
    VectorPhysicsResidualInfo,
    FreeEnergyProxy,
    CurrentHelicityProxy,
)
from .collocation import mix_pil_uniform, clip_and_renorm_importance, ess, sample_xy_from_mask
from .model import PINNModel, PINNOutput
from .hybrid_model import HybridPINNModel, HybridPINNOutput
from .encoder import TinyEncoder, TemporalEncoder, MultiScaleEncoder
from .rf_guidance import (
    RFImportances,
    RFGuidedFeatureWeighter,
    compute_handcrafted_features,
    train_rf_for_importances,
    compute_rf_importances_from_dataset,
)

__all__ = [
    # Config
    "PINNConfig",
    "ModelConfig", 
    "ClassifierConfig",
    "PhysicsConfig",
    "EtaConfig",
    "LossWeightsConfig",
    "CollocationConfig",
    "TrainConfig",
    "DataConfig",
    "FourierConfig",
    "SchedulerConfig",
    "SamplerConfig",
    # Core models
    "PINNBackbone",
    "ClassifierHead",
    "B_perp_from_Az",
    "FourierFeatures",
    "PINNModel",
    "PINNOutput",
    "SpatialAttentionPool",
    "TemporalAttentionPool",
    "PhysicsFeatureExtractor",
    # Hybrid CNN-PINN
    "HybridPINNModel",
    "HybridPINNOutput",
    # Encoders
    "TinyEncoder",
    "TemporalEncoder",
    "MultiScaleEncoder",
    # Losses
    "focal_loss",
    "focal_loss_with_label_smoothing",
    "asymmetric_focal_loss",
    "class_balanced_focal_loss",
    "poly_focal_loss",
    "bce_logits",
    "l1_data",
    "curl_consistency_l1",
    "interp_schedule",
    "mixup_data",
    "mixup_criterion",
    "TemperatureScaling",
    "gradient_penalty",
    "confidence_penalty",
    # Physics
    "WeakFormInduction2p5D",
    "VectorInduction2p5D",
    "MultiScaleTestFunction",
    "PhysicsResidualInfo",
    "VectorPhysicsResidualInfo",
    "FreeEnergyProxy",
    "CurrentHelicityProxy",
    # Collocation
    "mix_pil_uniform",
    "clip_and_renorm_importance",
    "ess",
    "sample_xy_from_mask",
    # RF Guidance
    "RFImportances",
    "RFGuidedFeatureWeighter",
    "compute_handcrafted_features",
    "train_rf_for_importances",
    "compute_rf_importances_from_dataset",
]
