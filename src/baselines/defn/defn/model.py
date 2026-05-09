"""
DeFN (Deep Flare Net) — Faithful PyTorch reimplementation.

Translated line-by-line from the original TensorFlow code at
https://github.com/komeisugiura/defn18/blob/master/src/deepflarenet.py

Original architecture:
    extractor_num_nodes = [dim_X, 200, 200, dim_X, 200, 200, dim_X, 200, dim_Y]

    h1  = ReLU(W0 @ X + b0)            ;  Dropout(pkeep=0.75)
    h2  = ReLU(BN(W1 @ h1d))           ;  200 → 200
    h3  = ReLU(BN(W2 @ h2)) + X        ;  200 → dim_X, skip
    h4  = ReLU(BN(W3 @ h3))            ;  dim_X → 200
    h5  = ReLU(BN(W4 @ h4))            ;  200 → 200
    h6  = ReLU(BN(W5 @ h5)) + X        ;  200 → dim_X, skip
    h7  = ReLU(BN(W6 @ h6))            ;  dim_X → 200
    out = Softmax(W7 @ h7 + b7)        ;  200 → 2

Original BN: momentum=0.99, epsilon=0.001, center=True, scale=False
    (PyTorch momentum = 1 - TF momentum = 0.01)
    (scale=False → affine gamma fixed to 1, only beta is learned)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.baselines.defn.defn.config import DeFNModelConfig


class _BNNoScale(nn.Module):
    """BatchNorm matching original TF: center=True, scale=False."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(
            num_features, momentum=0.01, eps=0.001, affine=False,
        )
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x) + self.beta


class DeFN(nn.Module):
    """8-layer residual network — faithful to original TF code."""

    def __init__(self, n_features: int, cfg: DeFNModelConfig | None = None):
        super().__init__()
        cfg = cfg or DeFNModelConfig()
        d = cfg.hidden_dim  # 200
        dim_x = n_features

        self.fc0 = nn.Linear(dim_x, d)
        self.drop = nn.Dropout(1.0 - cfg.dropout_keep)

        self.fc1 = nn.Linear(d, d)
        self.bn1 = _BNNoScale(d)

        self.fc2 = nn.Linear(d, dim_x)
        self.bn2 = _BNNoScale(dim_x)

        self.fc3 = nn.Linear(dim_x, d)
        self.bn3 = _BNNoScale(d)

        self.fc4 = nn.Linear(d, d)
        self.bn4 = _BNNoScale(d)

        self.fc5 = nn.Linear(d, dim_x)
        self.bn5 = _BNNoScale(dim_x)

        self.fc6 = nn.Linear(dim_x, d)
        self.bn6 = _BNNoScale(d)

        self.fc_out = nn.Linear(d, cfg.n_classes)

        self._init_weights()

    def _init_weights(self):
        """Match original TF initializers: truncated_normal(stddev=0.1)."""
        for name, param in self.named_parameters():
            if "fc" in name and "weight" in name:
                nn.init.trunc_normal_(param, std=0.1)
            elif "fc0.bias" in name:
                nn.init.constant_(param, 1.1)
            elif "fc" in name and "bias" in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        h = torch.relu(self.fc0(x))
        h = self.drop(h)

        h = torch.relu(self.bn1(self.fc1(h)))
        h = torch.relu(self.bn2(self.fc2(h)))
        h = h + identity

        h = torch.relu(self.bn3(self.fc3(h)))
        h = torch.relu(self.bn4(self.fc4(h)))
        h = torch.relu(self.bn5(self.fc5(h)))
        h = h + identity

        h = torch.relu(self.bn6(self.fc6(h)))
        return self.fc_out(h)
