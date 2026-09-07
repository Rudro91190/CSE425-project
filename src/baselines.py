"""
Baseline models module.
Implements:
- B1: Majority-class and Random tag predictor
- B2: CNN on mel-spectrograms (2D audio-only baseline)
- B4: MLP on hand-crafted audio features
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineRandom:
    """B1: Predicts tags using class marginal frequencies or random probabilities."""

    def __init__(self, num_classes: int = 20):
        self.num_classes = num_classes
        self.class_priors = np.full(num_classes, 0.5, dtype=np.float32)

    def fit(self, labels: np.ndarray):
        """Fit empirical class priors from training labels."""
        self.class_priors = np.mean(labels, axis=0)

    def predict_proba(self, num_samples: int, mode: str = "prior") -> np.ndarray:
        if mode == "random":
            return np.random.uniform(0.0, 1.0, size=(num_samples, self.num_classes))
        # Broadcast prior probabilities
        return np.tile(self.class_priors, (num_samples, 1))


class MelSpectrogramCNN(nn.Module):
    """
    B2: 2D Convolutional Neural Network operating on log-mel spectrograms.
    Standard audio-only benchmark baseline (no graphs, no text).
    Input: [batch_size, 1, 128, frames]
    """

    def __init__(self, num_classes: int = 20, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, spectrograms: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        feat = self.features(spectrograms)
        logits = self.classifier(feat)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.bce_loss(logits, labels)

        return {"logits": logits, "probs": probs, "loss": loss}


class AudioFeatureMLP(nn.Module):
    """B4: Multi-Layer Perceptron operating on pooled hand-crafted features."""

    def __init__(self, in_features: int = 140, num_classes: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = self.bce_loss(logits, labels)
        return {"logits": logits, "probs": probs, "loss": loss}
