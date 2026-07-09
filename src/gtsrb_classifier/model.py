"""Model definitions for German traffic sign classification.

Author: Ziraddin Gulumjanli, 2026

Two model choices are provided on purpose: a compact CNN baseline for learning
and a ResNet18 transfer-learning model for stronger portfolio-grade results.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class TrafficSignCNN(nn.Module):
    """Compact CNN baseline for 43-class traffic sign classification."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        features = self.features(inputs)
        return self.classifier(features)


def build_model(model_name: str, num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    """Create a CNN baseline or ResNet18 model."""
    normalized_name = model_name.lower().strip()

    if normalized_name == "cnn":
        return TrafficSignCNN(num_classes=num_classes)

    if normalized_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    msg = f"Unsupported model_name={model_name!r}. Choose 'cnn' or 'resnet18'."
    raise ValueError(msg)
