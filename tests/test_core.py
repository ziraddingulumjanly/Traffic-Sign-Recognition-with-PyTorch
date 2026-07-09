"""Smoke tests for the GTSRB package.

Author: Ziraddin Gulumjanli, 2026

These tests are intentionally lightweight so GitHub Actions can verify package
imports, model construction, and prediction output shape without downloading
the full dataset.
"""

from __future__ import annotations

import torch
from PIL import Image

from gtsrb_classifier.config import CLASS_NAMES
from gtsrb_classifier.model import build_model
from gtsrb_classifier.predict import predict_pil_image


def test_class_names_have_43_labels() -> None:
    """The GTSRB task should expose exactly 43 class names."""
    assert len(CLASS_NAMES) == 43


def test_cnn_forward_shape() -> None:
    """The compact CNN should output logits with shape [batch, 43]."""
    model = build_model("cnn", num_classes=43, pretrained=False)
    inputs = torch.randn(2, 3, 64, 64)
    outputs = model(inputs)
    assert outputs.shape == (2, 43)


def test_prediction_top_k_shape() -> None:
    """Prediction utility should return the requested number of classes."""
    model = build_model("cnn", num_classes=43, pretrained=False)
    image = Image.new("RGB", (64, 64), color="white")
    predictions = predict_pil_image(image, model=model, top_k=3, device=torch.device("cpu"))
    assert len(predictions) == 3
    assert {"class_id", "class_name", "confidence"}.issubset(predictions[0])
