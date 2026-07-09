"""Single-image inference utilities for GTSRB traffic sign recognition.

Author: Ziraddin Gulumjanli, 2026

The same functions are used by the CLI and FastAPI app, which keeps serving
logic simple and avoids duplicate preprocessing code.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gtsrb_classifier.config import CLASS_NAMES, TrainingConfig
from gtsrb_classifier.data import build_transforms
from gtsrb_classifier.model import build_model
from gtsrb_classifier.utils import get_device, load_checkpoint


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained model and its metadata from a checkpoint."""
    device = device or get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    config = metadata.get("config", {})
    model_name = config.get("model_name", "resnet18")
    num_classes = int(config.get("num_classes", 43))

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


def predict_pil_image(
    image: Image.Image,
    model: torch.nn.Module,
    image_size: int = 64,
    top_k: int = 5,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Predict top-k traffic sign classes from a PIL image."""
    device = device or get_device()
    transform = build_transforms(image_size=image_size, split="eval")
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        scores, indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)))

    return [
        {
            "class_id": int(index),
            "class_name": CLASS_NAMES[int(index)],
            "confidence": float(score),
        }
        for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
    ]


def predict_image_file(
    image_path: str | Path,
    checkpoint_path: str | Path,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Predict top-k classes for an image file path."""
    device = get_device()
    model, metadata = load_trained_model(checkpoint_path, device=device)
    config = metadata.get("config", {})
    image_size = int(config.get("image_size", TrainingConfig().image_size))
    image = Image.open(image_path)
    return predict_pil_image(image, model=model, image_size=image_size, top_k=top_k, device=device)


def predict_image_bytes(
    image_bytes: bytes,
    model: torch.nn.Module,
    image_size: int = 64,
    top_k: int = 5,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Predict top-k classes for image bytes uploaded through an API."""
    image = Image.open(BytesIO(image_bytes))
    return predict_pil_image(image, model=model, image_size=image_size, top_k=top_k, device=device)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict a GTSRB traffic sign image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gtsrb_resnet18_best.pt"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run single-image prediction from the command line."""
    args = parse_args()
    predictions = predict_image_file(args.image, args.checkpoint, top_k=args.top_k)
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
