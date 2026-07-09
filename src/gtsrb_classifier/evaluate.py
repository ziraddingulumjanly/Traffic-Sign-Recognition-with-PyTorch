"""Evaluate a saved GTSRB checkpoint on the official test split.

Author: Ziraddin Gulumjanli, 2026

Example:
    python -m gtsrb_classifier.evaluate --checkpoint models/gtsrb_resnet18_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from gtsrb_classifier.config import CLASS_NAMES, TrainingConfig
from gtsrb_classifier.data import build_dataloaders
from gtsrb_classifier.engine import evaluate, make_classification_report
from gtsrb_classifier.model import build_model
from gtsrb_classifier.plots import plot_confusion_matrix
from gtsrb_classifier.utils import (
    ensure_dir,
    get_device,
    load_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gtsrb_resnet18_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/test_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint, evaluate it, and save metrics/figures."""
    args = parse_args()
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})
    saved_config = metadata.get("config", {})

    config = TrainingConfig(
        data_dir=args.data_dir,
        figure_dir=args.figure_dir,
        model_name=saved_config.get("model_name", "resnet18"),
        image_size=int(saved_config.get("image_size", 64)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(saved_config.get("seed", 42)),
        pretrained=False,
        max_test_samples=args.max_test_samples,
    )
    seed_everything(config.seed)
    ensure_dir(config.figure_dir)

    dataloaders = build_dataloaders(config)
    model = build_model(config.model_name, num_classes=config.num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, dataloaders.test, criterion, device)
    report = make_classification_report(result.y_true, result.y_pred, CLASS_NAMES)

    payload = {
        "test_metrics": result.to_metrics_dict(),
        "classification_report": report,
        "checkpoint": str(args.checkpoint),
    }
    save_json(payload, args.metrics_path)
    plot_confusion_matrix(
        result.y_true,
        result.y_pred,
        CLASS_NAMES,
        save_path=config.figure_dir / "test_confusion_matrix.png",
        normalize=True,
    )

    print("Test metrics")
    for name, value in result.to_metrics_dict().items():
        print(f"{name}: {value:.4f}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
