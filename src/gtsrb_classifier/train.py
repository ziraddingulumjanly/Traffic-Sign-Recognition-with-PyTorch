"""Train a GTSRB traffic sign classifier from the command line.

Author: Ziraddin Gulumjanli, 2026

Example:
    python -m gtsrb_classifier.train --model-name resnet18 --epochs 8

Fast smoke test:
    python -m gtsrb_classifier.train --fast
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn

from gtsrb_classifier.config import CLASS_NAMES, TrainingConfig
from gtsrb_classifier.data import build_dataloaders
from gtsrb_classifier.engine import evaluate, train_one_epoch
from gtsrb_classifier.model import build_model
from gtsrb_classifier.plots import plot_training_curves
from gtsrb_classifier.utils import (
    count_trainable_parameters,
    ensure_dir,
    get_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a GTSRB traffic sign classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--model-name", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use small subsets for a quick smoke test.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert CLI arguments into a TrainingConfig."""
    fast_train = 2_000 if args.fast else None
    fast_val = 500 if args.fast else None
    fast_test = 500 if args.fast else None
    fast_epochs = 1 if args.fast else args.epochs

    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=fast_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        max_train_samples=fast_train,
        max_val_samples=fast_val,
        max_test_samples=fast_test,
    )


def main() -> None:
    """Run model training, validation, checkpointing, and curve plotting."""
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)

    ensure_dir(config.output_dir)
    ensure_dir(config.figure_dir)
    device = get_device()
    dataloaders = build_dataloaders(config)

    model = build_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    print(f"Device: {device}")
    print(f"Model: {config.model_name}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }
    best_val_macro_f1 = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_metrics = train_one_epoch(model, dataloaders.train, criterion, optimizer, device)
        val_result = evaluate(model, dataloaders.val, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_result.loss)
        history["val_accuracy"].append(val_result.accuracy)
        history["val_macro_f1"].append(val_result.macro_f1)
        history["val_weighted_f1"].append(val_result.weighted_f1)

        print(
            "train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}".format(
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                val_loss=val_result.loss,
                val_acc=val_result.accuracy,
                val_macro_f1=val_result.macro_f1,
            )
        )

        if val_result.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_result.macro_f1
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    checkpoint_path = config.output_dir / f"gtsrb_{config.model_name}_best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        metadata={
            "config": config.to_dict(),
            "class_names": CLASS_NAMES,
            "best_val_macro_f1": best_val_macro_f1,
        },
    )
    save_json({"history": history, "config": config.to_dict()}, config.output_dir / "training_history.json")
    plot_training_curves(history, config.figure_dir / "training_curves.png")

    print(f"\nSaved best checkpoint: {checkpoint_path}")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
