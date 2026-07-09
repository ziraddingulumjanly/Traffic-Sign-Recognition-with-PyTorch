"""Training and evaluation loops for the GTSRB classifier.

Author: Ziraddin Gulumjanli, 2026

The functions here follow the classic PyTorch example style: explicit model
mode, explicit device movement, no gradients during evaluation, and clear
return values for logging and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and predictions returned by an evaluation pass."""

    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray

    def to_metrics_dict(self) -> dict[str, float]:
        """Return scalar metrics only."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(data_loader, desc="train", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = accuracy_score(y_true, y_pred)
    return {"loss": float(epoch_loss), "accuracy": float(epoch_accuracy)}


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate the model without gradients and return full metrics."""
    model.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        progress = tqdm(data_loader, desc="eval", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return EvaluationResult(
        loss=float(running_loss / len(data_loader.dataset)),
        accuracy=float(accuracy_score(y_true_array, y_pred_array)),
        macro_f1=float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
        y_true=y_true_array,
        y_pred=y_pred_array,
    )


def make_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Create a scikit-learn classification report as a dictionary."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
