"""Matplotlib plotting utilities for the GTSRB project.

Author: Ziraddin Gulumjanli, 2026

The plotting style follows Ziraddin's Matplotlib recipe preference: high-DPI
figures, serif text, Computer Modern math fonts, mathematical labels, clean
grids, and reusable plotting functions that can be imported from scripts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def apply_plot_style() -> None:
    """Apply a clean high-resolution Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_or_show(path: str | Path | None) -> None:
    """Save the current figure if a path is provided; otherwise show it."""
    if path is None:
        plt.show()
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: dict[str, list[float]], save_path: str | Path | None = None) -> None:
    """Plot training and validation loss/accuracy curves."""
    apply_plot_style()
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label=r"$L_{train}$")
    axes[0].plot(epochs, history["val_loss"], marker="s", label=r"$L_{val}$")
    axes[0].set_title(r"Cross-Entropy Loss")
    axes[0].set_xlabel(r"Epoch")
    axes[0].set_ylabel(r"Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label=r"$Acc_{train}$")
    axes[1].plot(epochs, history["val_accuracy"], marker="s", label=r"$Acc_{val}$")
    axes[1].set_title(r"Classification Accuracy")
    axes[1].set_xlabel(r"Epoch")
    axes[1].set_ylabel(r"Accuracy")
    axes[1].legend()

    fig.tight_layout()
    _save_or_show(save_path)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix with optional row normalization."""
    apply_plot_style()
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(r"Normalized Confusion Matrix" if normalize else r"Confusion Matrix")
    ax.set_xlabel(r"Predicted Class")
    ax.set_ylabel(r"True Class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(np.arange(len(class_names)), rotation=90, fontsize=6)
    ax.set_yticklabels(np.arange(len(class_names)), fontsize=6)
    fig.tight_layout()
    _save_or_show(save_path)


def plot_class_distribution(
    labels: Sequence[int],
    class_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Plot the number of samples per class."""
    apply_plot_style()
    counts = np.bincount(np.asarray(labels), minlength=len(class_names))
    x_values = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x_values, counts)
    ax.set_title(r"GTSRB Class Distribution")
    ax.set_xlabel(r"Class Index")
    ax.set_ylabel(r"Number of Images")
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=90, fontsize=7)
    fig.tight_layout()
    _save_or_show(save_path)
