"""General utilities for reproducibility, devices, checkpoints, and JSON I/O.

Author: Ziraddin Gulumjanli, 2026

These helpers are intentionally small and explicit. They make scripts easier
to read without hiding important PyTorch behavior behind a heavy framework.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return the best available PyTorch device: CUDA, MPS, XPU, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return torch.device("xpu")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as an indented JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save model weights and metadata in one checkpoint file."""
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint using safe CPU-compatible defaults."""
    return torch.load(Path(path), map_location=map_location)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
