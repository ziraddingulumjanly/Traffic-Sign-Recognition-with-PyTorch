"""German Traffic Sign Recognition package.

Author: Ziraddin Gulumjanli, 2026

This package contains a compact, production-style PyTorch pipeline for the
GTSRB traffic sign classification task: data loading, model building,
training, evaluation, prediction, plotting, and FastAPI serving.
"""

from gtsrb_classifier.config import CLASS_NAMES, TrainingConfig

__all__ = ["CLASS_NAMES", "TrainingConfig"]
