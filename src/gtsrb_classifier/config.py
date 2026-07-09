"""Configuration objects and label metadata for GTSRB classification.

Author: Ziraddin Gulumjanli, 2026

The values here keep the project reproducible and simple: one dataclass for
training settings, one canonical class-name list, and image-normalization
constants shared by training, evaluation, prediction, and the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES: list[str] = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

IMAGE_NET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_NET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for the GTSRB pipeline."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("models")
    figure_dir: Path = Path("reports/figures")
    model_name: str = "resnet18"
    image_size: int = 64
    num_classes: int = 43
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15
    num_workers: int = 2
    seed: int = 42
    pretrained: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the configuration."""
        raw = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}
