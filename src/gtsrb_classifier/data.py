"""Dataset, transform, and DataLoader utilities for GTSRB.

Author: Ziraddin Gulumjanli, 2026

The project uses torchvision's GTSRB dataset wrapper so training does not rely
on fragile manual download links. Train/validation splitting is deterministic,
and small subsets can be enabled for fast smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from gtsrb_classifier.config import IMAGE_NET_MEAN, IMAGE_NET_STD, TrainingConfig


@dataclass(frozen=True)
class DataLoaders:
    """Container for train, validation, and test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_transforms(
    image_size: int,
    split: Literal["train", "eval"],
) -> transforms.Compose:
    """Build image transforms for training or evaluation."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )


def _limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Return the full dataset or a deterministic first-N subset."""
    if max_samples is None:
        return dataset
    max_samples = min(max_samples, len(dataset))
    return Subset(dataset, list(range(max_samples)))


def build_datasets(config: TrainingConfig) -> tuple[Dataset, Dataset, Dataset]:
    """Download/load GTSRB and return train, validation, and test datasets."""
    data_dir = Path(config.data_dir)
    train_full = datasets.GTSRB(
        root=data_dir,
        split="train",
        download=True,
        transform=build_transforms(config.image_size, split="train"),
    )
    test_dataset = datasets.GTSRB(
        root=data_dir,
        split="test",
        download=True,
        transform=build_transforms(config.image_size, split="eval"),
    )

    val_size = int(len(train_full) * config.validation_fraction)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=generator)

    train_dataset = _limit_dataset(train_dataset, config.max_train_samples)
    val_dataset = _limit_dataset(val_dataset, config.max_val_samples)
    test_dataset = _limit_dataset(test_dataset, config.max_test_samples)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(config: TrainingConfig) -> DataLoaders:
    """Create train, validation, and test DataLoaders."""
    train_dataset, val_dataset, test_dataset = build_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)
