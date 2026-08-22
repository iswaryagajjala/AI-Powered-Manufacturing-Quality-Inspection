"""
dataset.py
----------
PyTorch Dataset for the manufacturing surface-defect classification task.

Expects images laid out as:

    data/processed/<split>/<class_name>/*.jpg

where <split> is one of {train, val, test} and <class_name> is one of the
defect categories (e.g. crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches).

This mirrors the standard torchvision.datasets.ImageFolder layout, but is
implemented explicitly (rather than just calling ImageFolder) so that:
  - class list ordering is deterministic and saved alongside the model
  - we can plug in OpenCV-based preprocessing/validation before the PIL/
    torchvision transform pipeline
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def discover_classes(processed_dir: Path, split: str = "train") -> list[str]:
    """Return a sorted, deterministic list of class names found in a split."""
    split_dir = processed_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected processed split at {split_dir}. "
            "Run `python src/preprocessing.py` first."
        )
    classes = sorted(
        [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    if not classes:
        raise ValueError(f"No class subfolders found in {split_dir}")
    return classes


def save_class_mapping(classes: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)


def load_class_mapping(path: Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


class DefectDataset(Dataset):
    """Loads (image, label) pairs for one split of the defect dataset.

    A lightweight OpenCV validity check (`_is_valid_image`) is used to skip
    corrupted files instead of crashing training on a bad file — a realistic
    concern with real-world manufacturing camera captures.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        classes: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.classes = classes or discover_classes(self.root_dir, split)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.split_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    if self._is_valid_image(img_path):
                        self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No valid images found under {self.split_dir}")

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        """Use OpenCV to reject unreadable/corrupted images up front."""
        img = cv2.imread(str(path))
        return img is not None and img.size > 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        # Loaded via PIL (RGB) so torchvision transforms behave normally;
        # OpenCV was already used above purely for the validity check.
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts
