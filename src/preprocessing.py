"""
preprocessing.py
-----------------
Splits the raw NEU-CLS dataset (data/raw/NEU-CLS/<class>/*.bmp|jpg) into
stratified train/val/test folders under data/processed/, and defines the
torchvision transform pipelines used for training and evaluation.

Run directly to perform the split:
    python src/preprocessing.py --val-size 0.15 --test-size 0.15
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import transforms

IMG_SIZE = 224  # standard ResNet input size

# ImageNet normalization stats (required for a pretrained ResNet18 backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Augmentations appropriate for surface-defect images.

    Kept deliberately conservative: manufacturing defect texture/orientation
    is meaningful, so we avoid aggressive color jitter or large rotations
    that could distort what actually defines a defect. Horizontal/vertical
    flips and small rotations are safe because a scratch or crack is still
    the same defect type regardless of camera orientation.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """No augmentation — deterministic preprocessing for val/test/inference."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_dataset(
    raw_dir: Path,
    processed_dir: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found. Download NEU-CLS and place class folders "
            f"under {raw_dir} (see README.md section 2)."
        )

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {raw_dir}")

    for split in ("train", "val", "test"):
        for cls_dir in class_dirs:
            (processed_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls_dir in class_dirs:
        images = sorted(
            [p for p in cls_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if len(images) < 5:
            print(f"WARNING: class '{cls_dir.name}' has only {len(images)} images — skipping split safety checks.")

        train_val, test = train_test_split(images, test_size=test_size, random_state=seed)
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)

        for split_name, split_files in (("train", train), ("val", val), ("test", test)):
            dest_dir = processed_dir / split_name / cls_dir.name
            for f in split_files:
                shutil.copy2(f, dest_dir / f.name)

        summary[cls_dir.name] = {"train": len(train), "val": len(val), "test": len(test)}

    print("Split complete:")
    for cls, counts in summary.items():
        print(f"  {cls:20s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw NEU-CLS data into train/val/test")
    parser.add_argument("--raw-dir", default="data/raw/NEU-CLS")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
