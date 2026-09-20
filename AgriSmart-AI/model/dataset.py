"""
AgriSmart AI — Dataset Module
==============================
Handles dataset discovery, image validation, train/val splitting,
preprocessing, augmentation, and DataLoader creation.

Design decisions
----------------
* Class names are **auto-discovered** from sub-directory names inside
  ``config.DATASET_ROOT``.  Nothing is hard-coded.
* The train/val split is **stratified** and **reproducible** (fixed seed).
* Images are validated with Pillow's ``Image.verify()`` and a full
  pixel decode pass so truly corrupted files are caught early.
* Augmentation follows best practices for transfer learning on ImageNet-
  pre-trained backbones (resize → random crop → flip → colour jitter →
  normalise).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Image validation
# ──────────────────────────────────────────────

def is_valid_image(path: Path) -> bool:
    """Return True if *path* can be opened and fully decoded as an image.

    Two-stage check:
      1. ``Image.verify()`` — fast header/structure check.
      2. Full pixel load — catches truncated / partially corrupted files.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() can pass for truncated images, so re-open and load pixels
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# Class discovery
# ──────────────────────────────────────────────

def discover_classes(dataset_root: Path | str) -> list[str]:
    """Return a sorted list of class names found as sub-directories.

    Args:
        dataset_root: Top-level dataset directory (e.g. ``data/raw/PlantVillage``).

    Returns:
        Sorted list of class-name strings.

    Raises:
        FileNotFoundError: If *dataset_root* does not exist.
        ValueError: If no class sub-directories are found.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Please download the PlantVillage dataset and place it at:\n"
            f"  {config.DATASET_ROOT}"
        )

    class_dirs = sorted(
        [d.name for d in dataset_root.iterdir() if d.is_dir()]
    )
    if not class_dirs:
        raise ValueError(
            f"No class sub-directories found in {dataset_root}.\n"
            "Expected a PlantVillage-style layout where each folder is a class."
        )
    return class_dirs


def build_class_to_index(class_names: list[str]) -> dict[str, int]:
    """Create a deterministic class-name → integer mapping.

    Args:
        class_names: Sorted list of class-name strings.

    Returns:
        ``{"Apple___healthy": 0, "Apple___scab": 1, ...}``
    """
    return {name: idx for idx, name in enumerate(class_names)}


def save_class_index(
    class_to_idx: dict[str, int],
    output_path: Path | str | None = None,
) -> Path:
    """Persist the class→index mapping as a JSON file.

    Args:
        class_to_idx: Mapping produced by :func:`build_class_to_index`.
        output_path: Destination file.  Defaults to ``config.CLASS_INDEX_FILE``.

    Returns:
        The path the JSON was written to.
    """
    output_path = Path(output_path or config.CLASS_INDEX_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    logger.info("Class index saved to %s", output_path)
    return output_path


def load_class_index(path: Path | str | None = None) -> dict[str, int]:
    """Load a previously saved class→index mapping from JSON.

    Args:
        path: JSON file path.  Defaults to ``config.CLASS_INDEX_FILE``.

    Returns:
        ``{"Apple___healthy": 0, ...}``
    """
    path = Path(path or config.CLASS_INDEX_FILE)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Collect image paths + validate
# ──────────────────────────────────────────────

def collect_image_paths(
    dataset_root: Path | str,
    class_names: list[str],
    *,
    verify: bool = True,
) -> tuple[list[Path], list[int], list[Path]]:
    """Walk the dataset tree and gather (path, label) pairs.

    Args:
        dataset_root: Root directory containing one sub-folder per class.
        class_names: Sorted list of class names (sub-folder names).
        verify: If True, run :func:`is_valid_image` on every file.

    Returns:
        A 3-tuple of:
          - ``image_paths``: list of valid image :class:`Path` objects
          - ``labels``:      matching integer labels
          - ``corrupted``:   list of paths that failed validation
    """
    dataset_root = Path(dataset_root)
    class_to_idx = build_class_to_index(class_names)
    valid_exts = config.VALID_IMAGE_EXTENSIONS

    image_paths: list[Path] = []
    labels: list[int] = []
    corrupted: list[Path] = []

    for cls_name in class_names:
        cls_dir = dataset_root / cls_name
        if not cls_dir.is_dir():
            logger.warning("Expected directory missing: %s", cls_dir)
            continue

        for file_path in sorted(cls_dir.iterdir()):
            if file_path.suffix.lower() not in valid_exts:
                continue

            if verify and not is_valid_image(file_path):
                corrupted.append(file_path)
                logger.warning("Corrupted image skipped: %s", file_path)
                continue

            image_paths.append(file_path)
            labels.append(class_to_idx[cls_name])

    return image_paths, labels, corrupted


# ──────────────────────────────────────────────
# Train / validation split
# ──────────────────────────────────────────────

def split_dataset(
    image_paths: list[Path],
    labels: list[int],
    val_ratio: float = config.VALIDATION_SPLIT,
    seed: int = config.RANDOM_SEED,
    stratify: bool = config.STRATIFIED_SPLIT,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Split images into train and validation sets.

    Args:
        image_paths: All valid image paths.
        labels: Corresponding integer labels.
        val_ratio: Fraction of data reserved for validation.
        seed: Random seed for reproducibility.
        stratify: If True, use stratified splitting to preserve class balance.

    Returns:
        ``(train_paths, train_labels, val_paths, val_labels)``
    """
    stratify_arg = labels if stratify else None
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_arg,
    )
    return train_paths, train_labels, val_paths, val_labels


# ──────────────────────────────────────────────
# Transforms / augmentation
# ──────────────────────────────────────────────

def get_transforms(split: str = "train") -> transforms.Compose:
    """Return image transforms for the given split.

    Training split includes augmentation (random crop, flip, colour jitter).
    Validation split uses a deterministic resize + centre crop.

    All splits normalise with ImageNet statistics to match pre-trained
    backbone expectations.

    Args:
        split: ``"train"`` or ``"val"``.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    img_h, img_w = config.IMAGE_SIZE
    mean = config.IMAGENET_MEAN
    std = config.IMAGENET_STD

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_h + 32, img_w + 32)),
            transforms.RandomResizedCrop((img_h, img_w), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # val / inference
    return transforms.Compose([
        transforms.Resize((img_h + 32, img_w + 32)),
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────

class PlantDiseaseDataset(Dataset):
    """A generic image-folder dataset that reads from a list of paths.

    Unlike ``torchvision.datasets.ImageFolder`` this class works with an
    explicit list of paths + labels so it supports pre-validated, pre-split
    data and avoids re-scanning the filesystem.

    Args:
        image_paths: List of image file paths.
        labels: Matching integer class labels.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloaders(
    train_paths: list[Path],
    train_labels: list[int],
    val_paths: list[Path],
    val_labels: list[int],
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """Build and return train and validation DataLoaders.

    Args:
        train_paths: Training image paths.
        train_labels: Training labels.
        val_paths: Validation image paths.
        val_labels: Validation labels.
        batch_size: Batch size.
        num_workers: Number of worker processes for data loading.

    Returns:
        ``(train_loader, val_loader)``
    """
    train_ds = PlantDiseaseDataset(
        train_paths, train_labels, transform=get_transforms("train")
    )
    val_ds = PlantDiseaseDataset(
        val_paths, val_labels, transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
