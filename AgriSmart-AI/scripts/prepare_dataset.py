#!/usr/bin/env python
"""
AgriSmart AI — Dataset Preparation Script
===========================================
Scans the PlantVillage dataset directory, validates every image,
creates a reproducible train/validation split, and saves a
class-to-index mapping as JSON.

This script does NOT train anything.  It only inspects and prepares
the data so you can verify everything is correct before training.

Usage
-----
    python scripts/prepare_dataset.py                      # defaults
    python scripts/prepare_dataset.py --dataset-root PATH  # custom path
    python scripts/prepare_dataset.py --no-verify           # skip image check
    python scripts/prepare_dataset.py --val-ratio 0.15      # 15 % val

Output
------
* ``data/processed/class_index.json``   — class-name → integer mapping
* ``report/corrupted_images.log``       — list of corrupted files (if any)
* A summary table printed to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Ensure the project root is on sys.path so we can import `model.*`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import config
from model.dataset import (
    build_class_to_index,
    collect_image_paths,
    discover_classes,
    save_class_index,
    split_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _print_header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_class_distribution(
    class_names: list[str],
    labels: list[int],
    class_to_idx: dict[str, int],
) -> None:
    """Print a per-class image count table."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    counts = Counter(labels)

    print(f"\n{'Idx':<5} {'Class Name':<45} {'Images':>7}")
    print("-" * 60)
    for idx in sorted(counts):
        name = idx_to_class[idx]
        print(f"{idx:<5} {name:<45} {counts[idx]:>7,}")
    print("-" * 60)
    print(f"{'':5} {'TOTAL':<45} {len(labels):>7,}")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def prepare(
    dataset_root: Path,
    val_ratio: float,
    seed: int,
    verify: bool,
) -> None:
    _print_header("AgriSmart AI — Dataset Preparation")

    # 1. Discover classes ──────────────────────
    logger.info("Scanning dataset root: %s", dataset_root)
    class_names = discover_classes(dataset_root)
    class_to_idx = build_class_to_index(class_names)

    print(f"\n✅  Discovered {len(class_names)} classes")

    # 2. Collect & validate images ─────────────
    logger.info("Collecting images (verify=%s) …", verify)
    image_paths, labels, corrupted = collect_image_paths(
        dataset_root, class_names, verify=verify
    )

    print(f"✅  Valid images : {len(image_paths):,}")
    if corrupted:
        print(f"⚠️  Corrupted    : {len(corrupted):,}")
        # Persist the corrupted list
        config.CORRUPTED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.CORRUPTED_LOG_FILE, "w", encoding="utf-8") as f:
            for p in corrupted:
                f.write(f"{p}\n")
        logger.info(
            "Corrupted image log saved to %s", config.CORRUPTED_LOG_FILE
        )

    # 3. Per-class distribution ────────────────
    _print_class_distribution(class_names, labels, class_to_idx)

    # 4. Train / validation split ──────────────
    train_paths, train_labels, val_paths, val_labels = split_dataset(
        image_paths, labels, val_ratio=val_ratio, seed=seed
    )

    print(f"\n📊  Split (seed={seed}, val_ratio={val_ratio}):")
    print(f"    Train      : {len(train_paths):>7,} images")
    print(f"    Validation : {len(val_paths):>7,} images")

    # 5. Save class→index mapping ──────────────
    save_path = save_class_index(class_to_idx)
    print(f"\n💾  Class index saved to: {save_path}")

    _print_header("Preparation Complete")
    print("  The dataset is ready.  You can now train a model.\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the PlantVillage dataset for AgriSmart AI.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config.DATASET_ROOT,
        help=(
            "Path to the dataset root directory.  "
            f"Default: {config.DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=config.VALIDATION_SPLIT,
        help=f"Validation split ratio.  Default: {config.VALIDATION_SPLIT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for reproducibility.  Default: {config.RANDOM_SEED}",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip image integrity validation (faster, less safe).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        dataset_root=args.dataset_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        verify=not args.no_verify,
    )
