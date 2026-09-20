#!/usr/bin/env python
"""
AgriSmart AI — Dataset Verification Script
===========================================
Verifies that the PlantVillage development dataset is properly installed
and ready for training.

Checks performed:
1. Dataset root directory exists.
2. Detects nested archive structures (e.g. data/raw/PlantVillage/color/...)
   and provides clear corrective instructions.
3. Dynamically discovers class subdirectories (no hardcoded names).
4. Validates image readability and detects corrupted files.
5. Counts images per class and total images.
6. Emits advisory notice regarding SIH 2026 class-count expectations.

Usage:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --dataset-root data/raw/PlantVillage
    python scripts/check_dataset.py --sample 50    # check up to 50 images per class
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Safe encoding handling on Windows consoles
for stream in (sys.stdout, sys.stderr):
    reconf = getattr(stream, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import config
from model.dataset import is_valid_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_dataset(
    dataset_root: Path | str = config.DATASET_ROOT,
    sample_limit: int | None = None,
) -> bool:
    """Verify the dataset directory and print diagnostic report.

    Args:
        dataset_root: Path to the dataset root folder.
        sample_limit: If set, only validate up to this many images per class
            for faster checks. None means validate every image.

    Returns:
        True if the dataset exists, has valid classes, and contains images.
        False otherwise.
    """
    dataset_root = Path(dataset_root)

    print()
    print("=" * 70)
    print("  AgriSmart AI -- Dataset Verification & Health Check")
    print("=" * 70)
    print(f"  Target Path : {dataset_root.resolve()}")
    print("=" * 70)

    # ── Check 1: Root directory existence ──────────────────
    if not dataset_root.exists():
        print("\n[ERROR] Dataset directory does NOT exist!")
        print(f"        Looked for: {dataset_root.resolve()}")
        print("\n  How to fix:")
        print(f"    1. Create directory: mkdir -p {dataset_root}")
        print("    2. Download the PlantVillage dataset (Color RGB format).")
        print(f"    3. Unpack class folders directly into: {dataset_root}")
        print("       Expected layout:")
        print(f"         {dataset_root}/<Class_Name>/image1.jpg")
        print("=" * 70)
        return False

    # ── Check 2: Nested archive detection ──────────────────
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    if not subdirs:
        print("\n[ERROR] Dataset directory is empty! No subdirectories found.")
        print(f"        Path: {dataset_root.resolve()}")
        print("\n  How to fix:")
        print("    Place each disease/healthy class as a sub-folder inside this directory.")
        print("=" * 70)
        return False

    # Check for common nested layouts (e.g., PlantVillage/color/<classes> or PlantVillage/raw/color)
    nested_candidates = ["color", "raw", "segmented", "plantvillage", "plantdisease"]
    if len(subdirs) == 1 and subdirs[0].name.lower() in nested_candidates:
        nested_folder = subdirs[0]
        print(f"\n[WARNING] Detected nested structure: '{nested_folder.name}/'")
        print(f"          Found only 1 subdirectory: {nested_folder}")
        print("          The dataset loader expects class folders directly under the root.")
        print("\n  Recommended solution:")
        print(f"    Either move the contents of '{nested_folder.name}/' up one level, or run:")
        print(f'      python scripts/check_dataset.py --dataset-root "{nested_folder}"')
        print("=" * 70)
        return False

    # Check if a 'color' subfolder exists alongside others
    for d in subdirs:
        if d.name.lower() in ("color", "raw"):
            inner_dirs = [x for x in d.iterdir() if x.is_dir()]
            if len(inner_dirs) > 5:
                print(f"\n[INFO] Found '{d.name}/' with {len(inner_dirs)} subdirectories inside.")
                print(f"       You likely want to use: {d}")
                print(f'       Run: python scripts/check_dataset.py --dataset-root "{d}"')

    # ── Check 3: Dynamic Class Discovery ───────────────────
    class_dirs = sorted([d for d in subdirs if not d.name.startswith(".")])
    num_classes = len(class_dirs)
    print(f"\n[INFO] Discovered {num_classes} class directories.")

    # SIH advisory check
    lo, hi = config.EXPECTED_CLASS_RANGE
    if not (lo <= num_classes <= hi):
        print(f"\n  [NOTICE] Discovered {num_classes} classes.")
        print(f"           SIH 2026 problem statement specifies ~15-20 crop-disease classes + healthy.")
        print("           The organizers' final class list will be provided at kickoff.")
        print("           Full development datasets (e.g. 38 classes) or subsets are supported.")

    # ── Check 4: Image Validation & Per-Class Counts ────────
    valid_exts = config.VALID_IMAGE_EXTENSIONS
    class_counts: dict[str, int] = {}
    class_corrupted: dict[str, int] = {}
    total_valid = 0
    total_corrupted = 0
    total_ignored_files = 0
    all_corrupted_paths: list[Path] = []

    print(f"\n{'#':<4} {'Class Name':<45} {'Images':>8}  {'Status':<10}")
    print("-" * 70)

    for i, cdir in enumerate(class_dirs, 1):
        files = sorted(cdir.iterdir())
        image_files = [f for f in files if f.suffix.lower() in valid_exts]
        non_image_files = [f for f in files if f.suffix.lower() not in valid_exts and f.is_file()]
        total_ignored_files += len(non_image_files)

        files_to_check = image_files[:sample_limit] if sample_limit else image_files
        valid_in_class = 0
        corrupted_in_class = 0

        for img_path in files_to_check:
            if is_valid_image(img_path):
                valid_in_class += 1
            else:
                corrupted_in_class += 1
                all_corrupted_paths.append(img_path)

        if sample_limit and len(image_files) > sample_limit:
            count_display = f"{valid_in_class}/{len(image_files)}*"
            class_counts[cdir.name] = len(image_files)
        else:
            count_display = f"{valid_in_class:,}"
            class_counts[cdir.name] = valid_in_class

        class_corrupted[cdir.name] = corrupted_in_class
        total_valid += valid_in_class
        total_corrupted += corrupted_in_class

        status_str = "[OK]" if corrupted_in_class == 0 and valid_in_class > 0 else (
            "[EMPTY]" if valid_in_class == 0 else f"[BAD: {corrupted_in_class}]"
        )
        print(f"{i:<4} {cdir.name:<45} {count_display:>8}  {status_str:<10}")

    print("-" * 70)
    print(f"{'':4} {'TOTAL DISCOVERED CLASSES':<45} {num_classes:>8}")
    print(f"{'':4} {'TOTAL VALID IMAGES':<45} {total_valid:>8,}")
    if total_corrupted > 0:
        print(f"{'':4} {'TOTAL CORRUPTED IMAGES':<45} {total_corrupted:>8,}")
    if total_ignored_files > 0:
        print(f"{'':4} {'IGNORED NON-IMAGE FILES':<45} {total_ignored_files:>8,}")
    print("=" * 70)

    if sample_limit:
        print(f"* Note: Fast check mode inspected up to {sample_limit} images per class.")

    # ── Check 5: Diagnostic Summary ────────────────────────
    if total_valid == 0:
        print("\n[ERROR] No valid images found in any class directory!")
        print(f"        Allowed extensions: {', '.join(sorted(valid_exts))}")
        return False

    if total_corrupted > 0:
        print(f"\n[WARNING] Found {total_corrupted} corrupted/unreadable image(s).")
        log_path = config.REPORT_DIR / "corrupted_images.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for p in all_corrupted_paths:
                f.write(f"{p}\n")
        print(f"          Corrupted file list written to: {log_path}")

    print("\n[PASSED] Dataset check PASSED!")
    print(f"         - Number of classes : {num_classes}")
    print(f"         - Total valid images: {total_valid:,}")
    print("         - Class discovery   : Completely dynamic")
    print("         - Ready for training: YES (run: python model/train.py)")
    print("=" * 70 + "\n")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and check the PlantVillage dataset for AgriSmart AI.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config.DATASET_ROOT,
        help=f"Path to dataset root. Default: {config.DATASET_ROOT}",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional limit of images to check per class for fast validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = check_dataset(
        dataset_root=args.dataset_root,
        sample_limit=args.sample,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
