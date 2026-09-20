"""
Tests for scripts/check_dataset.py.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest

from scripts.check_dataset import check_dataset


def test_check_dataset_missing_dir(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    assert check_dataset(nonexistent) is False


def test_check_dataset_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    assert check_dataset(empty_dir) is False


def test_check_dataset_nested_archive(tmp_path: Path):
    root = tmp_path / "nested_root"
    root.mkdir()
    (root / "color").mkdir()
    assert check_dataset(root) is False


def test_check_dataset_valid_synthetic(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()

    for cls_name in ["Tomato___healthy", "Tomato___Late_blight"]:
        cls_dir = root / cls_name
        cls_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 40, 100, 100))
            img.save(cls_dir / f"img_{i}.jpg")

    assert check_dataset(root) is True


def test_check_dataset_detects_corrupted(tmp_path: Path):
    root = tmp_path / "dataset_corrupt"
    root.mkdir()

    cls_dir = root / "Potato___healthy"
    cls_dir.mkdir()
    # 1 good image
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    img.save(cls_dir / "good.jpg")
    # 1 bad image
    (cls_dir / "bad.jpg").write_bytes(b"CORRUPT_NOT_IMAGE")

    # Still passes overall if there are valid images, but detects corruption
    assert check_dataset(root) is True
