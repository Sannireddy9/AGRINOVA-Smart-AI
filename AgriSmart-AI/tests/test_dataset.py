"""
Tests for model.dataset — class discovery, image validation,
splitting, and transforms.

Uses a tiny synthetic dataset created in a temp directory so we
never depend on real PlantVillage data being present.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from model.dataset import (
    build_class_to_index,
    collect_image_paths,
    discover_classes,
    get_transforms,
    is_valid_image,
    save_class_index,
    split_dataset,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def mini_dataset(tmp_path: Path) -> Path:
    """Create a tiny PlantVillage-style directory with 3 classes."""
    classes = ["ClassA", "ClassB", "ClassC"]
    images_per_class = 10
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(images_per_class):
            img = Image.new("RGB", (64, 64), color=(i * 25, 100, 50))
            img.save(cls_dir / f"img_{i:03d}.jpg")
    return tmp_path


@pytest.fixture()
def dataset_with_corrupt(mini_dataset: Path) -> Path:
    """Add a corrupted file to the mini dataset."""
    corrupt_file = mini_dataset / "ClassA" / "bad_image.jpg"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE")
    return mini_dataset


# ──────────────────────────────────────────────
# Tests — class discovery
# ──────────────────────────────────────────────

def test_discover_classes(mini_dataset: Path):
    classes = discover_classes(mini_dataset)
    assert classes == ["ClassA", "ClassB", "ClassC"]


def test_discover_classes_missing_dir():
    with pytest.raises(FileNotFoundError):
        discover_classes(Path("/nonexistent/path"))


def test_discover_classes_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        discover_classes(tmp_path)


def test_build_class_to_index():
    mapping = build_class_to_index(["Alpha", "Beta", "Gamma"])
    assert mapping == {"Alpha": 0, "Beta": 1, "Gamma": 2}


# ──────────────────────────────────────────────
# Tests — image validation
# ──────────────────────────────────────────────

def test_is_valid_image_good(mini_dataset: Path):
    good = mini_dataset / "ClassA" / "img_000.jpg"
    assert is_valid_image(good) is True


def test_is_valid_image_corrupted(dataset_with_corrupt: Path):
    bad = dataset_with_corrupt / "ClassA" / "bad_image.jpg"
    assert is_valid_image(bad) is False


# ──────────────────────────────────────────────
# Tests — collection
# ──────────────────────────────────────────────

def test_collect_image_paths(mini_dataset: Path):
    classes = discover_classes(mini_dataset)
    paths, labels, corrupted = collect_image_paths(
        mini_dataset, classes, verify=True
    )
    assert len(paths) == 30  # 3 classes × 10
    assert len(labels) == 30
    assert len(corrupted) == 0


def test_collect_detects_corruption(dataset_with_corrupt: Path):
    classes = discover_classes(dataset_with_corrupt)
    paths, labels, corrupted = collect_image_paths(
        dataset_with_corrupt, classes, verify=True
    )
    assert len(corrupted) == 1
    assert "bad_image" in corrupted[0].name
    assert len(paths) == 30  # the good ones only


# ──────────────────────────────────────────────
# Tests — splitting
# ──────────────────────────────────────────────

def test_split_no_leakage(mini_dataset: Path):
    classes = discover_classes(mini_dataset)
    paths, labels, _ = collect_image_paths(mini_dataset, classes, verify=False)

    train_p, train_l, val_p, val_l = split_dataset(
        paths, labels, val_ratio=0.2, seed=42
    )
    # No overlap
    assert set(train_p).isdisjoint(set(val_p))
    # All accounted for
    assert len(train_p) + len(val_p) == len(paths)


def test_split_reproducible(mini_dataset: Path):
    classes = discover_classes(mini_dataset)
    paths, labels, _ = collect_image_paths(mini_dataset, classes, verify=False)

    split_a = split_dataset(paths, labels, seed=42)
    split_b = split_dataset(paths, labels, seed=42)
    assert split_a[0] == split_b[0]  # same train paths
    assert split_a[2] == split_b[2]  # same val paths


# ──────────────────────────────────────────────
# Tests — class index persistence
# ──────────────────────────────────────────────

def test_save_and_load_class_index(tmp_path: Path):
    mapping = {"A": 0, "B": 1, "C": 2}
    out = save_class_index(mapping, tmp_path / "class_index.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == {"A": 0, "B": 1, "C": 2}


# ──────────────────────────────────────────────
# Tests — transforms
# ──────────────────────────────────────────────

def test_get_transforms_returns_compose():
    train_tf = get_transforms("train")
    val_tf = get_transforms("val")
    assert train_tf is not None
    assert val_tf is not None
    # Train pipeline has more ops than val
    assert len(train_tf.transforms) > len(val_tf.transforms)
