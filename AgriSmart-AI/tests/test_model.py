"""
Tests for model architecture, checkpoint save/load, metadata
preservation, prediction pipeline, and class-count warnings.

Uses synthetic data and tiny models so tests run fast without
any real dataset or GPU.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from model import config
from model.dataset import build_class_to_index, discover_classes
from model.model import build_model, count_parameters
from model.predict import (
    get_inference_transform,
    load_and_preprocess_image,
    load_checkpoint,
    load_model,
    predict,
)
from model.train import compute_class_weights, save_checkpoint, seed_everything


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def tiny_model():
    """Build a small resnet18 model with 4 classes."""
    return build_model(num_classes=4, backbone_name="resnet18", pretrained=False)


@pytest.fixture()
def class_to_idx():
    return {"ClassA": 0, "ClassB": 1, "ClassC": 2, "ClassD": 3}


@pytest.fixture()
def saved_checkpoint(tmp_path: Path, tiny_model, class_to_idx):
    """Save a self-describing checkpoint and return its path."""
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        model=tiny_model,
        optimizer=optimizer,
        epoch=5,
        metrics={"val_loss": 0.5, "val_accuracy": 0.85, "val_macro_f1": 0.80},
        path=ckpt_path,
        backbone_name="resnet18",
        num_classes=4,
        class_to_idx=class_to_idx,
    )
    return ckpt_path


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a valid test image."""
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    path = tmp_path / "test_leaf.jpg"
    img.save(path)
    return path


@pytest.fixture()
def corrupted_image(tmp_path: Path) -> Path:
    """Create a corrupted image file."""
    path = tmp_path / "bad_leaf.jpg"
    path.write_bytes(b"NOT_AN_IMAGE_AT_ALL")
    return path


# ──────────────────────────────────────────────
# Tests — Checkpoint save/load
# ──────────────────────────────────────────────

class TestCheckpointSaveLoad:
    """Verify self-describing checkpoint format."""

    def test_checkpoint_contains_required_fields(self, saved_checkpoint):
        ckpt = torch.load(saved_checkpoint, weights_only=False)
        required = {
            "checkpoint_version",
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "metrics",
            "backbone_name",
            "num_classes",
            "class_to_idx",
            "image_size",
            "imagenet_mean",
            "imagenet_std",
        }
        assert required.issubset(set(ckpt.keys())), (
            f"Missing fields: {required - set(ckpt.keys())}"
        )

    def test_checkpoint_metadata_values(self, saved_checkpoint, class_to_idx):
        ckpt = torch.load(saved_checkpoint, weights_only=False)
        assert ckpt["backbone_name"] == "resnet18"
        assert ckpt["num_classes"] == 4
        assert ckpt["class_to_idx"] == class_to_idx
        assert ckpt["epoch"] == 5
        assert ckpt["checkpoint_version"] == config.CHECKPOINT_VERSION

    def test_checkpoint_image_config(self, saved_checkpoint):
        ckpt = torch.load(saved_checkpoint, weights_only=False)
        assert ckpt["image_size"] == list(config.IMAGE_SIZE)
        assert ckpt["imagenet_mean"] == config.IMAGENET_MEAN
        assert ckpt["imagenet_std"] == config.IMAGENET_STD

    def test_checkpoint_metrics_preserved(self, saved_checkpoint):
        ckpt = torch.load(saved_checkpoint, weights_only=False)
        assert ckpt["metrics"]["val_loss"] == 0.5
        assert ckpt["metrics"]["val_accuracy"] == 0.85
        assert ckpt["metrics"]["val_macro_f1"] == 0.80

    def test_load_checkpoint_validates_fields(self, tmp_path):
        """A checkpoint missing required fields should raise KeyError."""
        bad_path = tmp_path / "bad.pt"
        torch.save({"epoch": 1, "model_state_dict": {}}, bad_path)
        with pytest.raises(KeyError, match="missing required metadata"):
            load_checkpoint(bad_path)

    def test_load_checkpoint_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/checkpoint.pt")


# ──────────────────────────────────────────────
# Tests — Model reconstruction from checkpoint
# ──────────────────────────────────────────────

class TestModelReconstruction:
    """Verify model can be rebuilt from checkpoint metadata."""

    def test_load_model_returns_eval_mode(self, saved_checkpoint):
        model, class_to_idx, image_size = load_model(saved_checkpoint)
        assert not model.training, "Model should be in eval mode"

    def test_load_model_correct_output_shape(self, saved_checkpoint):
        model, _, image_size = load_model(saved_checkpoint)
        x = torch.randn(1, 3, *image_size)
        out = model(x)
        assert out.shape == (1, 4), f"Expected (1, 4), got {out.shape}"

    def test_load_model_class_mapping(self, saved_checkpoint, class_to_idx):
        _, loaded_mapping, _ = load_model(saved_checkpoint)
        assert loaded_mapping == class_to_idx

    def test_load_model_weights_match(self, saved_checkpoint, tiny_model):
        """Reconstructed model should have identical weights."""
        loaded_model, _, _ = load_model(saved_checkpoint)
        for (n1, p1), (n2, p2) in zip(
            tiny_model.state_dict().items(),
            loaded_model.state_dict().items(),
        ):
            assert n1 == n2
            assert torch.equal(p1.cpu(), p2.cpu()), f"Weight mismatch: {n1}"


# ──────────────────────────────────────────────
# Tests — Dynamic class count
# ──────────────────────────────────────────────

class TestDynamicClassCount:
    """Verify the model adapts to any number of classes."""

    @pytest.mark.parametrize("n_classes", [3, 10, 15, 20, 38, 50])
    def test_model_adapts_to_any_class_count(self, n_classes):
        model = build_model(
            num_classes=n_classes, backbone_name="resnet18", pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, n_classes)

    def test_class_count_from_directory(self, tmp_path):
        """discover_classes returns exactly the folders present."""
        for name in ["Alpha", "Beta", "Gamma"]:
            (tmp_path / name).mkdir()
        classes = discover_classes(tmp_path)
        assert len(classes) == 3
        assert classes == ["Alpha", "Beta", "Gamma"]

    def test_class_to_idx_consistent(self, tmp_path):
        """class_to_idx must be deterministic across calls."""
        for name in ["Z_class", "A_class", "M_class"]:
            (tmp_path / name).mkdir()
        classes = discover_classes(tmp_path)
        mapping_1 = build_class_to_index(classes)
        mapping_2 = build_class_to_index(classes)
        assert mapping_1 == mapping_2
        # Sorted order: A_class=0, M_class=1, Z_class=2
        assert mapping_1["A_class"] == 0
        assert mapping_1["M_class"] == 1
        assert mapping_1["Z_class"] == 2


# ──────────────────────────────────────────────
# Tests — Prediction pipeline
# ──────────────────────────────────────────────

class TestPrediction:
    """Verify the end-to-end prediction pipeline."""

    def test_predict_returns_expected_keys(self, saved_checkpoint, sample_image):
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
            device="cpu",
            top_k=3,
        )
        assert "predicted_class" in result
        assert "confidence" in result
        assert "top_k" in result
        assert "image_path" in result

    def test_predict_class_is_valid(self, saved_checkpoint, sample_image, class_to_idx):
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
        )
        assert result["predicted_class"] in class_to_idx

    def test_predict_confidence_range(self, saved_checkpoint, sample_image):
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
        )
        assert 0.0 <= result["confidence"] <= 1.0
        for pred in result["top_k"]:
            assert 0.0 <= pred["confidence"] <= 1.0

    def test_predict_top_k_length(self, saved_checkpoint, sample_image):
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
            top_k=2,
        )
        assert len(result["top_k"]) == 2

    def test_predict_top_k_capped_at_num_classes(
        self, saved_checkpoint, sample_image
    ):
        """top_k cannot exceed the number of classes."""
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
            top_k=100,  # more than 4 classes
        )
        assert len(result["top_k"]) == 4  # capped at num_classes


# ──────────────────────────────────────────────
# Tests — Invalid image handling
# ──────────────────────────────────────────────

class TestInvalidImageHandling:
    """Verify graceful handling of bad inputs."""

    def test_predict_missing_image(self, saved_checkpoint):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            predict(
                image_path="/nonexistent/image.jpg",
                checkpoint_path=saved_checkpoint,
            )

    def test_predict_corrupted_image(self, saved_checkpoint, corrupted_image):
        with pytest.raises(ValueError, match="Cannot open"):
            predict(
                image_path=corrupted_image,
                checkpoint_path=saved_checkpoint,
            )

    def test_predict_missing_checkpoint(self, sample_image):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            predict(
                image_path=sample_image,
                checkpoint_path="/nonexistent/model.pt",
            )

    def test_preprocess_missing_image(self):
        transform = get_inference_transform((224, 224))
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_image("/no/such/file.jpg", transform)


# ──────────────────────────────────────────────
# Tests — Class mapping consistency
# ──────────────────────────────────────────────

class TestClassMappingConsistency:
    """Verify the class mapping roundtrips through checkpoint save/load."""

    def test_mapping_survives_checkpoint_roundtrip(
        self, saved_checkpoint, class_to_idx
    ):
        ckpt = torch.load(saved_checkpoint, weights_only=False)
        assert ckpt["class_to_idx"] == class_to_idx

    def test_idx_to_class_invertible(self, class_to_idx):
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        for name, idx in class_to_idx.items():
            assert idx_to_class[idx] == name

    def test_predict_uses_correct_mapping(
        self, saved_checkpoint, sample_image, class_to_idx
    ):
        """Every predicted class name must exist in the original mapping."""
        result = predict(
            image_path=sample_image,
            checkpoint_path=saved_checkpoint,
            top_k=4,
        )
        valid_names = set(class_to_idx.keys())
        for pred in result["top_k"]:
            assert pred["class"] in valid_names


# ──────────────────────────────────────────────
# Tests — Config sanity
# ──────────────────────────────────────────────

class TestConfigSanity:
    """Verify config no longer has dead NUM_CLASSES."""

    def test_no_num_classes_in_config(self):
        assert not hasattr(config, "NUM_CLASSES"), (
            "NUM_CLASSES should have been removed from config.py"
        )

    def test_expected_class_range_exists(self):
        assert hasattr(config, "EXPECTED_CLASS_RANGE")
        lo, hi = config.EXPECTED_CLASS_RANGE
        assert lo < hi
        assert isinstance(lo, int) and isinstance(hi, int)

    def test_checkpoint_version_exists(self):
        assert hasattr(config, "CHECKPOINT_VERSION")
        assert isinstance(config.CHECKPOINT_VERSION, str)


# ──────────────────────────────────────────────
# Tests — Reproducibility
# ──────────────────────────────────────────────

class TestReproducibility:
    """Verify seed_everything produces deterministic results."""

    def test_seed_produces_same_weights(self):
        seed_everything(42)
        m1 = build_model(num_classes=5, backbone_name="resnet18", pretrained=False)
        seed_everything(42)
        m2 = build_model(num_classes=5, backbone_name="resnet18", pretrained=False)
        for (_, p1), (_, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            assert torch.equal(p1, p2)
