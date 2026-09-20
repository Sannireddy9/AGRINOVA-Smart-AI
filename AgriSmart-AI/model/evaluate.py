"""
AgriSmart AI — Model Evaluation Module
========================================
Evaluates a trained model on a DataLoader and computes per-epoch
validation metrics (loss, accuracy, macro-F1).  Also provides a
full classification report generator for end-of-training analysis.

All metric computation lives here so that both train.py (per-epoch
validation) and standalone evaluation scripts use the same logic.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from model import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Per-epoch validation pass
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one full validation pass and return aggregated metrics.

    Args:
        model: Trained model (set to eval mode internally).
        dataloader: Validation DataLoader.
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss``).
        device: Target device (CPU / CUDA).

    Returns:
        ``{"val_loss": float, "val_accuracy": float, "val_macro_f1": float}``
    """
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0
    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = float(running_loss / max(num_samples, 1))
    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
    }


# ──────────────────────────────────────────────
# Full evaluation (classification report + CM)
# ──────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    save_dir: Path | str | None = None,
) -> dict:
    """Run a complete evaluation and optionally save report artifacts.

    Args:
        model: Trained model.
        dataloader: Evaluation DataLoader.
        device: Target device.
        class_names: Optional list of class names for the report.
        save_dir: Directory to save reports to (defaults to ``config.REPORT_DIR``).

    Returns:
        Dict containing ``accuracy``, ``macro_f1``, ``classification_report``
        (as string), and ``confusion_matrix`` (as nested list).
    """
    model.eval()
    save_dir = Path(save_dir or config.REPORT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    report_str = str(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    ))
    cm = confusion_matrix(all_labels, all_preds)

    # Save text report
    report_path = save_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy  : {accuracy:.4f}\n")
        f.write(f"Macro F1  : {macro_f1:.4f}\n\n")
        f.write(report_str)
    logger.info("Classification report saved to %s", report_path)

    # Save confusion matrix as JSON
    cm_path = save_dir / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({"matrix": cm.tolist(), "class_names": class_names}, f, indent=2)
    logger.info("Confusion matrix saved to %s", cm_path)

    # Plot confusion matrix
    _plot_confusion_matrix(cm, class_names, save_dir / "confusion_matrix.png")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report_str,
        "confusion_matrix": cm.tolist(),
    }


# ──────────────────────────────────────────────
# Training history plots
# ──────────────────────────────────────────────

def plot_training_history(
    history: dict[str, list[float]],
    save_dir: Path | str | None = None,
) -> None:
    """Save loss and metric curves from training history.

    Args:
        history: Dict with keys like ``train_loss``, ``val_loss``,
            ``val_accuracy``, ``val_macro_f1`` mapping to per-epoch lists.
        save_dir: Output directory.  Defaults to ``config.REPORT_DIR``.
    """
    save_dir = Path(save_dir or config.REPORT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # ── Loss curve ──
    if "train_loss" in history and "val_loss" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Loss curve saved")

    # ── Accuracy & F1 curve ──
    if "val_accuracy" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
        if "val_macro_f1" in history:
            ax.plot(epochs, history["val_macro_f1"], label="Val Macro F1", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Accuracy & Macro F1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(save_dir / "metrics_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Metrics curve saved")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None,
    save_path: Path,
) -> None:
    """Render and save a confusion matrix heatmap."""
    n = cm.shape[0]
    figsize = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names and n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix plot saved to %s", save_path)
