"""
AgriSmart AI — Training Pipeline
==================================
End-to-end training script.  Discovers classes from the dataset
directory, builds a transfer-learning model, trains with class-aware
weighting, logs metrics every epoch, and saves the best checkpoint.

Usage
-----
    python model/train.py                           # defaults from config.py
    python model/train.py --epochs 15               # override epochs
    python model/train.py --backbone resnet34       # different backbone
    python model/train.py --freeze-backbone         # only train the head
    python model/train.py --no-verify               # skip image validation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure project root is importable when running as `python model/train.py`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import config
from model.dataset import (  # pyrefly: ignore [missing-import]  # type: ignore[import-not-found]
    build_class_to_index,
    collect_image_paths,
    discover_classes,
    get_dataloaders,
    save_class_index,
    split_dataset,
)
from model.evaluate import full_evaluation, plot_training_history, validate  # pyrefly: ignore [missing-import]  # type: ignore[import-not-found]
from model.model import build_model, count_parameters  # pyrefly: ignore [missing-import]  # type: ignore[import-not-found]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# ──────────────────────────────────────────────
# Class-aware loss weighting
# ──────────────────────────────────────────────

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Weight for class *c* = total_samples / (num_classes × count_c).

    Args:
        labels: List of integer labels in the training set.
        num_classes: Total number of classes.

    Returns:
        A ``torch.FloatTensor`` of shape ``(num_classes,)``.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)  # avoid division by zero
        weights[cls_idx] = total / (num_classes * count)
    return weights


# ──────────────────────────────────────────────
# Single training epoch
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss.

    Args:
        model: The model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser instance.
        device: Target device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    *,
    backbone_name: str,
    num_classes: int,
    class_to_idx: dict[str, int],
    image_size: tuple[int, int] = config.IMAGE_SIZE,
) -> None:
    """Save a self-describing training checkpoint.

    The checkpoint contains everything needed to reconstruct the model
    for inference without relying on external config files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Versioning
            "checkpoint_version": config.CHECKPOINT_VERSION,
            # Training state
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            # Architecture metadata (needed to reconstruct the model)
            "backbone_name": backbone_name,
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            # Input configuration
            "image_size": list(image_size),
            "imagenet_mean": config.IMAGENET_MEAN,
            "imagenet_std": config.IMAGENET_STD,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def save_training_config(
    training_args: dict,
    class_names: list[str],
    param_counts: dict,
    path: Path,
) -> None:
    """Persist the full training configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_args": {k: str(v) for k, v in training_args.items()},
        "num_classes": len(class_names),
        "class_names": class_names,
        "param_counts": param_counts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Training config saved → %s", path)


# ──────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:  # noqa: C901 — acceptable for a pipeline
    """Run the full training pipeline end-to-end."""
    start_time = time.time()

    # ── 0. Reproducibility ────────────────────
    seed_everything(args.seed)

    # ── 1. Device ─────────────────────────────
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # ── 2. Discover classes ───────────────────
    logger.info("Dataset root: %s", args.dataset_root)
    class_names = discover_classes(args.dataset_root)
    class_to_idx = build_class_to_index(class_names)
    num_classes = len(class_names)
    logger.info("Discovered %d classes", num_classes)

    # ── Class-count advisory check (P2) ───────
    lo, hi = config.EXPECTED_CLASS_RANGE
    if not (lo <= num_classes <= hi):
        logger.warning(
            "\n"
            "  ⚠️  CLASS-COUNT NOTICE\n"
            "  Discovered %d classes, but the SIH 2026 problem statement\n"
            "  specifies approximately 15–20 crop-disease classes + healthy.\n"
            "  The organisers' final class list will be provided at kickoff.\n"
            "  The current development dataset may contain a different\n"
            "  number of classes.  Training will proceed normally.\n",
            num_classes,
        )

    # ── 3. Collect images ─────────────────────
    image_paths, labels, corrupted = collect_image_paths(
        args.dataset_root, class_names, verify=args.verify
    )
    logger.info(
        "Images: %d valid, %d corrupted", len(image_paths), len(corrupted)
    )

    # ── 4. Train / val split ──────────────────
    train_paths, train_labels, val_paths, val_labels = split_dataset(
        image_paths, labels, val_ratio=args.val_ratio, seed=args.seed
    )
    logger.info("Train: %d  |  Val: %d", len(train_paths), len(val_paths))

    # ── 5. DataLoaders ────────────────────────
    train_loader, val_loader = get_dataloaders(
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── 6. Build model ────────────────────────
    model = build_model(
        num_classes=num_classes,
        backbone_name=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)
    param_info = count_parameters(model)
    logger.info(
        "Parameters — total: %s  trainable: %s",
        f"{param_info['total']:,}",
        f"{param_info['trainable']:,}",
    )

    # ── 7. Loss, optimiser, scheduler ─────────
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Using class-weighted CrossEntropyLoss")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # ── 8. Save class mapping & config ────────
    save_dir = Path(args.save_dir)
    save_class_index(class_to_idx, save_dir / "class_index.json")
    save_training_config(
        training_args=vars(args),
        class_names=class_names,
        param_counts=param_info,
        path=save_dir / "training_config.json",
    )

    # ── 9. Training loop ─────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    best_f1 = 0.0
    patience_counter = 0

    print()
    print("=" * 70)
    print("  AgriSmart AI — Training Started")
    print(f"  Backbone : {args.backbone}  |  Classes : {num_classes}")
    print(f"  Epochs   : {args.epochs}  |  Batch   : {args.batch_size}")
    print(f"  LR       : {args.lr}  |  Device  : {device}")
    print("=" * 70)
    print()
    header = (
        f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  "
        f"{'Val Acc':>8}  {'Val F1':>7}  {'LR':>10}  {'Status'}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]
        val_f1 = val_metrics["val_macro_f1"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        # Scheduler step (watching macro-F1)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Check for best model
        status = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                save_dir / "best_model.pt",
                backbone_name=args.backbone,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
            )
            status = "★ Best"
        else:
            patience_counter += 1

        # Log epoch
        print(
            f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
            f"{val_acc:>8.4f}  {val_f1:>7.4f}  {current_lr:>10.2e}  {status}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                args.patience,
            )
            print(f"\n⏹  Early stopping at epoch {epoch}")
            break

    save_checkpoint(
        model, optimizer, epoch, val_metrics, save_dir / "last_model.pt",
        backbone_name=args.backbone,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
    )

    # ── 11. Post-training evaluation ──────────
    # Load best weights for final evaluation
    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Loaded best checkpoint (epoch %d) for final evaluation", best_ckpt["epoch"])

    eval_results = full_evaluation(
        model, val_loader, device, class_names=class_names
    )

    # ── 12. Save training curves ──────────────
    plot_training_history(history)

    # Save raw history as JSON
    history_path = config.REPORT_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("  Training Complete")
    print(f"  Best Val Macro F1 : {best_f1:.4f}")
    print(f"  Best checkpoint   : {save_dir / 'best_model.pt'}")
    print(f"  Reports           : {config.REPORT_DIR}")
    print(f"  Elapsed time      : {elapsed / 60:.1f} min")
    print("=" * 70)
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the AgriSmart AI crop-disease classifier.",
    )
    p.add_argument(
        "--dataset-root", type=Path, default=config.DATASET_ROOT,
        help=f"Path to dataset root.  Default: {config.DATASET_ROOT}",
    )
    p.add_argument(
        "--backbone", type=str, default=config.MODEL_NAME,
        help=f"Backbone architecture.  Default: {config.MODEL_NAME}",
    )
    p.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone; only train the classification head.",
    )
    p.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help=f"Max training epochs.  Default: {config.NUM_EPOCHS}",
    )
    p.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size.  Default: {config.BATCH_SIZE}",
    )
    p.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Learning rate.  Default: {config.LEARNING_RATE}",
    )
    p.add_argument(
        "--val-ratio", type=float, default=config.VALIDATION_SPLIT,
        help=f"Validation split ratio.  Default: {config.VALIDATION_SPLIT}",
    )
    p.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (epochs without F1 improvement).  Default: 7",
    )
    p.add_argument(
        "--num-workers", type=int, default=config.NUM_WORKERS,
        help=f"DataLoader workers.  Default: {config.NUM_WORKERS}",
    )
    p.add_argument(
        "--device", type=str, default=config.DEVICE,
        help=f"Device (cuda / cpu).  Default: {config.DEVICE}",
    )
    p.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed.  Default: {config.RANDOM_SEED}",
    )
    p.add_argument(
        "--save-dir", type=Path, default=config.MODEL_SAVE_DIR,
        help=f"Directory to save checkpoints.  Default: {config.MODEL_SAVE_DIR}",
    )
    p.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip image validation (faster startup).",
    )
    p.set_defaults(verify=config.VERIFY_IMAGES)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
