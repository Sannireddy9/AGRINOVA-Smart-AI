"""
AgriSmart AI — Model Architecture
===================================
Builds a transfer-learning classifier from a pre-trained torchvision
backbone.  The final fully-connected head is replaced with a new one
whose output size equals the number of discovered dataset classes.

Supported backbones (all ImageNet-pre-trained):
  resnet50, resnet34, resnet18, efficientnet_b0, mobilenet_v3_small

Adding a new backbone only requires extending ``_BACKBONE_REGISTRY``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn
import torch.nn as nn
from torchvision import models

from model import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Backbone registry
# ──────────────────────────────────────────────
# Each entry maps a config name to:
#   (constructor, weights_enum, classifier_attr, in_features_fn)
#
# * constructor      — torchvision model factory
# * weights_enum     — the "DEFAULT" weights enum for pretrained loading
# * classifier_attr  — attribute name of the head (e.g. "fc" or "classifier")
# * in_features_fn   — callable(model) → int giving the head's input dim

def _resnet_in_features(m: Any) -> int:
    fc: Any = getattr(m, "fc")
    return int(fc.in_features)


def _efficientnet_in_features(m: Any) -> int:
    classifier: Any = getattr(m, "classifier")
    return int(classifier[1].in_features)


def _mobilenet_in_features(m: Any) -> int:
    classifier: Any = getattr(m, "classifier")
    return int(classifier[3].in_features)


_BACKBONE_REGISTRY: dict[str, tuple] = {
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.DEFAULT,
        "fc",
        _resnet_in_features,
    ),
    "resnet34": (
        models.resnet34,
        models.ResNet34_Weights.DEFAULT,
        "fc",
        _resnet_in_features,
    ),
    "resnet50": (
        models.resnet50,
        models.ResNet50_Weights.DEFAULT,
        "fc",
        _resnet_in_features,
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        "classifier",
        _efficientnet_in_features,
    ),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.DEFAULT,
        "classifier",
        _mobilenet_in_features,
    ),
}


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────

def build_model(
    num_classes: int,
    backbone_name: str = config.MODEL_NAME,
    pretrained: bool = config.PRETRAINED,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a transfer-learning classifier.

    Steps:
      1. Load a pre-trained backbone from torchvision.
      2. Optionally freeze all backbone parameters.
      3. Replace the classification head with a new ``nn.Linear`` whose
         output size equals *num_classes*.

    Args:
        num_classes: Number of output classes (auto-discovered from data).
        backbone_name: Key into ``_BACKBONE_REGISTRY``.
        pretrained: If True, load ImageNet-pre-trained weights.
        freeze_backbone: If True, freeze all backbone layers so only the
            new head is trained (useful for fast fine-tuning).

    Returns:
        A ``torch.nn.Module`` ready for training.

    Raises:
        ValueError: If *backbone_name* is not in the registry.
    """
    if backbone_name not in _BACKBONE_REGISTRY:
        supported = ", ".join(sorted(_BACKBONE_REGISTRY))
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Supported: {supported}"
        )

    constructor, weights_enum, cls_attr, in_feat_fn = _BACKBONE_REGISTRY[
        backbone_name
    ]

    # 1. Load backbone
    weights = weights_enum if pretrained else None
    model = constructor(weights=weights)
    logger.info(
        "Loaded backbone '%s' (pretrained=%s)", backbone_name, pretrained
    )

    # 2. Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all backbone parameters")

    # 3. Replace classification head
    in_features = in_feat_fn(model)

    if cls_attr == "fc":
        # ResNet-family: single nn.Linear
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif cls_attr == "classifier":
        # EfficientNet / MobileNet: nn.Sequential classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise RuntimeError(f"Unsupported classifier attribute: {cls_attr}")

    logger.info(
        "Replaced head: %d → %d classes (dropout=0.3)",
        in_features,
        num_classes,
    )
    return model


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only parameters that require gradients."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
