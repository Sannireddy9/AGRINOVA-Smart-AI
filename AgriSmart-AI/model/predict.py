"""
AgriSmart AI — Prediction / Inference Module
==============================================
Loads a trained checkpoint (which is self-describing) and runs
inference on new images.

The checkpoint contains all metadata needed to reconstruct the
model — backbone name, num_classes, class_to_idx, image_size,
and normalisation statistics.  No external config is required.

Usage
-----
    python model/predict.py --image path/to/leaf.jpg
    python model/predict.py --image path/to/leaf.jpg --checkpoint model/checkpoints/best_model.pt
    python model/predict.py --image path/to/leaf.jpg --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure project root is importable when running as `python model/predict.py`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import config
from model.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict:
    """Load and validate a self-describing checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to map tensors to.

    Returns:
        The full checkpoint dict.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If required metadata fields are missing.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Validate required fields
    required = {
        "model_state_dict", "backbone_name", "num_classes", "class_to_idx",
    }
    missing = required - set(ckpt.keys())
    if missing:
        raise KeyError(
            f"Checkpoint is missing required metadata fields: {missing}. "
            "This checkpoint may have been saved with an older format. "
            "Please retrain or update the checkpoint."
        )

    logger.info(
        "Loaded checkpoint: epoch=%s  backbone=%s  classes=%d",
        ckpt.get("epoch", "?"),
        ckpt["backbone_name"],
        ckpt["num_classes"],
    )
    return ckpt


# ──────────────────────────────────────────────
# Model reconstruction
# ──────────────────────────────────────────────

def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, int], tuple[int, int]]:
    """Reconstruct a trained model from a self-describing checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        device: Target device.

    Returns:
        A 3-tuple of:
          - The model in eval mode with weights loaded.
          - ``class_to_idx`` mapping (``{"Apple___healthy": 0, ...}``).
          - ``image_size`` tuple (height, width).
    """
    device = torch.device(device)
    ckpt = load_checkpoint(checkpoint_path, device=device)

    # Reconstruct architecture from metadata
    model = build_model(
        num_classes=ckpt["num_classes"],
        backbone_name=ckpt["backbone_name"],
        pretrained=False,  # no need to download weights; we load our own
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    image_size = tuple(ckpt.get("image_size", config.IMAGE_SIZE))

    return model, class_to_idx, image_size


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────

def get_inference_transform(
    image_size: tuple[int, int],
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Build the deterministic inference transform pipeline.

    Matches the validation transforms used during training.

    Args:
        image_size: ``(height, width)`` for the model input.
        mean: Channel means for normalisation.
        std: Channel stds for normalisation.

    Returns:
        A torchvision Compose pipeline.
    """
    mean = mean or config.IMAGENET_MEAN
    std = std or config.IMAGENET_STD
    h, w = image_size
    return transforms.Compose([
        transforms.Resize((h + 32, w + 32)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_and_preprocess_image(
    image_path: str | Path,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Load a single image and apply the inference transform.

    Args:
        image_path: Path to the image file.
        transform: The preprocessing pipeline.

    Returns:
        A ``(1, C, H, W)`` tensor ready for the model.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Cannot open '{image_path}' as an image: {exc}"
        ) from exc

    tensor = transform(img)
    return tensor.unsqueeze(0)  # add batch dimension


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

@torch.no_grad()
def predict(
    image_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    top_k: int = 3,
) -> dict:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image.
        checkpoint_path: Path to a self-describing checkpoint.
            Defaults to ``config.MODEL_SAVE_DIR / "best_model.pt"``.
        device: ``"cpu"`` or ``"cuda"``.
        top_k: Number of top predictions to return.

    Returns:
        Dict with keys:
          - ``predicted_class``: str — top-1 class name
          - ``confidence``: float — top-1 softmax probability
          - ``top_k``: list of ``{"class": str, "confidence": float}``
          - ``image_path``: str — the input path
    """
    checkpoint_path = Path(
        checkpoint_path or config.MODEL_SAVE_DIR / "best_model.pt"
    )

    # 1. Load model
    model, class_to_idx, image_size = load_model(
        checkpoint_path, device=device
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 2. Load checkpoint for normalisation stats
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean = ckpt.get("imagenet_mean", config.IMAGENET_MEAN)
    std = ckpt.get("imagenet_std", config.IMAGENET_STD)

    # 3. Preprocess
    transform = get_inference_transform(image_size, mean=mean, std=std)
    input_tensor = load_and_preprocess_image(image_path, transform)
    input_tensor = input_tensor.to(device)

    # 4. Forward pass
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)

    # 5. Top-k
    top_k = min(top_k, len(probs))
    top_probs, top_indices = probs.topk(top_k)

    top_predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        top_predictions.append({
            "class": idx_to_class[idx],
            "confidence": round(prob, 6),
        })

    return {
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_k": top_predictions,
        "image_path": str(image_path),
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained AgriSmart AI model.",
    )
    p.add_argument(
        "--image", type=Path, required=True,
        help="Path to the input image.",
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=config.MODEL_SAVE_DIR / "best_model.pt",
        help="Path to the model checkpoint.",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (cpu / cuda).  Default: cpu",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to show.  Default: 5",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = predict(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            device=args.device,
            top_k=args.top_k,
        )
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"\n❌  Checkpoint error: {e}")
        sys.exit(1)

    # Pretty-print result
    print()
    print("=" * 60)
    print("  AgriSmart AI — Prediction Result")
    print("=" * 60)
    print(f"  Image       : {result['image_path']}")
    print(f"  Prediction  : {result['predicted_class']}")
    print(f"  Confidence  : {result['confidence']:.4f}")
    print()
    print(f"  Top-{len(result['top_k'])} predictions:")
    for i, pred in enumerate(result["top_k"], 1):
        bar = "█" * int(pred["confidence"] * 30)
        print(f"    {i}. {pred['class']:<40} {pred['confidence']:.4f}  {bar}")
    print("=" * 60)
    print()

    # Also print as JSON for programmatic consumption
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
