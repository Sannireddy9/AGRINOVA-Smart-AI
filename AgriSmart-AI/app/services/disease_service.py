"""
AgriSmart AI — Crop Disease Detection Service
==============================================
Provides an abstraction layer between the web application and the
underlying machine learning inference module.

Key behaviors:
1. Checks whether a trained model checkpoint exists.
2. If trained checkpoint exists: runs real inference via `model.predict.predict()`.
3. If checkpoint does NOT exist: runs in explicitly labeled MOCK/DEVELOPMENT mode
   for UI testing only, never presenting mock results as real AI predictions.
4. Dynamically formats class names without hardcoding any disease catalogs.
5. Provides generic agricultural precautionary guidance (not pesticide prescriptions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import AppConfig

logger = logging.getLogger(__name__)


class DiseaseDetectionService:
    """Service layer for plant leaf disease analysis."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or AppConfig.CHECKPOINT_PATH)

    def is_checkpoint_available(self) -> bool:
        """Return True if a trained model checkpoint file exists."""
        return self.checkpoint_path.exists() and self.checkpoint_path.is_file()

    def format_class_name(self, raw_class_name: str) -> dict[str, Any]:
        """Dynamically decompose a class identifier into human-readable components.

        Does NOT rely on hardcoded class lists. Supports PlantVillage-style
        'Crop___Disease' conventions as well as simple names.

        Args:
            raw_class_name: Raw string identifier (e.g. 'Tomato___Early_blight'
                or 'Apple___healthy' or 'Grape___Black_rot').

        Returns:
            Dict containing:
                - crop: Cleaned crop name (e.g. 'Tomato')
                - condition: Cleaned disease or health condition (e.g. 'Early Blight')
                - is_healthy: Boolean flag indicating healthy status
                - display_title: Full readable string (e.g. 'Tomato — Early Blight')
        """
        clean_raw = raw_class_name.strip()
        is_healthy = "healthy" in clean_raw.lower()

        if "___" in clean_raw:
            parts = clean_raw.split("___", 1)
            crop_raw, disease_raw = parts[0], parts[1]
        elif "_" in clean_raw:
            parts = clean_raw.split("_", 1)
            crop_raw, disease_raw = parts[0], parts[1]
        else:
            crop_raw = "Crop"
            disease_raw = clean_raw

        # Format crop name
        crop = crop_raw.replace("_", " ").strip().title()
        # Clean up specific notations like (maize)
        crop = crop.replace("(Maize)", "(Maize)")

        # Format condition
        condition = disease_raw.replace("_", " ").strip().title()
        if is_healthy and condition.lower() != "healthy":
            condition = "Healthy"

        display_title = f"{crop} — {condition}"

        return {
            "crop": crop,
            "condition": condition,
            "is_healthy": is_healthy,
            "display_title": display_title,
        }

    def get_precautionary_guidance(self, is_healthy: bool, condition_name: str) -> list[str]:
        """Provide safe, basic agricultural precautionary steps.

        Note: These are general cultural practices and advisory steps,
        NOT definitive chemical/pesticide prescriptions.
        """
        if is_healthy:
            return [
                "Continue standard irrigation and balanced nutrient management.",
                "Periodically inspect the undersides of leaves for early pest or lesion onset.",
                "Ensure adequate plant spacing to maintain airflow and reduce microclimate humidity.",
                "Keep farm equipment sanitized to prevent introduction of pathogens from neighboring plots.",
            ]
        else:
            return [
                "Isolate or gently prune heavily symptomatic leaves to slow pathogen spread.",
                "Avoid overhead sprinkler irrigation; water at the root base to keep foliage dry.",
                "Disinfect all cutting tools with a 70% alcohol or diluted bleach solution between plants.",
                "Collect and safely dispose of fallen infected plant debris away from the growing area.",
                "Consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer for verified local treatments.",
            ]

    def analyze_crop_image(self, image_path: Path | str) -> dict[str, Any]:
        """Analyze a crop image and return diagnosis with guidance.

        If a trained checkpoint is present, uses `model.predict.predict()`.
        Otherwise, operates in clearly marked DEVELOPMENT/MOCK mode.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            Dict with diagnosis, confidence, formatting, guidance, and mock flag.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Check if real checkpoint exists
        if self.is_checkpoint_available():
            logger.info("Running inference with trained checkpoint: %s", self.checkpoint_path)
            # Import here so app can start even if torch is still loading
            from model.predict import predict

            pred_result = predict(
                image_path=image_path,
                checkpoint_path=self.checkpoint_path,
                device="cpu",
                top_k=3,
            )
            raw_class = pred_result["predicted_class"]
            confidence = float(pred_result["confidence"])
            is_mock = False
            disclaimer = None
            top_k = pred_result.get("top_k", [])
        else:
            logger.warning(
                "Trained checkpoint not found at %s. Using MOCK/DEVELOPMENT mode.",
                self.checkpoint_path,
            )
            # Simulated development result for UI verification
            raw_class = "Tomato___Early_blight"
            confidence = 0.942
            is_mock = True
            disclaimer = (
                "[DEVELOPMENT MOCK MODE]: The model checkpoint has not been trained yet. "
                "This result is a simulated demonstration for UI testing only and does NOT "
                "represent a real AI prediction."
            )
            top_k = [
                {"class": "Tomato___Early_blight", "confidence": 0.942},
                {"class": "Tomato___Late_blight", "confidence": 0.041},
                {"class": "Tomato___healthy", "confidence": 0.017},
            ]

        # Decompose class name dynamically
        formatted = self.format_class_name(raw_class)
        guidance = self.get_precautionary_guidance(
            is_healthy=formatted["is_healthy"],
            condition_name=formatted["condition"],
        )

        return {
            "status": "success",
            "is_mock": is_mock,
            "mock_disclaimer": disclaimer,
            "raw_class": raw_class,
            "crop": formatted["crop"],
            "condition": formatted["condition"],
            "is_healthy": formatted["is_healthy"],
            "display_title": formatted["display_title"],
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "precautionary_guidance": guidance,
            "top_predictions": top_k,
            "image_filename": image_path.name,
        }

    def diagnose(self, image_path: Path | str) -> dict[str, Any]:
        """Convenience alias for analyze_crop_image."""
        return self.analyze_crop_image(image_path)
