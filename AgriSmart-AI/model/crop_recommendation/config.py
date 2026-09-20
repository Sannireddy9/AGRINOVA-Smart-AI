"""
AgriSmart AI — Crop Recommendation Configuration
=================================================
Centralized settings, paths, validation bounds, and hyperparameters for
the tabular crop recommendation baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class CropRecommendationConfig:
    """Configuration for Crop Recommendation dataset, model, and inference."""

    # Base workspace directory
    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    # File paths
    DATA_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "crop_recommendation"
    DATASET_PATH: ClassVar[Path] = DATA_DIR / "Crop_recommendation.csv"
    MODEL_PATH: ClassVar[Path] = DATA_DIR / "crop_model.joblib"
    METADATA_PATH: ClassVar[Path] = DATA_DIR / "metadata.json"
    REPORT_DIR: ClassVar[Path] = PROJECT_ROOT / "report" / "crop_recommendation"
    METRICS_PATH: ClassVar[Path] = REPORT_DIR / "metrics.json"
    CONFUSION_MATRIX_PATH: ClassVar[Path] = REPORT_DIR / "confusion_matrix.json"
    EVALUATION_REPORT_PATH: ClassVar[Path] = REPORT_DIR / "evaluation_report.md"

    # Public benchmark dataset source information
    PUBLIC_DATASET_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv"
    )
    DATASET_NAME: ClassVar[str] = "Crop Recommendation Dataset"
    DATASET_SOURCE_CITATION: ClassVar[str] = (
        "Atharva Inamdar / gabbygab1233 (Public Domain / Open Data for precision agriculture)"
    )

    # Features used directly by the tabular ML model
    MODEL_FEATURES: ClassVar[list[str]] = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    # Additional contextual attributes collected from the farmer for UI/advisory
    # but NOT present in the benchmark dataset and NOT used as fake training features
    CONTEXTUAL_FIELDS: ClassVar[list[str]] = [
        "soil_type",
        "water_availability",
        "season",
        "location",
        "previous_crop",
    ]

    TARGET_COLUMN: ClassVar[str] = "label"

    # Permissible physical bounds for input validation
    FEATURE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "N": (0.0, 500.0),
        "P": (0.0, 500.0),
        "K": (0.0, 500.0),
        "temperature": (-10.0, 60.0),
        "humidity": (0.0, 100.0),
        "ph": (0.0, 14.0),
        "rainfall": (0.0, 5000.0),
    }

    # Model training hyperparameters
    RANDOM_SEED: ClassVar[int] = 42
    TEST_SIZE: ClassVar[float] = 0.15
    VAL_SIZE: ClassVar[float] = 0.15
    N_ESTIMATORS: ClassVar[int] = 100
    MAX_DEPTH: ClassVar[int | None] = None
