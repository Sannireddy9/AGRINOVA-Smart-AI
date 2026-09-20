"""
AgriSmart AI — Crop Recommendation Dataset Pipeline
===================================================
Handles loading, downloading, validation, and reproducible stratified splitting
for the public crop recommendation dataset without data leakage.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.crop_recommendation.config import CropRecommendationConfig

logger = logging.getLogger(__name__)


def download_dataset_if_needed(destination: Path | None = None) -> Path:
    """Download the benchmark crop recommendation dataset if not present locally."""
    target_path = destination or CropRecommendationConfig.DATASET_PATH
    if target_path.exists() and target_path.stat().st_size > 0:
        logger.info("Crop dataset already exists at %s", target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading crop recommendation dataset from %s...", CropRecommendationConfig.PUBLIC_DATASET_URL)
    try:
        urllib.request.urlretrieve(CropRecommendationConfig.PUBLIC_DATASET_URL, target_path)
        logger.info("Successfully saved dataset to %s (%d bytes)", target_path, target_path.stat().st_size)
    except Exception as exc:
        logger.error("Failed to download crop dataset: %s", exc)
        raise RuntimeError(
            f"Could not download Crop Recommendation dataset from {CropRecommendationConfig.PUBLIC_DATASET_URL}: {exc}"
        ) from exc
    return target_path


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the Crop Recommendation dataset from CSV."""
    path = csv_path or CropRecommendationConfig.DATASET_PATH
    if not path.exists():
        path = download_dataset_if_needed(path)

    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, completeness, and values in the crop recommendation dataset."""
    missing_features = [f for f in CropRecommendationConfig.MODEL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    if CropRecommendationConfig.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column: '{CropRecommendationConfig.TARGET_COLUMN}'")

    if df.isnull().any().any():
        null_counts = df.isnull().sum().to_dict()
        raise ValueError(f"Dataset contains null values: {null_counts}")

    for feat, (low, high) in CropRecommendationConfig.FEATURE_BOUNDS.items():
        if (df[feat] < low).any() or (df[feat] > high).any():
            raise ValueError(f"Feature '{feat}' has values outside permissible bounds ({low}, {high})")

    discovered_classes = sorted(df[CropRecommendationConfig.TARGET_COLUMN].unique().tolist())
    if len(discovered_classes) < 2:
        raise ValueError(f"Expected at least 2 distinct crop classes, found: {len(discovered_classes)}")

    return {
        "num_rows": len(df),
        "num_features": len(CropRecommendationConfig.MODEL_FEATURES),
        "num_classes": len(discovered_classes),
        "classes": discovered_classes,
    }


def get_train_val_test_splits(
    df: pd.DataFrame,
    random_state: int = CropRecommendationConfig.RANDOM_SEED,
    test_size: float = CropRecommendationConfig.TEST_SIZE,
    val_size: float = CropRecommendationConfig.VAL_SIZE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the dataset into Train, Validation, and Test sets using stratified sampling.

    Prevents data leakage:
    1. First split: separates held-out Test set (e.g. 15%).
    2. Second split: divides remaining data into Train (e.g. 70%) and Validation (e.g. 15%).

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X = df[CropRecommendationConfig.MODEL_FEATURES]
    y = df[CropRecommendationConfig.TARGET_COLUMN]

    # First split: Hold out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: Divide remaining into train and validation
    # Adjusted validation proportion relative to remaining data
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(
        "Dataset split: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(X_train),
        len(X_val),
        len(X_test),
        len(df),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
