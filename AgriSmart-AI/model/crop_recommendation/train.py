"""
AgriSmart AI — Crop Recommendation Training Pipeline
====================================================
Trains a reproducible Random Forest tabular classifier on the Crop Recommendation
dataset, validates on the validation split, and serializes the model and metadata.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from model.crop_recommendation.config import CropRecommendationConfig
from model.crop_recommendation.dataset import get_train_val_test_splits, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_crop_model(
    data_path: Path | None = None,
    output_model_path: Path | None = None,
    output_metadata_path: Path | None = None,
    n_estimators: int = CropRecommendationConfig.N_ESTIMATORS,
    random_seed: int = CropRecommendationConfig.RANDOM_SEED,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Train a Random Forest classifier on the crop recommendation dataset.

    Returns:
        (trained_model, metadata_dictionary)
    """
    model_path = output_model_path or CropRecommendationConfig.MODEL_PATH
    meta_path = output_metadata_path or CropRecommendationConfig.METADATA_PATH

    # 1. Load and split dataset
    df = load_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
        df,
        random_state=random_seed,
        test_size=CropRecommendationConfig.TEST_SIZE,
        val_size=CropRecommendationConfig.VAL_SIZE,
    )

    # 2. Instantiate tabular baseline
    logger.info("Instantiating RandomForestClassifier (n_estimators=%d, seed=%d)", n_estimators, random_seed)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=-1,
    )

    # 3. Fit on training split
    logger.info("Fitting model on %d training samples...", len(X_train))
    model.fit(X_train, y_train)

    # 4. Evaluate on validation split (for development monitoring)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_acc = float(accuracy_score(y_train, train_preds))
    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

    logger.info("Train Accuracy: %.4f | Validation Accuracy: %.4f | Val Macro-F1: %.4f", train_acc, val_acc, val_macro_f1)

    # 5. Extract discovered class labels and feature importances
    classes = [str(cls) for cls in model.classes_]
    feature_importances = {
        feat: float(imp)
        for feat, imp in zip(CropRecommendationConfig.MODEL_FEATURES, model.feature_importances_)
    }

    # 6. Build metadata artifact
    metadata: dict[str, Any] = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "random_seed": random_seed,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": CropRecommendationConfig.DATASET_NAME,
        "dataset_source": CropRecommendationConfig.DATASET_SOURCE_CITATION,
        "total_dataset_rows": len(df),
        "split_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "features_used": CropRecommendationConfig.MODEL_FEATURES,
        "contextual_fields_unmodeled": CropRecommendationConfig.CONTEXTUAL_FIELDS,
        "classes": classes,
        "num_classes": len(classes),
        "feature_importances": feature_importances,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
    }

    # 7. Serialize model artifact and metadata
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved trained crop recommendation model to: %s", model_path)
    logger.info("Saved model metadata to: %s", meta_path)

    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train Crop Recommendation Tabular Classifier")
    parser.add_argument("--data", type=Path, default=None, help="Path to Crop_recommendation.csv")
    parser.add_argument("--output-model", type=Path, default=None, help="Output path for .joblib model")
    parser.add_argument("--output-meta", type=Path, default=None, help="Output path for metadata JSON")
    parser.add_argument("--n-estimators", type=int, default=CropRecommendationConfig.N_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=CropRecommendationConfig.RANDOM_SEED)
    args = parser.parse_args()

    train_crop_model(
        data_path=args.data,
        output_model_path=args.output_model,
        output_metadata_path=args.output_meta,
        n_estimators=args.n_estimators,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
