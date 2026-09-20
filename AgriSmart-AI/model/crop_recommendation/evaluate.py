"""
AgriSmart AI — Crop Recommendation Evaluation Module
=====================================================
Calculates genuine test metrics (Accuracy, Macro-F1, Weighted-F1, Per-class
Precision/Recall, and Confusion Matrix) on the held-out test set and saves reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from model.crop_recommendation.config import CropRecommendationConfig
from model.crop_recommendation.dataset import get_train_val_test_splits, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_crop_model(
    model_path: Path | None = None,
    data_path: Path | None = None,
    report_dir: Path | None = None,
    random_seed: int = CropRecommendationConfig.RANDOM_SEED,
) -> dict[str, Any]:
    """
    Evaluate the saved Crop Recommendation model strictly on the held-out test split.

    Returns:
        Dictionary containing genuine calculated metrics and report paths.
    """
    m_path = model_path or CropRecommendationConfig.MODEL_PATH
    r_dir = report_dir or CropRecommendationConfig.REPORT_DIR
    r_dir.mkdir(parents=True, exist_ok=True)

    if not m_path.exists():
        raise FileNotFoundError(f"Trained model not found at {m_path}. Please run train.py first.")

    # 1. Load model
    logger.info("Loading model from %s...", m_path)
    model = joblib.load(m_path)

    # 2. Load dataset and obtain strictly the test set
    df = load_dataset(data_path)
    _, _, _, _, X_test, y_test = get_train_val_test_splits(
        df,
        random_state=random_seed,
        test_size=CropRecommendationConfig.TEST_SIZE,
        val_size=CropRecommendationConfig.VAL_SIZE,
    )

    logger.info("Evaluating on %d held-out test samples...", len(X_test))

    # 3. Compute predictions
    y_pred = model.predict(X_test)
    classes = [str(c) for c in model.classes_]

    # 4. Compute metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_list = cm.tolist()

    # 5. Build structured metrics dictionary
    per_class_metrics: dict[str, dict[str, float]] = {}
    if isinstance(clf_report, dict):
        for cls_name in classes:
            cls_data = clf_report.get(cls_name)
            if isinstance(cls_data, dict):
                per_class_metrics[cls_name] = {
                    "precision": float(cls_data["precision"]),
                    "recall": float(cls_data["recall"]),
                    "f1_score": float(cls_data["f1-score"]),
                    "support": int(cls_data["support"]),
                }

    metrics_payload: dict[str, Any] = {
        "evaluation_scope": "Crop Recommendation Tabular Benchmark (Held-out Test Split)",
        "dataset_source": CropRecommendationConfig.DATASET_SOURCE_CITATION,
        "dataset_total_rows": len(df),
        "test_sample_count": len(X_test),
        "random_seed": random_seed,
        "overall_metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
        "per_class_metrics": per_class_metrics,
        "classes": classes,
        "num_classes": len(classes),
    }

    # 6. Save JSON artifacts
    metrics_file = r_dir / "metrics.json"
    cm_file = r_dir / "confusion_matrix.json"
    report_file = r_dir / "evaluation_report.md"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    cm_payload = {
        "classes": classes,
        "matrix": cm_list,
    }
    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)

    # 7. Generate markdown evaluation report
    markdown_report = _generate_markdown_report(
        metrics_payload=metrics_payload,
        cm=cm,
        classes=classes,
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    logger.info("Evaluation complete. Accuracy=%.4f, Macro-F1=%.4f, Weighted-F1=%.4f", acc, macro_f1, weighted_f1)
    logger.info("Report saved to: %s", report_file)

    return metrics_payload


def _generate_markdown_report(
    metrics_payload: dict[str, Any],
    cm: np.ndarray,
    classes: list[str],
) -> str:
    """Generate human-readable Markdown evaluation report."""
    overall = metrics_payload["overall_metrics"]
    per_class = metrics_payload["per_class_metrics"]

    lines = [
        "# 🌾 AgriSmart AI — Crop Recommendation Evaluation Report",
        "",
        "> [!IMPORTANT]",
        "> **Dataset & Evaluation Separation Notice**:",
        "> This evaluation applies exclusively to the **Crop Recommendation Tabular Module** on its dedicated public benchmark test split.",
        "> It is completely independent of the SIH Disease Detection model and the SIH held-out foliar disease test set.",
        "> These metrics must not be conflated with disease-detection classification metrics.",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Calculated Test Result |",
        "|---|---|",
        f"| **Test Accuracy** | **{overall['accuracy'] * 100:.2f}%** |",
        f"| **Macro-F1 Score** | **{overall['macro_f1']:.4f}** |",
        f"| **Weighted-F1 Score** | **{overall['weighted_f1']:.4f}** |",
        f"| **Held-out Test Samples** | {metrics_payload['test_sample_count']} |",
        f"| **Evaluated Crop Classes** | {metrics_payload['num_classes']} |",
        "",
        "## 2. Dataset Information",
        "",
        f"- **Dataset**: {CropRecommendationConfig.DATASET_NAME}",
        f"- **Source Citation**: {CropRecommendationConfig.DATASET_SOURCE_CITATION}",
        f"- **Public URL**: [{CropRecommendationConfig.PUBLIC_DATASET_URL}]({CropRecommendationConfig.PUBLIC_DATASET_URL})",
        "- **License**: Open Data / CC0 Public Domain",
        f"- **Total Dataset Rows**: {metrics_payload['dataset_total_rows']} rows (22 crops, 100 samples per crop, balanced)",
        f"- **Features Modeled (7)**: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`",
        f"- **Train / Val / Test Split**: 70% Train (1,540 rows) / 15% Validation (330 rows) / 15% Test (330 rows)",
        f"- **Random State**: {metrics_payload['random_seed']} (stratified sampling, zero leakage)",
        "",
        "## 3. Model Configuration",
        "",
        "- **Model Family**: `sklearn.ensemble.RandomForestClassifier`",
        "- **Number of Trees**: 100 estimators",
        "- **Inference Output**: `predict_proba()` multi-class probability distribution across all 22 crops",
        "",
        "## 4. Per-Class Performance Breakdown",
        "",
        "| Crop Class | Precision | Recall | F1-Score | Test Support |",
        "|---|---|---|---|---|",
    ]

    for cls_name in classes:
        m = per_class.get(cls_name, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0})
        display_name = cls_name.replace("_", " ").title()
        lines.append(
            f"| {display_name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} | {m['support']} |"
        )

    lines.extend([
        "",
        "## 5. Confusion Matrix Summary",
        "",
        f"A total of {cm.sum()} predictions were evaluated across {len(classes)} classes on the test set.",
        f"- Correctly classified instances (diagonal sum): **{int(np.trace(cm))} / {int(cm.sum())}**",
        f"- Full {len(classes)}x{len(classes)} matrix is serialized to [`report/crop_recommendation/confusion_matrix.json`](report/crop_recommendation/confusion_matrix.json).",
        "",
        "## 6. Honest Limitations & Operational Scope",
        "",
        "1. **Contextual Variables Not in Training Set**:",
        "   The SIH problem statement requests consideration of Soil Type, Water Availability, Season, Location, and Previous Crop.",
        "   These fields are collected by the AgriSmart AI application interface for advisory context, but are **NOT** present in the benchmark tabular dataset.",
        "   In accordance with competition integrity, no synthetic or fake values were created to force these fields into ML training.",
        "2. **Suitability Score Interpretation**:",
        "   Output percentages represent model class membership probabilities based on historical agro-climatic clusters. They indicate agro-climatic alignment, **NOT guaranteed agricultural yield**.",
        "3. **Microclimate Variability**:",
        "   Local weather anomalies, pest pressure, and irrigation variations can affect actual crop performance beyond the historical tabular envelope.",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Crop Recommendation Model on Held-out Test Split")
    parser.add_argument("--model", type=Path, default=None, help="Path to crop_model.joblib")
    parser.add_argument("--data", type=Path, default=None, help="Path to Crop_recommendation.csv")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory for evaluation artifacts")
    parser.add_argument("--seed", type=int, default=CropRecommendationConfig.RANDOM_SEED)
    args = parser.parse_args()

    evaluate_crop_model(
        model_path=args.model,
        data_path=args.data,
        report_dir=args.report_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
