"""
AgriSmart AI — Crop Recommendation Prediction Module
=====================================================
Loads trained Random Forest model and metadata, validates input features,
computes multi-class class probabilities, and returns ranked crop recommendations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from model.crop_recommendation.config import CropRecommendationConfig

logger = logging.getLogger(__name__)

# Cache for loaded model and metadata in-memory
_MODEL_CACHE: dict[str, Any] = {}


def load_crop_model(model_path: Path | None = None) -> tuple[Any, list[str]]:
    """
    Load the trained Random Forest model and discovered class list.

    Returns:
        (model_instance, class_names_list)
    """
    path = model_path or CropRecommendationConfig.MODEL_PATH
    cache_key = str(path.resolve())

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]["model"], _MODEL_CACHE[cache_key]["classes"]

    if not path.exists():
        raise FileNotFoundError(
            f"Trained Crop Recommendation model not found at {path}. "
            "Please train the model using 'python -m model.crop_recommendation.train'."
        )

    model = joblib.load(path)
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if not classes:
        raise ValueError("Loaded model does not contain valid class labels in 'classes_' attribute.")

    _MODEL_CACHE[cache_key] = {"model": model, "classes": classes}
    return model, classes


def validate_input_features(features: dict[str, Any] | list[float] | np.ndarray) -> np.ndarray:
    """
    Validate and extract ordered feature array for model prediction.

    Expected order: [N, P, K, temperature, humidity, ph, rainfall]
    """
    if isinstance(features, (list, tuple, np.ndarray)):
        arr = np.array(features, dtype=float)
        if arr.ndim == 1:
            if len(arr) != len(CropRecommendationConfig.MODEL_FEATURES):
                raise ValueError(
                    f"Expected {len(CropRecommendationConfig.MODEL_FEATURES)} features, received {len(arr)}: {arr}"
                )
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] != len(CropRecommendationConfig.MODEL_FEATURES):
                raise ValueError(
                    f"Expected shape (1, {len(CropRecommendationConfig.MODEL_FEATURES)}), received {arr.shape}"
                )
        else:
            raise ValueError(f"Invalid feature array dimensions: {arr.ndim}")

        # Validate bounds
        for i, feat_name in enumerate(CropRecommendationConfig.MODEL_FEATURES):
            val = float(arr[0, i])
            low, high = CropRecommendationConfig.FEATURE_BOUNDS[feat_name]
            if not (low <= val <= high):
                raise ValueError(
                    f"Feature '{feat_name}' value {val} is outside permissible physical bounds [{low}, {high}]."
                )
        return arr

    if isinstance(features, dict):
        # Normalize keys to lowercase for flexible matching
        norm_dict = {k.strip().lower(): v for k, v in features.items()}

        # Key mapping aliases
        key_aliases = {
            "n": ["n", "nitrogen"],
            "p": ["p", "phosphorus", "phosphorous"],
            "k": ["k", "potassium"],
            "temperature": ["temperature", "temp"],
            "humidity": ["humidity", "relative_humidity"],
            "ph": ["ph", "soil_ph"],
            "rainfall": ["rainfall", "rain", "precipitation"],
        }

        extracted: list[float] = []
        for feat in CropRecommendationConfig.MODEL_FEATURES:
            aliases = key_aliases.get(feat.lower(), [feat.lower()])
            found_val = None
            for alias in aliases:
                if alias in norm_dict and norm_dict[alias] is not None:
                    found_val = norm_dict[alias]
                    break

            if found_val is None:
                raise ValueError(
                    f"Missing required model feature: '{feat}'. "
                    f"Required features: {CropRecommendationConfig.MODEL_FEATURES}"
                )

            try:
                num_val = float(found_val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{feat}' must be a numeric value, got: '{found_val}'") from exc

            low, high = CropRecommendationConfig.FEATURE_BOUNDS[feat]
            if not (low <= num_val <= high):
                raise ValueError(
                    f"Feature '{feat}' value {num_val} is outside permissible physical bounds [{low}, {high}]."
                )

            extracted.append(num_val)

        return np.array([extracted], dtype=float)

    raise TypeError(f"Features must be a dictionary or numeric array, received {type(features).__name__}")


def format_crop_display_name(raw_class: str) -> str:
    """Format raw crop identifier into clean human-readable title."""
    clean = raw_class.replace("_", " ").strip()
    special_names = {
        "pigeonpeas": "Pigeon Peas",
        "mothbeans": "Moth Beans",
        "mungbean": "Mung Bean",
        "blackgram": "Black Gram",
        "kidneybeans": "Kidney Beans",
        "chickpea": "Chickpea",
        "muskmelon": "Muskmelon",
        "watermelon": "Watermelon",
    }
    return special_names.get(clean.lower(), clean.title())


def predict(
    features: dict[str, Any] | list[float] | np.ndarray,
    model_path: Path | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Generate ranked crop recommendations given soil and climate features.

    Args:
        features: Dictionary or list of [N, P, K, temperature, humidity, ph, rainfall]
        model_path: Optional custom path to model artifact
        top_k: Number of ranked recommendations to return (default 3)

    Returns:
        List of ranked recommendation dicts containing:
        - rank (int): 1-indexed rank
        - crop (str): Clean formatted crop name
        - raw_class (str): Discovered model class name
        - suitability_percentage (float): Model probability expressed as percentage (e.g. 91.2)
        - probability (float): Raw model probability [0.0 - 1.0]
    """
    model, classes = load_crop_model(model_path)
    X = validate_input_features(features)

    # Convert to DataFrame with feature names to maintain sklearn feature parity without warnings
    import pandas as pd
    X_df = pd.DataFrame(X, columns=CropRecommendationConfig.MODEL_FEATURES)

    # Compute softmax-like class probabilities from ensemble
    proba = model.predict_proba(X_df)[0]

    # Rank classes in descending order of probability
    ranked_indices = np.argsort(proba)[::-1]
    capped_k = min(top_k, len(classes))

    recommendations: list[dict[str, Any]] = []
    for rank_idx, idx in enumerate(ranked_indices[:capped_k], start=1):
        raw_crop = classes[idx]
        prob_val = float(proba[idx])
        pct_val = round(prob_val * 100.0, 1)

        recommendations.append({
            "rank": rank_idx,
            "crop": format_crop_display_name(raw_crop),
            "raw_class": raw_crop,
            "suitability_percentage": pct_val,
            "probability": round(prob_val, 4),
        })

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Run Crop Recommendation Inference")
    parser.add_argument("--n", type=float, required=True, help="Nitrogen ratio in soil (kg/ha)")
    parser.add_argument("--p", type=float, required=True, help="Phosphorus ratio in soil (kg/ha)")
    parser.add_argument("--k", type=float, required=True, help="Potassium ratio in soil (kg/ha)")
    parser.add_argument("--temp", "--temperature", type=float, required=True, help="Temperature in deg C")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity in %")
    parser.add_argument("--ph", type=float, required=True, help="Soil pH (0-14)")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall in mm")
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations to return")
    parser.add_argument("--model", type=Path, default=None, help="Custom model path")
    args = parser.parse_args()

    feat_dict = {
        "N": args.n,
        "P": args.p,
        "K": args.k,
        "temperature": args.temp,
        "humidity": args.humidity,
        "ph": args.ph,
        "rainfall": args.rainfall,
    }

    try:
        results = predict(feat_dict, model_path=args.model, top_k=args.top_k)
        print("\nCROP RECOMMENDATIONS:")
        for r in results:
            print(f"  {r['rank']}. {r['crop']} - Suitability: {r['suitability_percentage']}% (Confidence: {r['probability']})")
    except Exception as exc:
        print(f"Prediction error: {exc}")


if __name__ == "__main__":
    main()
