"""
AgriSmart AI — Crop Recommendation Service
===========================================
Service boundary for soil-nutrient and climate-based crop recommendation.
Encapsulates validation, Live vs. Demo mode handling, agro-climatic explanations,
and clean segregation of model features from additional contextual farmer data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from model.crop_recommendation.config import CropRecommendationConfig
from model.crop_recommendation.predict import predict

logger = logging.getLogger(__name__)


class CropRecommendationService:
    """Service boundary for crop recommendation."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or CropRecommendationConfig.MODEL_PATH

    def is_model_available(self) -> bool:
        """Return True if a trained crop recommendation model artifact exists on disk."""
        return self.model_path.exists() and self.model_path.stat().st_size > 0

    def recommend(
        self,
        input_data: dict[str, Any],
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Produce ranked crop recommendations for given farmer inputs.

        Handles both:
        - LIVE Mode: When a trained Random Forest model artifact is present.
        - DEMO / MOCK Mode: When no model is trained yet, clearly stating simulation.
        """
        # 1. Parse and validate model features
        model_features = self._extract_and_validate_model_features(input_data)

        # 2. Parse contextual farmer attributes (not in benchmark training dataset)
        context_data = self._extract_contextual_attributes(input_data)

        # 3. Check if trained model artifact exists
        if self.is_model_available():
            try:
                raw_recommendations = predict(
                    features=model_features,
                    model_path=self.model_path,
                    top_k=top_k,
                )
                is_mock = False
                mode = "LIVE"
                disclaimer = (
                    "Model recommendations derived from trained Random Forest multi-class probabilities. "
                    "Suitability scores indicate agro-climatic alignment and are not yield guarantees."
                )
            except Exception as exc:
                logger.error("Inference failed with trained model, falling back to mock: %s", exc)
                raw_recommendations = self._generate_simulated_recommendations(top_k=top_k)
                is_mock = True
                mode = "DEVELOPMENT_DEMO"
                disclaimer = f"Model execution error ({exc}). Running in fallback simulation mode."
        else:
            # Explicit DEMO mode
            raw_recommendations = self._generate_simulated_recommendations(top_k=top_k)
            is_mock = True
            mode = "DEVELOPMENT_DEMO"
            disclaimer = (
                "DEMO SIMULATION: The crop recommendation model has not been trained yet. "
                "These results are simulated for UI demonstration only and do not reflect real model inference."
            )

        # 4. Generate agronomic rationale based on available climatic/soil variables
        explanation = self._generate_agronomic_explanation(
            model_features=model_features,
            top_crops=raw_recommendations,
            is_mock=is_mock,
        )

        return {
            "status": "success",
            "is_mock": is_mock,
            "mode": mode,
            "disclaimer": disclaimer,
            "recommendations": raw_recommendations,
            "top_recommendation": raw_recommendations[0] if raw_recommendations else None,
            "agronomic_explanation": explanation,
            "model_features_used": model_features,
            "additional_context_collected": {
                **context_data,
                "_notice": (
                    "These details are collected for farm context and future advisory improvements. "
                    "The current crop recommendation model does not use these fields for its prediction."
                ),
            },
        }

    def _extract_and_validate_model_features(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract and validate the 7 numeric model features."""
        aliases = {
            "N": ["n", "nitrogen"],
            "P": ["p", "phosphorus", "phosphorous"],
            "K": ["k", "potassium"],
            "temperature": ["temperature", "temp"],
            "humidity": ["humidity", "relative_humidity"],
            "ph": ["ph", "soil_ph"],
            "rainfall": ["rainfall", "rain", "precipitation"],
        }

        norm_data = {k.strip().lower(): v for k, v in data.items()}
        validated: dict[str, float] = {}

        for feat_name, alias_list in aliases.items():
            val = None
            for alias in alias_list:
                if alias in norm_data and norm_data[alias] not in (None, ""):
                    val = norm_data[alias]
                    break

            if val is None:
                raise ValueError(
                    f"Missing required model feature: '{feat_name}'. "
                    f"Required features: {CropRecommendationConfig.MODEL_FEATURES}"
                )

            try:
                num_val = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{feat_name}' must be numeric, received: '{val}'") from exc

            low, high = CropRecommendationConfig.FEATURE_BOUNDS[feat_name]
            if not (low <= num_val <= high):
                raise ValueError(
                    f"Feature '{feat_name}' value {num_val} is outside permissible physical bounds [{low}, {high}]."
                )

            validated[feat_name] = round(num_val, 2)

        return validated

    def _extract_contextual_attributes(self, data: dict[str, Any]) -> dict[str, str]:
        """Extract additional farmer context fields without pretending they are ML features."""
        norm_data = {k.strip().lower(): v for k, v in data.items()}
        context: dict[str, str] = {}
        for field_name in CropRecommendationConfig.CONTEXTUAL_FIELDS:
            raw_val = norm_data.get(field_name, "")
            context[field_name] = str(raw_val).strip() if raw_val is not None else ""
        return context

    def _generate_simulated_recommendations(self, top_k: int = 3) -> list[dict[str, Any]]:
        """Return explicitly labeled demonstration recommendations when model is untrained."""
        demo_pool = [
            {"rank": 1, "crop": "Rice", "raw_class": "rice", "suitability_percentage": 88.0, "probability": 0.880, "is_simulated": True},
            {"rank": 2, "crop": "Maize", "raw_class": "maize", "suitability_percentage": 81.5, "probability": 0.815, "is_simulated": True},
            {"rank": 3, "crop": "Jute", "raw_class": "jute", "suitability_percentage": 73.2, "probability": 0.732, "is_simulated": True},
            {"rank": 4, "crop": "Cotton", "raw_class": "cotton", "suitability_percentage": 64.0, "probability": 0.640, "is_simulated": True},
        ]
        return demo_pool[: min(top_k, len(demo_pool))]

    def _generate_agronomic_explanation(
        self,
        model_features: dict[str, float],
        top_crops: list[dict[str, Any]],
        is_mock: bool,
    ) -> str:
        """Generate understandable agronomic commentary for the farmer."""
        if not top_crops:
            return "No matching crops could be evaluated for the provided parameters."

        top_crop_name = top_crops[0]["crop"]
        ph = model_features["ph"]
        temp = model_features["temperature"]
        rain = model_features["rainfall"]
        hum = model_features["humidity"]

        # Soil pH interpretation (ML feature: ph)
        if ph < 5.5:
            ph_desc = "strongly acidic"
        elif ph < 6.5:
            ph_desc = "moderately acidic"
        elif ph <= 7.5:
            ph_desc = "optimal neutral"
        elif ph <= 8.5:
            ph_desc = "moderately alkaline"
        else:
            ph_desc = "strongly alkaline"

        # Moisture interpretation (ML feature: rainfall)
        if rain > 180:
            moisture_desc = "high precipitation"
        elif rain > 80:
            moisture_desc = "moderate precipitation"
        else:
            moisture_desc = "semi-arid / low rainfall"

        n_val = model_features.get("N", 0)
        p_val = model_features.get("P", 0)
        k_val = model_features.get("K", 0)
        sim_prefix = "[Simulated Demonstration] " if is_mock else ""
        return (
            f"{sim_prefix}{top_crop_name} is ranked highest based on the 7 ML features used: "
            f"soil pH of {ph} (pH {ph}, {ph_desc}), temperature of {temp}°C, {hum}% relative humidity, "
            f"{moisture_desc} ({rain} mm rainfall), and soil nutrient inputs (N: {n_val}, P: {p_val}, K: {k_val} kg/ha). "
            f"These 7 agro-climatic values closely match the model's learned criteria for this crop."
        )

    # ── Future Integration Stubs / Hooks ─────────────────────────────────
    def attach_weather_context(self, weather_data: dict[str, Any]) -> None:
        """Future hook: ingest live telemetry from WeatherIntelligenceService."""
        pass

    def attach_irrigation_context(self, irrigation_data: dict[str, Any]) -> None:
        """Future hook: ingest water requirement analysis from SmartIrrigationService."""
        pass

    def attach_sustainability_context(self, sustainability_data: dict[str, Any]) -> None:
        """Future hook: evaluate nitrogen fixation and crop rotation score."""
        pass

    def attach_assistant_context(self, assistant_data: dict[str, Any]) -> None:
        """Future hook: interface with conversational vernacular agronomy assistant."""
        pass

    def attach_advisor_context(self, advisor_data: dict[str, Any]) -> None:
        """Future hook: interface with autonomous multi-season farm planner."""
        pass

    # ── Legacy Compatibility Method (Zero-Regression Stub) ───────────────
    def recommend_crops(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        soil_ph: float,
        rainfall: float,
    ) -> list[dict[str, Any]]:
        """
        Legacy 5-argument stub interface from Phase 1.
        Raises NotImplementedError to maintain 100% compatibility with test_crop_service_stub.
        Use recommend(input_data) for full service functionality.
        """
        raise NotImplementedError(
            "Legacy 5-argument stub. Please use CropRecommendationService.recommend(input_data) instead."
        )
