"""
AgriSmart AI — Sustainability Score Service Layer
==================================================
Orchestrates the Sustainability Score module.
Enforces strict input validation, transparent missing-data weight renormalization,
and optional meteorological forecast integration without hardcoded fallback cities.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.weather_service import WeatherIntelligenceService
from model.sustainability.config import SustainabilityConfig
from model.sustainability.analyzer import analyze_sustainability_score

logger = logging.getLogger(__name__)


class SustainabilityService:
    """Service boundary for agricultural sustainability scoring."""

    def __init__(self, weather_service: WeatherIntelligenceService | None = None) -> None:
        self.weather_service = weather_service or WeatherIntelligenceService()

    def compute_farm_score(
        self,
        fertilizer_usage_kg_ha: float,
        water_usage_liters_m2: float,
        cover_crop_present: bool,
    ) -> dict[str, Any]:
        """Legacy interface stub for backward compatibility with boundary tests."""
        raise NotImplementedError("Sustainability scoring module is planned for a future phase.")

    def calculate_score(
        self,
        soil_moisture: float | int | str,
        irrigation_method: str,
        water_availability: str = "Moderate",
        nutrient_practice: str = "Integrated",
        soil_cover: str = "Bare_Soil",
        crop_health_status: str | None = "Not_Assessed",
        crop: str = "Tomato",
        growth_stage: str = "Vegetative",
        location: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Validate inputs and calculate deterministic sustainability score (0-100).
        """
        # 1. Validate Soil Moisture
        try:
            moisture_val = float(soil_moisture)
        except (ValueError, TypeError):
            raise ValueError("Manual soil moisture must be a valid number between 0% and 100%.")

        if moisture_val < 0.0 or moisture_val > 100.0:
            raise ValueError("Manual soil moisture must be between 0% and 100%.")

        # 2. Validate Irrigation Method
        clean_method_raw = (irrigation_method or "").strip()
        clean_method = {
            "drip": "Drip",
            "drip micro-irrigation": "Drip",
            "sprinkler": "Sprinkler",
            "overhead sprinkler": "Sprinkler",
            "flood": "Flood",
            "surface": "Flood",
            "surface / flood": "Flood",
            "rainfed": "Rainfed",
        }.get(clean_method_raw.lower(), clean_method_raw)

        if clean_method not in SustainabilityConfig.ALLOWED_IRRIGATION_METHODS:
            raise ValueError(
                f"Invalid irrigation method '{clean_method_raw}'. Allowed options: "
                f"{', '.join(sorted(SustainabilityConfig.ALLOWED_IRRIGATION_METHODS))}."
            )

        # 3. Validate Water Availability
        clean_water_raw = (water_availability or "Moderate").strip()
        clean_water = {
            "abundant": "Abundant",
            "moderate": "Moderate",
            "scarce": "Scarce",
            "rainfed": "Rainfed",
        }.get(clean_water_raw.lower(), clean_water_raw)

        if clean_water not in SustainabilityConfig.ALLOWED_WATER_AVAILABILITIES:
            raise ValueError(
                f"Invalid water availability '{clean_water_raw}'. Allowed options: "
                f"{', '.join(sorted(SustainabilityConfig.ALLOWED_WATER_AVAILABILITIES))}."
            )

        # 4. Validate Nutrient Practice
        clean_nutrient_raw = (nutrient_practice or "Integrated").strip()
        clean_nutrient = {
            "organic": "Organic",
            "integrated": "Integrated",
            "inm": "Integrated",
            "moderate_chemical": "Moderate_Chemical",
            "moderate chemical": "Moderate_Chemical",
            "intensive_chemical": "Intensive_Chemical",
            "intensive chemical": "Intensive_Chemical",
            "synthetic_heavy": "Synthetic_Heavy",
            "synthetic heavy": "Synthetic_Heavy",
            "chemical": "Intensive_Chemical",
        }.get(clean_nutrient_raw.lower(), clean_nutrient_raw)

        if clean_nutrient not in SustainabilityConfig.ALLOWED_NUTRIENT_PRACTICES:
            raise ValueError(
                f"Invalid nutrient practice '{clean_nutrient_raw}'. Allowed options: "
                f"{', '.join(sorted(SustainabilityConfig.ALLOWED_NUTRIENT_PRACTICES))}."
            )

        # 5. Clean and normalize optional contextual fields
        clean_cover_raw = (soil_cover or "Bare_Soil").strip()
        clean_cover = {
            "cover_crops": "Cover_Crops",
            "cover crops": "Cover_Crops",
            "mulch": "Mulch",
            "mulched": "Mulch",
            "minimum_tillage": "Minimum_Tillage",
            "minimum tillage": "Minimum_Tillage",
            "conservation tillage": "Minimum_Tillage",
            "bare_soil": "Bare_Soil",
            "bare soil": "Bare_Soil",
            "bare": "Bare_Soil",
            "none": "Bare_Soil",
        }.get(clean_cover_raw.lower(), clean_cover_raw)

        if clean_cover not in SustainabilityConfig.ALLOWED_SOIL_COVERS:
            clean_cover = "Bare_Soil"

        clean_health_raw = (crop_health_status or "Not_Assessed").strip()
        clean_health = {
            "healthy": "Healthy",
            "healthy foliage": "Healthy",
            "minor_stress": "Minor_Stress",
            "minor stress": "Minor_Stress",
            "mild_stress": "Mild_Stress",
            "mild stress": "Mild_Stress",
            "disease_detected": "Disease_Detected",
            "disease detected": "Disease_Detected",
            "moderate_disease": "Moderate_Disease",
            "moderate disease": "Moderate_Disease",
            "severe_damage": "Severe_Damage",
            "severe damage": "Severe_Damage",
            "severe_disease": "Severe_Disease",
            "severe disease": "Severe_Disease",
            "not_assessed": "Not_Assessed",
            "not assessed": "Not_Assessed",
            "none": "Not_Assessed",
            "": "Not_Assessed",
        }.get(clean_health_raw.lower(), clean_health_raw)

        if clean_health not in SustainabilityConfig.ALLOWED_CROP_HEALTH_STATUSES:
            clean_health = "Not_Assessed"

        clean_crop_raw = (crop or "").strip()
        clean_crop = clean_crop_raw if clean_crop_raw else "Other / Not Listed"
        clean_stage = (growth_stage or "Vegetative").strip() or "Vegetative"

        # 6. Optional Weather Integration (Zero hidden city default)
        weather_summary: dict[str, Any] | None = None
        weather_status = "UNAVAILABLE"

        clean_location = (location or "").strip()
        if clean_location:
            try:
                weather_data = self.weather_service.analyze(
                    location=clean_location,
                    force_refresh=force_refresh,
                )
                is_mock = bool(weather_data.get("is_mock", False))
                weather_status = "DEMO" if is_mock else "LIVE"

                forecast_days = weather_data.get("forecast", [])
                near_term_days = forecast_days[:3] if forecast_days else []
                max_prob = max([float(d.get("precipitation_probability_max", 0.0)) for d in near_term_days] or [0.0])
                total_rain = sum([float(d.get("precipitation_sum", 0.0)) for d in near_term_days])

                weather_summary = {
                    "has_rain_forecast": max_prob >= 50.0 or total_rain >= 4.0,
                    "max_rain_prob": max_prob,
                    "expected_rain_mm": total_rain,
                    "is_mock": is_mock,
                    "location": weather_data.get("location", clean_location),
                }
            except Exception as exc:
                logger.warning("Optional weather query failed for sustainability scoring on '%s': %s", clean_location, exc)
                weather_summary = None
                weather_status = "UNAVAILABLE"

        # 7. Coordinate structured analysis
        result = analyze_sustainability_score(
            soil_moisture=moisture_val,
            irrigation_method=clean_method,
            water_availability=clean_water,
            nutrient_practice=clean_nutrient,
            soil_cover=clean_cover,
            crop_health_status=clean_health,
            crop=clean_crop,
            growth_stage=clean_stage,
            weather_summary=weather_summary,
            weather_status=weather_status,
        )

        return result
