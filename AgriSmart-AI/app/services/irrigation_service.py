"""
AgriSmart AI — Smart Irrigation Advisor Service Layer
=====================================================
Orchestrates the Smart Irrigation Advisor module. Reuses the existing
WeatherIntelligenceService for normalized meteorological data, enforces
strict input validation and Zero-IoT standards, and runs deterministic
rule-based evaluations.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.weather_service import WeatherIntelligenceService
from model.smart_irrigation.analyzer import analyze_irrigation_advisory

logger = logging.getLogger(__name__)


class SmartIrrigationService:
    """Service boundary for rule-based agricultural irrigation advisory."""

    def __init__(self, weather_service: WeatherIntelligenceService | None = None) -> None:
        self.weather_service = weather_service or WeatherIntelligenceService()

    def is_live_configured(self) -> bool:
        """Return True if the underlying weather service is configured for live fetching."""
        return self.weather_service.is_live_configured()

    def get_advisory(
        self,
        location: str | None,
        crop: str = "Tomato",
        growth_stage: str = "Vegetative",
        soil_moisture: float | int | str = 50.0,
        soil_type: str = "Loamy",
        irrigation_method: str = "Drip",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Validate inputs, retrieve normalized weather from WeatherIntelligenceService,
        and generate a deterministic, explainable smart irrigation recommendation.
        """
        # 1. Validate Location (empty, whitespace-only, null, or missing)
        if location is None:
            raise ValueError("Please enter your farm location so we can retrieve the correct weather forecast.")

        clean_location = location.strip()
        if not clean_location:
            raise ValueError("Please enter your farm location so we can retrieve the correct weather forecast.")

        # 2. Pre-validate geocoding in live mode (do not substitute another city or fallback to demo on invalid location)
        if hasattr(self.weather_service, "geocode_location") and getattr(self.weather_service, "mode", "auto") != "demo":
            try:
                self.weather_service.geocode_location(clean_location)
            except ValueError as geo_err:
                logger.warning("Farm location '%s' cannot be geocoded: %s", clean_location, geo_err)
                raise ValueError("We couldn't locate that farm location. Please enter a nearby city, district, or region.") from geo_err
            except Exception as geo_exc:
                logger.warning("Geocoding network check encountered non-fatal error for '%s': %s", clean_location, geo_exc)

        # 3. Validate Soil Moisture
        try:
            moisture_val = float(soil_moisture)
        except (ValueError, TypeError):
            raise ValueError("Manual soil moisture must be a valid number between 0% and 100%.")

        if moisture_val < 0.0 or moisture_val > 100.0:
            raise ValueError("Manual soil moisture must be between 0% and 100%.")

        # 4. Clean optional/contextual parameters
        clean_crop = (crop or "Tomato").strip() or "Tomato"
        clean_stage = (growth_stage or "Vegetative").strip() or "Vegetative"
        clean_soil_type = (soil_type or "Loamy").strip() or "Loamy"
        clean_method = (irrigation_method or "Drip").strip() or "Drip"

        farm_context = {
            "crop": clean_crop,
            "growth_stage": clean_stage,
            "soil_moisture": moisture_val,
            "soil_type": clean_soil_type,
            "irrigation_method": clean_method,
        }

        # 5. Query existing Weather Intelligence Service (reuse normalized weather data)
        try:
            weather_data = self.weather_service.analyze(
                location=clean_location,
                farm_context=farm_context,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            logger.warning("Weather query failed for irrigation advisory on '%s': %s", clean_location, exc)
            # Re-raise with clean friendly error message
            raise RuntimeError(f"Unable to retrieve meteorological data for '{clean_location}'. Please retry.") from exc

        # Check if weather_data fell back to mock due to unresolvable geocoding
        if weather_data.get("is_mock"):
            err_reason = str(weather_data.get("error_reason") or "").lower()
            if "could not resolve coordinates" in err_reason or "couldn't locate" in err_reason or "cannot be geocoded" in err_reason:
                logger.warning("Location '%s' could not be resolved from weather data: %s", clean_location, weather_data.get("error_reason"))
                raise ValueError("We couldn't locate that farm location. Please enter a nearby city, district, or region.")

        current_weather = weather_data.get("current_weather", {})
        forecast_days = weather_data.get("forecast", [])
        is_mock = bool(weather_data.get("is_mock", False))
        resolved_location = weather_data.get("location", clean_location)
        requested_location = weather_data.get("requested_location", clean_location)
        coordinates = weather_data.get("coordinates", {"latitude": 0.0, "longitude": 0.0})
        retrieved_at = weather_data.get("retrieved_at", "")
        error_reason = weather_data.get("error_reason")

        # 5. Generate structured advisory
        advisory = analyze_irrigation_advisory(
            current_weather=current_weather,
            forecast_days=forecast_days,
            farm_context=farm_context,
            location_name=resolved_location,
            requested_location=requested_location,
            coordinates=coordinates,
            retrieved_at=retrieved_at,
            is_mock=is_mock,
            error_reason=error_reason,
        )

        return advisory

    def calculate_water_requirement(self, *args: Any, **kwargs: Any) -> Any:
        """
        Legacy stub for interface boundary testing.
        Smart Irrigation does not predict precise numerical water requirements or
        volumes, as that would require unverified empirical calibration.
        Use get_advisory() for rule-based decision support.
        """
        raise NotImplementedError(
            "calculate_water_requirement is not implemented. AgriSmart AI uses rule-based "
            "timing decision support (get_advisory) without uncalibrated numerical volume predictions."
        )
