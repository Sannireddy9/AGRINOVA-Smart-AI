"""
AgriSmart AI — Smart Irrigation Advisory Analyzer
=================================================
Coordinates input validation, rule evaluation, and output structuring
for the Smart Irrigation Advisor module.
"""

from __future__ import annotations
from typing import Any

from model.smart_irrigation.config import SmartIrrigationConfig
from model.smart_irrigation.rules import evaluate_smart_irrigation


def analyze_irrigation_advisory(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
    location_name: str,
    requested_location: str,
    coordinates: dict[str, float] | None = None,
    retrieved_at: str = "",
    is_mock: bool = False,
    error_reason: str | None = None,
) -> dict[str, Any]:
    """
    Validate inputs, run rule-based irrigation evaluation, and package
    a deterministic, transparent JSON response structure.
    """
    # 1. Clean & validate farm inputs
    context = farm_context or {}
    try:
        raw_moisture = float(context.get("soil_moisture", 50.0))
    except (ValueError, TypeError):
        raw_moisture = 50.0

    soil_moisture = max(0.0, min(100.0, raw_moisture))
    crop = str(context.get("crop", "Tomato")).strip() or "Tomato"
    growth_stage = str(context.get("growth_stage", "Vegetative")).strip() or "Vegetative"
    soil_type = str(context.get("soil_type", "Loamy")).strip() or "Loamy"
    irrigation_method = str(context.get("irrigation_method", "Drip")).strip() or "Drip"

    cleaned_context = {
        "crop": crop,
        "growth_stage": growth_stage,
        "soil_moisture": soil_moisture,
        "soil_type": soil_type,
        "irrigation_method": irrigation_method,
    }

    # 2. Run rule evaluation
    evaluation = evaluate_smart_irrigation(
        current_weather=current_weather,
        forecast_days=forecast_days,
        farm_context=cleaned_context,
    )

    # 3. Summarize weather context for display
    near_term = forecast_days[:3] if forecast_days else []
    max_rain_prob = max([float(d.get("precipitation_probability_max", 0.0)) for d in near_term] or [0.0])
    total_rain_expected = sum(float(d.get("precipitation_sum", 0.0)) for d in near_term)

    weather_summary = {
        "temperature": float(current_weather.get("temperature", 25.0)),
        "apparent_temperature": float(current_weather.get("apparent_temperature", 25.0)),
        "relative_humidity": float(current_weather.get("relative_humidity", 50.0)),
        "wind_speed": float(current_weather.get("wind_speed", 5.0)),
        "precipitation": float(current_weather.get("precipitation", 0.0)),
        "condition": str(current_weather.get("condition", "Clear")),
        "icon": str(current_weather.get("icon", "☀️")),
        "near_term_rain_prob_max": round(max_rain_prob, 1),
        "near_term_rain_sum_mm": round(total_rain_expected, 1),
    }

    # 4. Mode determination
    mode_str = "DEMO" if is_mock else "LIVE"

    disclaimer_text = (
        SmartIrrigationConfig.DEMO_WEATHER_WARNING if is_mock
        else SmartIrrigationConfig.METHODOLOGY_DISCLAIMER
    )

    limitations = [
        "Rule-Based Heuristic: Operating on transparent, deterministic logic rather than black-box ML inference.",
        SmartIrrigationConfig.ZERO_IOT_NOTICE,
        "No Calibrated Volume Claims: Recommendations provide qualitative timing windows rather than exact liters.",
        "Module-Specific Thresholds: Heuristics (25%, 35%, 65%) should be adjusted based on local soil and crop extension guidance.",
    ]

    return {
        "status": "success",
        "mode": mode_str,
        "weather_mode": mode_str,
        "is_mock": is_mock,
        "location": location_name,
        "requested_location": requested_location,
        "coordinates": coordinates or {"latitude": 0.0, "longitude": 0.0},
        "retrieved_at": retrieved_at,
        "recommendation": {
            "status": evaluation["status"],
            "badge_text": evaluation["badge_text"],
            "severity": evaluation["severity"],
            "headline": evaluation["headline"],
            "reason": evaluation["reason"],
            "decision_basis": evaluation["decision_basis"],
            "timeline": evaluation["timeline"],
            "priority_tier": evaluation["priority_tier"],
        },
        "inputs": {
            "soil_moisture": soil_moisture,
            "crop": crop,
            "growth_stage": growth_stage,
            "soil_type": soil_type,
            "irrigation_method": irrigation_method,
            "rain_probability": round(max_rain_prob, 1),
            "expected_rainfall": round(total_rain_expected, 1),
            "temperature": weather_summary["temperature"],
            "location": location_name,
        },
        "why_recommendation": evaluation["why_recommendation"],
        "crop_stage_context": evaluation["crop_stage_context"],
        "irrigation_method_context": evaluation["irrigation_method_context"],
        "future_rain_notice": evaluation["future_rain_notice"],
        "weather_summary": weather_summary,
        "water_saving_logic": SmartIrrigationConfig.WATER_SAVING_STATEMENT,
        "limitations": limitations,
        "disclaimer": disclaimer_text,
        "error_reason": error_reason,
    }
