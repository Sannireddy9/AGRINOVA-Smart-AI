"""
AgriSmart AI — Weather Intelligence Analyzer
============================================
Coordinates meteorological evaluation with farm context and formats structured
intelligence payloads for UI display and future module integration.
"""

from __future__ import annotations

import datetime
from typing import Any
from model.weather_intelligence.rules import (
    compute_forecast_extremes,
    evaluate_disease_weather_risk,
    evaluate_heat_and_spray_advisory,
    evaluate_irrigation_rule,
    generate_farm_advisory_summary,
)


def analyze_farm_weather(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
    location_name: str,
    provider_name: str,
    is_mock: bool,
    disclaimer: str,
    error_reason: str | None = None,
    requested_location: str | None = None,
    coordinates: dict[str, float] | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """
    Produce comprehensive weather-based farm intelligence.

    Args:
        current_weather: Normalized current weather parameters
        forecast_days: List of 7-day daily forecast summaries
        farm_context: Farmer-entered farm parameters (manual input)
        location_name: Resolved location string
        provider_name: Meteorological provider name
        is_mock: True if running on simulated/fallback data
        disclaimer: Clear advisory/simulation notice
        error_reason: Optional description of why live data was unavailable
        requested_location: Original string entered by farmer
        coordinates: Lat/Lon coordinates dictionary
        retrieved_at: Timestamp of observation / API query

    Returns:
        Structured dictionary matching integration interface specifications.
    """
    # 1. Evaluate agronomic rule engines
    irrigation_advice = evaluate_irrigation_rule(current_weather, forecast_days, farm_context)
    disease_risk = evaluate_disease_weather_risk(current_weather, forecast_days, farm_context)
    heat_and_spray = evaluate_heat_and_spray_advisory(current_weather, forecast_days, farm_context)

    # 2. Extract forecast extremes (highest probability & highest rainfall with dates)
    forecast_extremes = compute_forecast_extremes(forecast_days)

    # 3. Generate Today's Farm Advisory Summary
    advisory_summary = generate_farm_advisory_summary(
        irrigation_advice=irrigation_advice,
        disease_weather_risk=disease_risk,
        heat_and_spray=heat_and_spray,
        extremes=forecast_extremes,
        farm_context=farm_context,
    )

    # 4. Extract priority farm actions
    primary_actions = [
        irrigation_advice["headline"],
        disease_risk["monitoring_guidance"],
        heat_and_spray["spray_advisory"]["description"],
    ]

    mode = "DEMO" if is_mock else "LIVE"
    now_iso = retrieved_at or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "success",
        "mode": mode,
        "is_mock": is_mock,
        "error_reason": error_reason,
        "disclaimer": disclaimer,
        "provider": provider_name,
        "location": location_name,
        "requested_location": requested_location or location_name,
        "resolved_location": location_name,
        "coordinates": coordinates or {"latitude": 0.0, "longitude": 0.0},
        "retrieved_at": now_iso,
        "current_weather": current_weather,
        "forecast": forecast_days,
        "forecast_extremes": forecast_extremes,
        "farm_context": farm_context,
        "farm_advisory_summary": advisory_summary,
        "irrigation_advice": irrigation_advice,
        "disease_weather_risk": disease_risk,
        "heat_and_spray_advisory": heat_and_spray,
        "primary_actions": primary_actions,
    }
