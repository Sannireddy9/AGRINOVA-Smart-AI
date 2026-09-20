"""
AgriSmart AI — Weather Agronomic Rules Engine
=============================================
Pure, deterministic, rule-based evaluation functions that combine meteorological
measurements with manual farmer context to generate actionable recommendations.

NOTICE:
These rules implement heuristic agricultural logic for demonstration and decision support.
They do not constitute a machine-learning model or automated physical sensor controls.
All thresholds and logic are explainable, transparent, and configurable.
"""

from __future__ import annotations

from typing import Any
from model.weather_intelligence.config import WeatherIntelligenceConfig


def compute_forecast_extremes(forecast_days: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate the highest rain probability and highest expected rainfall
    across the returned forecast days.
    """
    if not forecast_days:
        return {
            "highest_rain_probability": {
                "probability": 0.0,
                "date": "N/A",
                "formatted": "Highest rain probability: 0% (N/A)",
            },
            "highest_expected_rainfall": {
                "rainfall_mm": 0.0,
                "date": "N/A",
                "formatted": "Highest expected rainfall: 0.0 mm (N/A)",
            },
        }

    # Find highest rain probability day
    max_prob_day = max(
        forecast_days,
        key=lambda d: float(d.get("precipitation_probability_max", 0.0)),
        default=forecast_days[0],
    )
    max_prob = float(max_prob_day.get("precipitation_probability_max", 0.0))
    prob_date = str(max_prob_day.get("date", "N/A"))

    # Find highest rainfall amount day
    max_rain_day = max(
        forecast_days,
        key=lambda d: float(d.get("precipitation_sum", 0.0)),
        default=forecast_days[0],
    )
    max_rain = float(max_rain_day.get("precipitation_sum", 0.0))
    rain_date = str(max_rain_day.get("date", "N/A"))

    return {
        "highest_rain_probability": {
            "probability": max_prob,
            "date": prob_date,
            "formatted": f"Highest rain probability: {max_prob:.0f}% on {prob_date}",
        },
        "highest_expected_rainfall": {
            "rainfall_mm": round(max_rain, 1),
            "date": rain_date,
            "formatted": f"Highest expected rainfall: {max_rain:.1f} mm on {rain_date}",
        },
    }


def detect_future_rainfall_event(forecast_days: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Scan later days (days 3 to 6) for significant upcoming rainfall.
    Acts purely as an advance warning and does NOT override the immediate
    irrigation recommendation based on near-term conditions.
    """
    if len(forecast_days) <= 3:
        return None

    later_days = forecast_days[3:]
    heavy_days = [
        d for d in later_days
        if float(d.get("precipitation_sum", 0.0)) >= WeatherIntelligenceConfig.SIGNIFICANT_RAIN_AMOUNT_THRESHOLD
    ]

    if not heavy_days:
        return None

    peak_day = max(heavy_days, key=lambda d: float(d.get("precipitation_sum", 0.0)))
    peak_amount = float(peak_day.get("precipitation_sum", 0.0))
    peak_date = str(peak_day.get("date", "later this week"))

    return {
        "is_advance_notice": True,
        "amount_mm": round(peak_amount, 1),
        "date": peak_date,
        "warning": (
            f"Heavy rainfall is expected later this week ({peak_amount:.1f} mm on {peak_date}) "
            "— review your irrigation schedule before that event."
        ),
    }


def evaluate_irrigation_rule(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate immediate irrigation timing based on precipitation forecasts
    and manual soil moisture.

    Priorities:
    A. Critical moisture (<= 25%): Does not blindly delay; distinguishes immediate soaking rain vs light showers.
    B. Significant near-term rain (prob >= 60% or rain >= 8 mm) with adequate moisture (> 25%): Recommends delay.
    C. Low moisture (<= 35%) with dry forecast: Recommends considering irrigation.
    D. Transitional conditions: Recommend monitoring & reassessing.
    E. Normal conditions: Maintain normal cycle.
    """
    # 1. Extract near-term meteorological factors (next 48-72h: days 0, 1, 2)
    next_3_days = forecast_days[:3] if forecast_days else []
    max_rain_prob = max((float(d.get("precipitation_probability_max", 0.0)) for d in next_3_days), default=0.0)
    total_rain_expected = sum((float(d.get("precipitation_sum", 0.0)) for d in next_3_days), 0.0)
    current_rain = float(current_weather.get("precipitation", 0.0))

    # 2. Extract manual farmer context (default to 50% if unprovided)
    try:
        soil_moisture = float(farm_context.get("soil_moisture", 50.0))
    except (ValueError, TypeError):
        soil_moisture = 50.0

    soil_type = str(farm_context.get("soil_type", "Loamy")).strip()
    irrigation_method = str(farm_context.get("irrigation_method", "Drip")).strip()

    # 3. Detect 7-day future rainfall event (advance notice only)
    future_rain_event = detect_future_rainfall_event(forecast_days)

    # 4. Context-aware priority evaluation

    # PRIORITY A: Critical Soil Moisture (<= 25%)
    if soil_moisture <= WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:
        # Check if rain is truly heavy and imminent within 24 hours
        if total_rain_expected >= WeatherIntelligenceConfig.RAIN_HEAVY_AMOUNT_THRESHOLD or current_rain >= 5.0:
            action = "DELAY_IRRIGATION"
            severity = "warning"
            badge_text = "⏸️ DELAY IRRIGATION"
            headline = "Heavy rain is imminent within 24–48 hours despite low soil moisture."
            reason = (
                f"Soil moisture is critically low ({soil_moisture:.1f}%), but substantial rainfall "
                f"({total_rain_expected:.1f} mm expected, {max_rain_prob:.0f}% probability) is forecast immediately. "
                "Hold routine irrigation temporarily to avoid waterlogging, but monitor field moisture closely."
            )
            rule_name = "Imminent heavy rainfall temporarily delays critical deficit irrigation"
        else:
            action = "CONSIDER_IRRIGATION"
            severity = "primary"
            badge_text = "💧 CONSIDER IRRIGATION"
            headline = f"Soil moisture is at a critical deficit ({soil_moisture:.0f}%); irrigation required."
            if max_rain_prob >= WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD and total_rain_expected < WeatherIntelligenceConfig.LIGHT_RAIN_AMOUNT_THRESHOLD:
                reason = (
                    f"Manual soil moisture ({soil_moisture:.1f}%) is below the critical threshold "
                    f"({WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%). Although rain probability reaches "
                    f"{max_rain_prob:.0f}%, the expected rainfall is only {total_rain_expected:.1f} mm (light shower), "
                    f"which will not sufficiently recharge the root zone. Apply controlled irrigation ({irrigation_method}) to prevent wilting."
                )
                rule_name = "Critical soil moisture deficit prioritizes irrigation over light shower forecast"
            else:
                reason = (
                    f"Manual soil moisture ({soil_moisture:.1f}%) is below the critical threshold "
                    f"({WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%) and near-term weather is dry "
                    f"({total_rain_expected:.1f} mm expected). Immediate irrigation ({irrigation_method}) is recommended to avoid crop water stress."
                )
                rule_name = "Critical soil moisture deficit with dry near-term weather"

        return {
            "action": action,
            "badge_text": badge_text,
            "severity": severity,
            "headline": headline,
            "reason": reason,
            "urgency": "HIGH PRIORITY",
            "future_rain_warning": future_rain_event["warning"] if future_rain_event else None,
            "metrics_evaluated": {
                "max_rain_probability_pct": max_rain_prob,
                "expected_precipitation_mm": round(total_rain_expected, 1),
                "manual_soil_moisture_pct": soil_moisture,
                "threshold_critical_moisture": WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW,
            },
            "why_recommendation": {
                "inputs_evaluated": [
                    f"Soil moisture: {soil_moisture:.1f}% (Critical threshold: {WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%)",
                    f"Near-term rain probability: {max_rain_prob:.0f}%",
                    f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
                    f"Irrigation method: {irrigation_method}",
                ],
                "rule_triggered": rule_name,
            },
        }

    # PRIORITY B: Significant Near-Term Rain (Prob >= 60% or Rain Sum >= 8 mm) with adequate moisture (> 25%)
    if (
        max_rain_prob >= WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD
        or total_rain_expected >= WeatherIntelligenceConfig.SIGNIFICANT_RAIN_AMOUNT_THRESHOLD
        or current_rain >= 2.0
    ):
        if total_rain_expected < WeatherIntelligenceConfig.LIGHT_RAIN_AMOUNT_THRESHOLD:
            # High probability but very low rainfall amount
            headline = "Rain is likely during the forecast window; hold routine watering."
            reason = (
                f"Rain probability reaches {max_rain_prob:.0f}% during the forecast window, although the expected rainfall "
                f"for that period is only {total_rain_expected:.1f} mm. Current soil moisture is {soil_moisture:.1f}%, "
                f"above the critical depletion threshold ({WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%), "
                "so holding scheduled watering prevents unnecessary saturation and runoff."
            )
            rule_name = "Likely shower with adequate soil moisture"
        else:
            headline = "Rain is likely during the forecast window."
            reason = (
                f"Rain probability reaches {max_rain_prob:.0f}% with {total_rain_expected:.1f} mm precipitation expected "
                f"in the next 48–72 hours. Current manual soil moisture ({soil_moisture:.1f}%) is above critical depletion "
                f"({WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%), so irrigating now risks waterlogging, root hypoxia, and nutrient runoff."
            )
            rule_name = "Significant near-term rainfall forecast with adequate soil moisture"

        return {
            "action": "DELAY_IRRIGATION",
            "badge_text": "⏸️ DELAY IRRIGATION",
            "severity": "warning",
            "headline": headline,
            "reason": reason,
            "urgency": "STANDARD",
            "future_rain_warning": future_rain_event["warning"] if future_rain_event else None,
            "metrics_evaluated": {
                "max_rain_probability_pct": max_rain_prob,
                "expected_precipitation_mm": round(total_rain_expected, 1),
                "manual_soil_moisture_pct": soil_moisture,
                "threshold_rain_prob": WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD,
            },
            "why_recommendation": {
                "inputs_evaluated": [
                    f"Soil moisture: {soil_moisture:.1f}% (Above critical: {WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:.0f}%)",
                    f"Near-term rain probability: {max_rain_prob:.0f}%",
                    f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
                ],
                "rule_triggered": rule_name,
            },
        }

    # PRIORITY C: Low Moisture + Dry Near-Term Forecast (Soil moisture <= 35%)
    if (
        soil_moisture <= WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED
        and max_rain_prob < WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD
        and total_rain_expected < 5.0
    ):
        return {
            "action": "CONSIDER_IRRIGATION",
            "badge_text": "💧 CONSIDER IRRIGATION",
            "severity": "primary",
            "headline": f"Soil moisture is depleted ({soil_moisture:.0f}%) with dry weather ahead.",
            "reason": (
                f"Manual soil moisture ({soil_moisture:.1f}%) is below the management threshold "
                f"({WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED:.0f}%), and precipitation probability remains low "
                f"({max_rain_prob:.0f}% max, {total_rain_expected:.1f} mm expected). Apply controlled irrigation ({irrigation_method}) to replenish root zone water content."
            ),
            "urgency": "STANDARD",
            "future_rain_warning": future_rain_event["warning"] if future_rain_event else None,
            "metrics_evaluated": {
                "max_rain_probability_pct": max_rain_prob,
                "expected_precipitation_mm": round(total_rain_expected, 1),
                "manual_soil_moisture_pct": soil_moisture,
                "threshold_depleted_moisture": WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED,
            },
            "why_recommendation": {
                "inputs_evaluated": [
                    f"Soil moisture: {soil_moisture:.1f}% (Depleted threshold: {WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED:.0f}%)",
                    f"Near-term rain probability: {max_rain_prob:.0f}%",
                    f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
                ],
                "rule_triggered": "Depleted soil moisture with dry near-term outlook",
            },
        }

    # PRIORITY D: Transitional Conditions (Rain prob 30-59% or moisture 36-45%)
    if 30.0 <= max_rain_prob < WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD:
        return {
            "action": "MONITOR_AND_REASSESS",
            "badge_text": "🔍 MONITOR & REASSESS",
            "severity": "info",
            "headline": "Moderate chance of precipitation; hold routine watering.",
            "reason": (
                f"Forecast indicates a {max_rain_prob:.0f}% chance of scattered showers ({total_rain_expected:.1f} mm). "
                f"With soil moisture at {soil_moisture:.1f}%, hold non-critical irrigation and reassess after the expected cloud cover window."
            ),
            "urgency": "LOW",
            "future_rain_warning": future_rain_event["warning"] if future_rain_event else None,
            "metrics_evaluated": {
                "max_rain_probability_pct": max_rain_prob,
                "expected_precipitation_mm": round(total_rain_expected, 1),
                "manual_soil_moisture_pct": soil_moisture,
            },
            "why_recommendation": {
                "inputs_evaluated": [
                    f"Soil moisture: {soil_moisture:.1f}%",
                    f"Near-term rain probability: {max_rain_prob:.0f}%",
                    f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
                ],
                "rule_triggered": "Transitional precipitation probability (30–59%)",
            },
        }

    # PRIORITY E: Stable / Adequate Baseline Conditions
    return {
        "action": "MAINTAIN_NORMAL_CYCLE",
        "badge_text": "✅ MAINTAIN NORMAL CYCLE",
        "severity": "success",
        "headline": "Weather conditions and soil moisture are stable.",
        "reason": (
            f"Soil moisture ({soil_moisture:.1f}%) is within the optimal range and no extreme rain "
            "or drought events are forecast. Continue your standard agricultural water management schedule."
        ),
        "urgency": "NORMAL",
        "future_rain_warning": future_rain_event["warning"] if future_rain_event else None,
        "metrics_evaluated": {
            "max_rain_probability_pct": max_rain_prob,
            "expected_precipitation_mm": round(total_rain_expected, 1),
            "manual_soil_moisture_pct": soil_moisture,
        },
        "why_recommendation": {
            "inputs_evaluated": [
                f"Soil moisture: {soil_moisture:.1f}% (Optimal range)",
                f"Near-term rain probability: {max_rain_prob:.0f}%",
                f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
            ],
            "rule_triggered": "Stable near-term weather and adequate soil moisture",
        },
    }


def evaluate_disease_weather_risk(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate weather-driven foliar disease risk index.

    IMPORTANT NON-DIAGNOSTIC NOTICE:
    This function analyzes meteorological conditions (humidity, leaf-wetness potential)
    and strictly does NOT perform plant pathology diagnosis.
    """
    humidity = float(current_weather.get("relative_humidity", 50.0))
    temp = float(current_weather.get("temperature", 25.0))
    current_rain = float(current_weather.get("precipitation", 0.0))

    next_3_days = forecast_days[:3] if forecast_days else []
    max_rain_prob = max((float(d.get("precipitation_probability_max", 0.0)) for d in next_3_days), default=0.0)
    total_rain_expected = sum((float(d.get("precipitation_sum", 0.0)) for d in next_3_days), 0.0)

    crop = str(farm_context.get("crop", "Crops")).strip()

    # Rule: High humidity + precipitation potential creates prolonged canopy wetness
    if (
        humidity >= WeatherIntelligenceConfig.HIGH_HUMIDITY_THRESHOLD
        and (current_rain > 0.0 or max_rain_prob >= WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD or total_rain_expected >= 5.0)
    ):
        risk_level = "ELEVATED"
        badge_text = "🟠 ELEVATED DISEASE WEATHER RISK"
        severity = "warning"
        headline = "High humidity and wet foliage promote fungal and bacterial pathogens."
        reason = (
            f"Relative humidity is {humidity:.0f}% (threshold {WeatherIntelligenceConfig.HIGH_HUMIDITY_THRESHOLD:.0f}%) "
            f"combined with precipitation potential ({total_rain_expected:.1f} mm forecast). "
            "Prolonged leaf wetness creates favorable microclimates for foliar spores."
        )
        guidance = (
            f"Inspect {crop} leaves and canopy regularly for visible symptoms following humid/wet conditions. "
            "Use the Computer Vision disease detection module to confirm visible symptoms."
        )
        rule_name = "High relative humidity combined with precipitation potential"

    elif humidity >= WeatherIntelligenceConfig.MODERATE_HUMIDITY_THRESHOLD or max_rain_prob >= 40.0:
        risk_level = "MODERATE"
        badge_text = "🟡 MODERATE DISEASE WEATHER RISK"
        severity = "info"
        headline = "Moderate humidity levels present. Standard scouting advised."
        reason = (
            f"Atmospheric humidity is {humidity:.0f}%. Microclimate conditions may support fungal development "
            "in dense, unpruned canopies or after rain."
        )
        guidance = (
            f"Increase visual scouting across the {crop} plot, especially in dense canopies or after wet weather."
        )
        rule_name = "Moderate atmospheric humidity (60–74%)"

    else:
        risk_level = "LOW"
        badge_text = "🟢 LOW DISEASE WEATHER RISK"
        severity = "success"
        headline = "Dry atmospheric conditions suppress foliar spore germination."
        reason = (
            f"Current humidity is low ({humidity:.0f}%) with dry weather forecast. "
            "Environmental pressure for foliar fungal infection is minimal."
        )
        guidance = "Routine scouting recommended across the field."
        rule_name = "Low atmospheric humidity (< 60%)"

    return {
        "risk_level": risk_level,
        "badge_text": badge_text,
        "severity": severity,
        "headline": headline,
        "reason": reason,
        "monitoring_guidance": guidance,
        "disclaimer": WeatherIntelligenceConfig.DISEASE_WEATHER_DISCLAIMER,
        "why_recommendation": {
            "inputs_evaluated": [
                f"Relative humidity: {humidity:.0f}% (Threshold: {WeatherIntelligenceConfig.HIGH_HUMIDITY_THRESHOLD:.0f}%)",
                f"Current precipitation: {current_rain:.1f} mm",
                f"Near-term rain forecast: {total_rain_expected:.1f} mm ({max_rain_prob:.0f}% prob)",
                f"Target crop: {crop}",
            ],
            "rule_triggered": rule_name,
        },
    }


def evaluate_heat_and_spray_advisory(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate heat stress risk and pesticide/fertilizer spray drift conditions.
    Clearly separates the Spray Window from Heat Conditions.
    """
    temp = float(current_weather.get("temperature", 25.0))
    wind_speed = float(current_weather.get("wind_speed", 5.0))
    current_rain = float(current_weather.get("precipitation", 0.0))
    growth_stage = str(farm_context.get("growth_stage", "Vegetative")).strip()
    crop = str(farm_context.get("crop", "Crops")).strip()

    # 1. Heat Conditions Evaluation
    if temp >= WeatherIntelligenceConfig.HEAT_STRESS_THRESHOLD:
        heat_status = "HEAT STRESS WARNING"
        heat_desc = (
            f"High temperature ({temp:.1f}°C, threshold {WeatherIntelligenceConfig.HEAT_STRESS_THRESHOLD:.0f}°C). "
            f"During the {growth_stage} stage of {crop}, heat stress accelerates transpiration and may cause flower abortion or leaf scorch."
        )
        heat_severity = "warning"
        heat_rule = "Ambient temperature exceeds heat stress threshold (>= 35°C)"
    else:
        heat_status = "NORMAL TEMPERATURE"
        heat_desc = f"Ambient temperature ({temp:.1f}°C) is within standard physiological growth bounds for {crop}."
        heat_severity = "success"
        heat_rule = "Ambient temperature is within normal bounds (< 35°C)"

    # 2. Spray Window Evaluation
    if wind_speed >= WeatherIntelligenceConfig.HIGH_WIND_SPRAY_THRESHOLD:
        spray_status = "UNFAVORABLE — HIGH WIND DRIFT"
        spray_desc = (
            f"Wind speed is {wind_speed:.1f} km/h (threshold {WeatherIntelligenceConfig.HIGH_WIND_SPRAY_THRESHOLD:.0f} km/h). "
            "Spraying is unfavorable due to elevated wind/drift conditions and rapid droplet evaporation."
        )
        spray_severity = "warning"
        spray_rule = "Wind speed exceeds safe spray threshold (>= 20 km/h)"
    elif current_rain > 0.0:
        spray_status = "UNFAVORABLE — ACTIVE RAIN"
        spray_desc = "Spraying is unfavorable during active precipitation. Foliar applications will be washed away."
        spray_severity = "warning"
        spray_rule = "Active precipitation detected (> 0 mm)"
    else:
        spray_status = "FAVORABLE SPRAY WINDOW"
        spray_desc = (
            f"Wind speed is calm ({wind_speed:.1f} km/h) and no rain is falling. "
            "Weather conditions are currently favorable for a spray window based on wind and precipitation conditions. "
            "Follow the pesticide product label and local agricultural guidance."
        )
        spray_severity = "success"
        spray_rule = "Calm wind (< 20 km/h) and dry weather"

    return {
        "heat_advisory": {
            "status": heat_status,
            "severity": heat_severity,
            "temperature_c": temp,
            "description": heat_desc,
            "rule_triggered": heat_rule,
        },
        "spray_advisory": {
            "status": spray_status,
            "severity": spray_severity,
            "wind_speed_kmh": wind_speed,
            "description": spray_desc,
            "disclaimer": WeatherIntelligenceConfig.SPRAY_PRODUCT_LABEL_DISCLAIMER,
            "rule_triggered": spray_rule,
        },
        "why_recommendation": {
            "inputs_evaluated": [
                f"Ambient temperature: {temp:.1f}°C (Threshold: {WeatherIntelligenceConfig.HEAT_STRESS_THRESHOLD:.0f}°C)",
                f"Wind speed: {wind_speed:.1f} km/h (Threshold: {WeatherIntelligenceConfig.HIGH_WIND_SPRAY_THRESHOLD:.0f} km/h)",
                f"Active precipitation: {current_rain:.1f} mm",
                f"Growth stage: {growth_stage}",
            ],
            "rules_triggered": [spray_rule, heat_rule],
        },
    }


def generate_farm_advisory_summary(
    irrigation_advice: dict[str, Any],
    disease_weather_risk: dict[str, Any],
    heat_and_spray: dict[str, Any],
    extremes: dict[str, Any],
    farm_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce a concise top-level summary banner for today's farm operations:
    - primary_action: The single highest-priority operational recommendation
    - watch_condition: Significant upcoming weather to monitor
    - field_condition: Soil moisture interpretation in plain agronomic terms
    """
    try:
        soil_moisture = float(farm_context.get("soil_moisture", 50.0))
    except (ValueError, TypeError):
        soil_moisture = 50.0

    # 1. Primary Action Synthesis
    irr_action = irrigation_advice.get("action", "")
    if irr_action == "DELAY_IRRIGATION":
        primary_action = "Review irrigation timing — rain is likely during the near-term forecast window."
    elif irr_action == "CONSIDER_IRRIGATION":
        primary_action = "Consider irrigation — soil moisture is depleted and near-term weather is dry."
    elif irr_action == "MONITOR_AND_REASSESS":
        primary_action = "Monitor field conditions — transitional showers expected; hold routine watering."
    else:
        primary_action = "Maintain regular agricultural operations — stable moisture and calm weather."

    # 2. Watch Condition Synthesis
    future_warning = irrigation_advice.get("future_rain_warning")
    high_rain = extremes.get("highest_expected_rainfall", {})
    high_prob = extremes.get("highest_rain_probability", {})

    if future_warning:
        watch_condition = f"Watch: {future_warning}"
    elif float(high_rain.get("rainfall_mm", 0.0)) >= 10.0:
        watch_condition = f"Watch heavy precipitation: {high_rain.get('rainfall_mm')} mm expected on {high_rain.get('date')}."
    elif disease_weather_risk.get("risk_level") == "ELEVATED":
        watch_condition = "Watch elevated humidity: microclimate supports foliar spore development; scout leaves."
    elif heat_and_spray.get("heat_advisory", {}).get("severity") == "warning":
        watch_condition = "Watch high temperatures: heat stress may accelerate transpiration."
    else:
        watch_condition = "No extreme meteorological stress forecast in the 7-day window."

    # 3. Field Condition Synthesis
    if soil_moisture <= WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW:
        field_condition = f"Manual soil moisture is critically depleted ({soil_moisture:.0f}%); plants risk water stress."
    elif soil_moisture <= WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED:
        field_condition = f"Manual soil moisture is in the managed depletion range ({soil_moisture:.0f}%)."
    else:
        field_condition = f"Manual soil moisture is currently above the critical threshold ({soil_moisture:.0f}%)."

    return {
        "primary_action": primary_action,
        "watch_condition": watch_condition,
        "field_condition": field_condition,
    }
