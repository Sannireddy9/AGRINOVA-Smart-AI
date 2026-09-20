"""
AgriSmart AI — Smart Irrigation Decision Rules Engine
=====================================================
Pure, deterministic, explainable rule functions that evaluate manual soil moisture,
meteorological forecasts, crop type, and growth stage to produce actionable
irrigation recommendations.

NOTICE:
- Rule-based decision-support heuristics only; no machine learning model.
- Thresholds are defined in SmartIrrigationConfig and are module-specific.
- Zero-IoT Architecture: soil moisture is strictly a farmer-provided manual observation.
- No calibrated crop water coefficients or numerical irrigation volume claims.
"""

from __future__ import annotations

from typing import Any
from model.smart_irrigation.config import SmartIrrigationConfig


def detect_future_rain_notice(forecast_days: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Scan later forecast days (days 3 to 6) for significant upcoming rainfall.
    Acts purely as an advance planning notice; does NOT override the immediate recommendation.
    """
    if not forecast_days or len(forecast_days) <= 3:
        return None

    later_days = forecast_days[3:]
    heavy_days = [
        d for d in later_days
        if float(d.get("precipitation_sum", 0.0)) >= SmartIrrigationConfig.FUTURE_RAIN_EVENT_THRESHOLD
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
            f"{peak_amount:.1f} mm rainfall is forecast on {peak_date}. "
            "Review upcoming irrigation schedules before this event."
        ),
    }


def evaluate_smart_irrigation(
    current_weather: dict[str, Any],
    forecast_days: list[dict[str, Any]],
    farm_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate irrigation timing using the 5-tier priority hierarchy:
    Priority A: Critical Moisture (<= 25%)
    Priority B: Adequate Moisture + Significant Rain
    Priority C: Low/Depleted Moisture (<= 35%) + Dry Forecast
    Priority D: Moderate / Uncertain Conditions
    Priority E: Adequate Moisture + Dry/Normal Conditions
    """
    # 1. Clean & validate inputs
    try:
        soil_moisture = float(farm_context.get("soil_moisture", 50.0))
    except (ValueError, TypeError):
        soil_moisture = 50.0

    soil_moisture = max(0.0, min(100.0, soil_moisture))
    crop = str(farm_context.get("crop", "Tomato")).strip() or "Tomato"
    growth_stage = str(farm_context.get("growth_stage", "Vegetative")).strip() or "Vegetative"
    soil_type = str(farm_context.get("soil_type", "Loamy")).strip() or "Loamy"
    irrigation_method = str(farm_context.get("irrigation_method", "Drip")).strip() or "Drip"

    temp = float(current_weather.get("temperature", 25.0))
    current_rain = float(current_weather.get("precipitation", 0.0))

    # Evaluate near-term forecast (next 48-72 hours, up to 3 days)
    near_term_days = forecast_days[:3] if forecast_days else []
    max_rain_prob = max(
        [float(d.get("precipitation_probability_max", 0.0)) for d in near_term_days] or [0.0]
    )
    total_rain_expected = sum(
        float(d.get("precipitation_sum", 0.0)) for d in near_term_days
    )

    # Detect future advance notice (days 3-6)
    future_notice = detect_future_rain_notice(forecast_days)

    # 2. Evaluate Decision Hierarchy

    # ──────────────────────────────────────────────────────────────────────
    # PRIORITY A — CRITICALLY LOW MOISTURE (<= 25%)
    # ──────────────────────────────────────────────────────────────────────
    if soil_moisture <= SmartIrrigationConfig.SOIL_MOISTURE_CRITICAL:
        # A.1: Critical moisture + high probability + meaningful expected rainfall
        if (
            max_rain_prob >= SmartIrrigationConfig.IMMINENT_RAIN_PROB_THRESHOLD
            and total_rain_expected >= SmartIrrigationConfig.IMMINENT_RAIN_AMOUNT_THRESHOLD
        ) or current_rain >= 2.0:
            status = "REVIEW_TIMING_IMMINENT_RAIN"
            badge_text = "⏸️ REVIEW TIMING — IMMINENT RAIN"
            severity = "warning"
            headline = "Critical soil moisture detected, but rainfall is forecast soon."
            reason = (
                f"Critical soil moisture ({soil_moisture:.1f}%) detected, but meaningful rainfall is forecast soon "
                f"({max_rain_prob:.0f}% chance, {total_rain_expected:.1f} mm expected). Review irrigation timing "
                "and avoid unnecessary watering immediately before rain."
            )
            decision_basis = "Conditional"
            timeline = "Review before forecast rainfall event"
            rule_name = "Critical soil moisture with meaningful near-term rainfall"

        # A.2: Critical moisture + high probability but negligible expected rainfall (< 2.0 mm)
        elif (
            max_rain_prob >= SmartIrrigationConfig.IMMINENT_RAIN_PROB_THRESHOLD
            and total_rain_expected < SmartIrrigationConfig.LIGHT_RAIN_AMOUNT_THRESHOLD
        ):
            status = "CRITICAL_REVIEW_NOW"
            badge_text = "🚨 CRITICAL: REVIEW IRRIGATION NOW"
            severity = "danger"
            headline = "CRITICAL: Review irrigation now — rain probability is high but expected amount is negligible."
            reason = (
                f"Manual soil moisture is at a critical deficit ({soil_moisture:.1f}%). Although a rain probability "
                f"of {max_rain_prob:.0f}% is forecast, the expected rainfall is only {total_rain_expected:.1f} mm, "
                "which is insufficient to replenish the root zone. Consider prompt irrigation to prevent severe crop stress."
            )
            decision_basis = "Strong"
            timeline = "Review irrigation now"
            rule_name = "Critical soil moisture with high rain probability but negligible rainfall"

        # A.3: Critical moisture + dry/low-rain forecast (< 60% prob or < 2.0 mm rain)
        else:
            status = "CRITICAL_REVIEW_NOW"
            badge_text = "🚨 CRITICAL: REVIEW IRRIGATION NOW"
            severity = "danger"
            headline = "CRITICAL: Review irrigation now — severe root moisture deficit."
            reason = (
                f"Critical soil moisture detected ({soil_moisture:.1f}% is at or below the critical threshold "
                f"{SmartIrrigationConfig.SOIL_MOISTURE_CRITICAL:.0f}%). Near-term forecast shows dry conditions "
                f"({max_rain_prob:.0f}% rain prob, {total_rain_expected:.1f} mm expected). Consider irrigation "
                "based on crop and field conditions to prevent severe water stress."
            )
            decision_basis = "Strong"
            timeline = "Review irrigation now"
            rule_name = "Critical soil moisture with dry near-term forecast"

        priority_tier = "A"

    # ──────────────────────────────────────────────────────────────────────
    # PRIORITY B — ADEQUATE MOISTURE + SIGNIFICANT RAIN
    # ──────────────────────────────────────────────────────────────────────
    elif (
        max_rain_prob >= SmartIrrigationConfig.IMMINENT_RAIN_PROB_THRESHOLD
        or total_rain_expected >= SmartIrrigationConfig.IMMINENT_RAIN_AMOUNT_THRESHOLD
        or current_rain >= 2.0
    ):
        status = "DELAY_IRRIGATION"
        badge_text = "⏸️ DELAY / REVIEW IRRIGATION"
        severity = "warning"
        headline = "Delay or review irrigation — rainfall is forecast."
        reason = (
            f"Soil moisture is currently adequate ({soil_moisture:.1f}%) and meaningful rainfall is forecast "
            f"({max_rain_prob:.0f}% probability, {total_rain_expected:.1f} mm expected in the next 48–72 hours). "
            "Scheduled irrigation may be unnecessary and could cause waterlogging, nutrient leaching, or root hypoxia."
        )
        decision_basis = "Strong"
        timeline = "Reassess after the next forecast update"
        priority_tier = "B"
        rule_name = "Adequate soil moisture with significant rainfall forecast"

    # ──────────────────────────────────────────────────────────────────────
    # PRIORITY C — LOW/DEPLETED MOISTURE (<= 35%) + DRY FORECAST
    # ──────────────────────────────────────────────────────────────────────
    elif (
        soil_moisture <= SmartIrrigationConfig.SOIL_MOISTURE_DEPLETED
        and max_rain_prob < SmartIrrigationConfig.IMMINENT_RAIN_PROB_THRESHOLD
        and total_rain_expected < 5.0
    ):
        status = "CONSIDER_IRRIGATION"
        badge_text = "💧 CONSIDER IRRIGATION"
        severity = "primary"
        headline = "Consider irrigation — soil moisture is depleted."
        reason = (
            f"Manual soil moisture ({soil_moisture:.1f}%) is below the configured management depletion threshold "
            f"({SmartIrrigationConfig.SOIL_MOISTURE_DEPLETED:.0f}%) and significant rainfall is not expected in the "
            f"near-term forecast ({max_rain_prob:.0f}% chance, {total_rain_expected:.1f} mm expected). Supplemental "
            "irrigation is recommended to sustain plant transpiration."
        )
        decision_basis = "Strong"
        timeline = "Review irrigation now"
        priority_tier = "C"
        rule_name = "Depleted soil moisture with dry near-term forecast"

    # ──────────────────────────────────────────────────────────────────────
    # PRIORITY D — MODERATE / UNCERTAIN CONDITIONS
    # ──────────────────────────────────────────────────────────────────────
    elif (
        (SmartIrrigationConfig.TRANSITIONAL_RAIN_PROB_LOW <= max_rain_prob <= SmartIrrigationConfig.TRANSITIONAL_RAIN_PROB_HIGH)
        or (2.0 <= total_rain_expected < SmartIrrigationConfig.IMMINENT_RAIN_AMOUNT_THRESHOLD)
        or (soil_moisture <= 45.0 and max_rain_prob >= 35.0)
    ):
        status = "MONITOR_AND_REASSESS"
        badge_text = "🔍 MONITOR & REASSESS"
        severity = "info"
        headline = "Monitor field conditions and reassess."
        reason = (
            f"Transitional meteorological conditions: rain probability is {max_rain_prob:.0f}% with "
            f"{total_rain_expected:.1f} mm precipitation expected, while manual soil moisture is {soil_moisture:.1f}%. "
            "Because neither drought stress nor guaranteed soaking rain is indicated, hold scheduled watering "
            "and reassess before the next irrigation cycle."
        )
        decision_basis = "Conditional"
        timeline = "Reassess within 6–12 hours"
        priority_tier = "D"
        rule_name = "Transitional forecast conditions with moderate field moisture"

    # ──────────────────────────────────────────────────────────────────────
    # PRIORITY E — ADEQUATE MOISTURE + DRY/NORMAL CONDITIONS
    # ──────────────────────────────────────────────────────────────────────
    else:
        status = "MAINTAIN_NORMAL_PLAN"
        badge_text = "✅ MAINTAIN NORMAL IRRIGATION PLAN"
        severity = "success"
        headline = "Maintain normal irrigation plan."
        reason = (
            f"Manual soil moisture ({soil_moisture:.1f}%) is currently adequate (above the "
            f"{SmartIrrigationConfig.SOIL_MOISTURE_DEPLETED:.0f}% depletion threshold) and no extreme meteorological "
            f"stress is forecast ({max_rain_prob:.0f}% rain prob, {total_rain_expected:.1f} mm expected). "
            "Standard agricultural management operations can proceed."
        )
        decision_basis = "Moderate"
        timeline = "Reassess after the next forecast update"
        priority_tier = "E"
        rule_name = "Adequate soil moisture with stable dry/normal weather"

    # 3. Contextual modifiers
    crop_context_text = SmartIrrigationConfig.get_crop_context(crop)
    stage_context_text = SmartIrrigationConfig.get_stage_context(growth_stage)
    method_context_text = SmartIrrigationConfig.get_method_context(irrigation_method)

    # 4. Construct Why This Recommendation Breakdown
    why_inputs = [
        f"Manual soil moisture: {soil_moisture:.1f}% (Critical threshold: {SmartIrrigationConfig.SOIL_MOISTURE_CRITICAL:.0f}%, Depleted: {SmartIrrigationConfig.SOIL_MOISTURE_DEPLETED:.0f}%)",
        f"Near-term rain probability: {max_rain_prob:.0f}%",
        f"Expected near-term rainfall: {total_rain_expected:.1f} mm",
        f"Ambient temperature: {temp:.1f}°C",
        f"Target crop: {crop}",
        f"Growth stage: {growth_stage}",
    ]

    return {
        "status": status,
        "badge_text": badge_text,
        "severity": severity,
        "headline": headline,
        "reason": reason,
        "decision_basis": decision_basis,
        "timeline": timeline,
        "priority_tier": priority_tier,
        "why_recommendation": {
            "inputs_evaluated": why_inputs,
            "rule_triggered": rule_name,
            "explanation": (
                f"Rule triggered under Priority Tier {priority_tier}: {rule_name}. "
                f"Recommendation based on evaluated soil moisture ({soil_moisture:.1f}%), rain probability ({max_rain_prob:.0f}%), "
                f"and expected rainfall ({total_rain_expected:.1f} mm)."
            ),
        },
        "crop_stage_context": {
            "crop": crop,
            "growth_stage": growth_stage,
            "crop_guidance": crop_context_text,
            "stage_guidance": stage_context_text,
        },
        "irrigation_method_context": {
            "method": irrigation_method,
            "guidance": method_context_text,
        },
        "future_rain_notice": future_notice,
        "metrics_evaluated": {
            "soil_moisture_pct": soil_moisture,
            "max_rain_prob_pct": max_rain_prob,
            "expected_rain_mm": round(total_rain_expected, 1),
            "temperature_c": temp,
            "critical_threshold_pct": SmartIrrigationConfig.SOIL_MOISTURE_CRITICAL,
            "depleted_threshold_pct": SmartIrrigationConfig.SOIL_MOISTURE_DEPLETED,
            "adequate_threshold_pct": SmartIrrigationConfig.SOIL_MOISTURE_ADEQUATE,
        },
    }
