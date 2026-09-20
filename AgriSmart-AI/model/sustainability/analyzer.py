"""
AgriSmart AI — Sustainability Score Coordinator & Analyzer
===========================================================
Coordinates component evaluation, missing-data handling, dynamic weight
renormalization, and structured payload formatting.
"""

from __future__ import annotations

import datetime
from typing import Any

from model.sustainability.config import SustainabilityConfig
from model.sustainability.crop_requirements import get_crop_requirement, CropRequirement
from model.sustainability.rules import (
    evaluate_water_efficiency,
    evaluate_resource_use,
    evaluate_crop_health,
    generate_sustainability_recommendations,
)


def analyze_sustainability_score(
    soil_moisture: float,
    irrigation_method: str,
    water_availability: str,
    nutrient_practice: str,
    soil_cover: str = "Bare_Soil",
    crop_health_status: str | None = None,
    crop: str = "Other / Not Listed",
    growth_stage: str = "Vegetative",
    weather_summary: dict[str, Any] | None = None,
    weather_status: str = "UNAVAILABLE",
    crop_requirement: CropRequirement | None = None,
) -> dict[str, Any]:
    """
    Execute full deterministic sustainability scoring analysis.
    Implements dynamic weight renormalization for missing/unassessed components.
    """
    # 0. Resolve data-driven crop requirement context
    crop_req = crop_requirement or get_crop_requirement(crop, growth_stage)

    # 1. Evaluate individual components
    water_eval = evaluate_water_efficiency(
        soil_moisture=soil_moisture,
        irrigation_method=irrigation_method,
        water_availability=water_availability,
        weather_summary=weather_summary,
        crop_requirement=crop_req,
        growth_stage=growth_stage,
    )

    resource_eval = evaluate_resource_use(
        nutrient_practice=nutrient_practice,
        soil_cover=soil_cover,
    )

    health_eval = evaluate_crop_health(
        crop_health_status=crop_health_status,
    )

    # 2. Dynamic weight renormalization
    all_components = [water_eval, resource_eval, health_eval]
    assessed_components = [c for c in all_components if c.get("is_assessed") and c.get("score") is not None]

    total_available_weight = sum(c["base_weight"] for c in assessed_components)
    is_renormalized = total_available_weight < 0.999

    weighted_sum = sum(float(c["score"]) * c["base_weight"] for c in assessed_components)

    if total_available_weight > 0.0:
        final_score_raw = weighted_sum / total_available_weight
    else:
        final_score_raw = 0.0

    final_score_float = round(final_score_raw, 1)
    final_score_int = round(final_score_raw)
    final_score_int = max(0, min(100, final_score_int))

    # 3. Calculate effective percentage weights for each component
    for c in all_components:
        if c.get("is_assessed") and c.get("score") is not None and total_available_weight > 0.0:
            eff_w = c["base_weight"] / total_available_weight
            c["effective_weight"] = round(eff_w, 4)
            c["effective_weight_pct"] = round(eff_w * 100.0, 1)
        else:
            c["effective_weight"] = 0.0
            c["effective_weight_pct"] = 0.0

    # 4. Formulate step-by-step arithmetic calculation string
    calc_terms = [
        f"{c['title']}: {c['score']} × {c['base_weight']*100:.0f}%"
        for c in assessed_components
    ]
    terms_joined = " + ".join(calc_terms)

    if is_renormalized:
        calc_str = (
            f"({terms_joined}) / {total_available_weight*100:.0f}% = "
            f"{weighted_sum:.2f} / {total_available_weight:.2f} = "
            f"{final_score_float} → {final_score_int} / 100"
        )
        data_status = "PARTIAL_RENORMALIZED"
    else:
        calc_str = (
            f"({terms_joined}) / 100% = "
            f"{weighted_sum:.2f} = "
            f"{final_score_float} → {final_score_int} / 100"
        )
        data_status = "COMPLETE"

    # 5. Determine qualitative category
    category_meta = SustainabilityConfig.get_category_meta(final_score_int)

    # 6. Collect farm inputs for auditing & recommendations
    farm_inputs = {
        "crop": crop,
        "growth_stage": growth_stage,
        "soil_moisture_pct": soil_moisture,
        "irrigation_method": irrigation_method,
        "water_availability": water_availability,
        "nutrient_practice": nutrient_practice,
        "soil_cover": soil_cover,
        "crop_health_status": crop_health_status or "Not_Assessed",
        "weather_status": weather_status,
    }

    # 7. Generate targeted actionable recommendations
    recommendations = generate_sustainability_recommendations(
        water_eval=water_eval,
        resource_eval=resource_eval,
        health_eval=health_eval,
        field_inputs=farm_inputs,
        crop_requirement=crop_req,
        weather_summary=weather_summary,
        weather_status=weather_status,
    )

    # 8. Data-driven crop sustainability context
    crop_context = {
        "crop_name": crop_req.display_name if crop_req.is_available else (crop or "Not Specified"),
        "growth_stage": growth_stage,
        "crop_data_status": crop_req.crop_data_status,
        "is_available": crop_req.is_available,
        "scientific_name": crop_req.scientific_name if crop_req.is_available else "N/A",
        "seasonal_water_need_mm": (
            f"{crop_req.seasonal_water_need_mm_min:.0f} – {crop_req.seasonal_water_need_mm_max:.0f} mm"
            if crop_req.is_available else "N/A"
        ),
        "seasonal_water_need_mm_min": crop_req.seasonal_water_need_mm_min if crop_req.is_available else None,
        "seasonal_water_need_mm_max": crop_req.seasonal_water_need_mm_max if crop_req.is_available else None,
        "agrismart_water_need_category": crop_req.agrismart_water_need_category if crop_req.is_available else "Unknown",
        "fao_depletion_fraction_p": crop_req.fao_depletion_fraction_p if crop_req.is_available else None,
        "critical_growth_stages": crop_req.critical_growth_stages if crop_req.is_available else [],
        "is_critical_stage_active": crop_req.is_stage_critical(growth_stage) if crop_req.is_available else False,
        "growth_stage_information": crop_req.growth_stage_information if crop_req.is_available else "Crop-specific agronomic parameters are not available in the database for this crop.",
        "source_reference": crop_req.source_reference if crop_req.is_available else "N/A",
        "scoring_basis_note": (
            "FAO data provides crop water-use and water-stress context. AgriSmart applies "
            "transparent project-defined alignment rules to convert that context into an advisory "
            "sustainability score. Numerical adjustments are not FAO-certified metrics."
        ),
    }

    return {
        "status": "success",
        "score": final_score_int,
        "score_float": final_score_float,
        "category": category_meta["category"],
        "severity": category_meta["severity"],
        "badge_class": category_meta["badge_class"],
        "description": category_meta["description"],
        "components": {
            "water_efficiency": water_eval,
            "resource_use": resource_eval,
            "crop_health": health_eval,
        },
        "weights": {
            "base": {
                "water_efficiency": SustainabilityConfig.WEIGHT_WATER_EFFICIENCY,
                "resource_use": SustainabilityConfig.WEIGHT_RESOURCE_USE,
                "crop_health": SustainabilityConfig.WEIGHT_CROP_HEALTH,
            },
            "effective": {
                "water_efficiency": water_eval["effective_weight"],
                "resource_use": resource_eval["effective_weight"],
                "crop_health": health_eval["effective_weight"],
            },
            "is_renormalized": is_renormalized,
            "total_evaluated_weight": round(total_available_weight, 4),
        },
        "calculation": calc_str,
        "recommendations": recommendations,
        "data_status": data_status,
        "weather_status": weather_status,
        "disclaimer": SustainabilityConfig.DISCLAIMER,
        "farm_inputs": farm_inputs,
        "crop_sustainability_context": crop_context,
        "evaluated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
