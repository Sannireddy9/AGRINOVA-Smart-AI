"""
AgriSmart AI — Sustainability Scoring Rules Engine
===================================================
Pure, deterministic, explainable rule functions that evaluate Water Efficiency,
Resource Use, and Crop Health to produce transparent component ratings and
practical sustainability recommendations.

NOTICE:
- All formulas and point adjustments are deterministic agrometeorological heuristics.
- No machine learning model or fake confidence scores.
- Zero-IoT compliance: all inputs are farmer-provided field observations.
- Never makes unsupported claims regarding exact liters saved, percentage water
  reductions, or kg of emissions/runoff prevented.
"""

from __future__ import annotations

from typing import Any
from model.sustainability.config import SustainabilityConfig
from model.sustainability.crop_requirements import CropRequirement


def evaluate_water_efficiency(
    soil_moisture: float,
    irrigation_method: str,
    water_availability: str,
    weather_summary: dict[str, Any] | None = None,
    crop_requirement: CropRequirement | None = None,
    growth_stage: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate agricultural water efficiency based on delivery system,
    soil moisture alignment, water availability, and rainfall forecast (if available).
    """
    # 1. Base score from irrigation delivery system
    base_score = SustainabilityConfig.IRRIGATION_BASE_SCORES.get(irrigation_method, 70)
    adjustments: list[str] = []
    current_score = float(base_score)
    breakdown_items: list[dict[str, Any]] = [
        {
            "factor": "Irrigation Delivery System",
            "detail": f"{irrigation_method} Method Baseline",
            "points": base_score,
        }
    ]

    # 2. Moisture alignment
    if 35.0 <= soil_moisture <= 65.0:
        if irrigation_method in ("Drip", "Rainfed"):
            current_score += 5.0
            adjustments.append("Maintains optimal root-zone moisture (35–65%) with high delivery precision (+5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Optimal moisture maintenance (35–65%)",
                "points": "+5",
            })
        else:
            adjustments.append("Soil moisture is currently in the adequate range (35–65%)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Adequate moisture level observed",
                "points": "0",
            })
    elif soil_moisture > 80.0:
        if irrigation_method in ("Flood", "Sprinkler"):
            current_score -= 15.0
            adjustments.append("Soil moisture is very high (>80%); active irrigation risks runoff, waterlogging, and nutrient leaching (-15 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "High saturation (>80%) deep percolation/runoff risk",
                "points": "-15",
            })
        elif irrigation_method == "Drip":
            current_score -= 5.0
            adjustments.append("Soil moisture exceeds field capacity (>80%); micro-irrigation should be paused (-5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Over-irrigation beyond field capacity",
                "points": "-5",
            })
        elif irrigation_method == "Rainfed":
            current_score += 5.0
            adjustments.append("High soil moisture achieved naturally via precipitation without pumping energy (+5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Natural rainfall moisture retention without pumping",
                "points": "+5",
            })
    elif 25.0 <= soil_moisture < 35.0:
        if irrigation_method == "Flood":
            current_score -= 5.0
            adjustments.append("Depleted soil moisture (25–35%) requires large surface recharge volume (-5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Depleted root moisture requires large flood recharge",
                "points": "-5",
            })
        elif irrigation_method == "Sprinkler":
            current_score -= 5.0
            adjustments.append("Depleted soil moisture (25–35%) requires extended sprinkler duration with evaporative exposure (-5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Extended sprinkler runtime on depleted moisture",
                "points": "-5",
            })
        elif irrigation_method == "Rainfed":
            current_score -= 5.0
            adjustments.append("Depleted soil moisture (25–35%) under rainfed conditions indicates emerging water stress (-5 pts)")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Emerging rainfed moisture stress",
                "points": "-5",
            })
        else:
            adjustments.append("Moderate moisture deficit managed via targeted micro-irrigation")
            breakdown_items.append({
                "factor": "Soil Moisture Balance",
                "detail": "Deficit maintained under drip precision",
                "points": "0",
            })
    elif soil_moisture < 25.0:
        if water_availability == "Scarce" and irrigation_method == "Flood":
            current_score -= 20.0
            adjustments.append("High-volume surface flood irrigation used under scarce water conditions (-20 pts)")
            breakdown_items.append({
                "factor": "Water Stress Alignment",
                "detail": "Surface flood application under scarce water availability",
                "points": "-20",
            })
        elif irrigation_method == "Flood":
            current_score -= 10.0
            adjustments.append("Critical root moisture deficit (<25%); large surface flood recharge creates high deep percolation loss (-10 pts)")
            breakdown_items.append({
                "factor": "Water Stress Alignment",
                "detail": "Severe moisture deficit flood recharge loss",
                "points": "-10",
            })
        elif irrigation_method == "Sprinkler":
            current_score -= 10.0
            adjustments.append("Critical soil dryness (<25%); overhead spray incurs high evaporative drift on dry ground (-10 pts)")
            breakdown_items.append({
                "factor": "Water Stress Alignment",
                "detail": "Overhead evaporation loss on parched soil",
                "points": "-10",
            })
        elif irrigation_method == "Drip":
            current_score -= 5.0
            adjustments.append("Critical root moisture deficit detected (<25%); requires immediate targeted recharge (-5 pts)")
            breakdown_items.append({
                "factor": "Water Stress Alignment",
                "detail": "Critical moisture deficit mitigation via drip",
                "points": "-5",
            })
        elif irrigation_method == "Rainfed":
            current_score -= 15.0
            adjustments.append("Severe rainfed drought deficit (<25%) observed (-15 pts)")
            breakdown_items.append({
                "factor": "Water Stress Alignment",
                "detail": "Severe rainfed drought deficit",
                "points": "-15",
            })

    # 3. Regional Water Availability Alignment
    if water_availability == "Scarce":
        if irrigation_method == "Drip":
            current_score += 5.0
            adjustments.append("Watershed stewardship bonus: practicing precision micro-irrigation in a water-scarce area (+5 pts)")
            breakdown_items.append({
                "factor": "Watershed Stewardship",
                "detail": "Precision micro-irrigation practiced in water-scarce basin",
                "points": "+5",
            })
        elif irrigation_method == "Sprinkler":
            current_score -= 5.0
            adjustments.append("Overhead sprinkler evaporation loss penalty in a water-scarce watershed (-5 pts)")
            breakdown_items.append({
                "factor": "Watershed Stewardship",
                "detail": "Overhead spray losses in water-scarce basin",
                "points": "-5",
            })
        elif irrigation_method == "Flood" and soil_moisture >= 25.0:
            current_score -= 5.0
            adjustments.append("Surface flood delivery penalty in a water-scarce watershed (-5 pts)")
            breakdown_items.append({
                "factor": "Watershed Stewardship",
                "detail": "Surface flood delivery in water-scarce basin",
                "points": "-5",
            })

    # 4. Weather context alignment (if available)
    if weather_summary and weather_summary.get("has_rain_forecast"):
        rain_prob = float(weather_summary.get("max_rain_prob", 0.0))
        rain_mm = float(weather_summary.get("expected_rain_mm", 0.0))
        if rain_prob >= 60.0 or rain_mm >= 8.0:
            if irrigation_method in ("Drip", "Rainfed"):
                current_score += 5.0
                adjustments.append(f"Near-term rainfall forecast ({rain_prob:.0f}%, {rain_mm:.1f} mm) aligns with efficient irrigation management (+5 pts)")
                breakdown_items.append({
                    "factor": "Weather Forecast Opportunity",
                    "detail": f"Forecast precipitation ({rain_mm:.1f} mm) utilized to defer pumping",
                    "points": "+5",
                })
            elif soil_moisture > 70.0 and irrigation_method == "Flood":
                current_score -= 10.0
                adjustments.append(f"Imminent rainfall ({rain_mm:.1f} mm) combined with saturated soil and flood delivery creates high percolation loss (-10 pts)")
                breakdown_items.append({
                    "factor": "Weather Forecast Opportunity",
                    "detail": "High percolation loss risk from imminent rain on saturated soil",
                    "points": "-10",
                })

    # 4.5 Data-driven crop agronomic context alignment (if crop_requirement provided)
    if crop_requirement is not None:
        if not crop_requirement.is_available:
            breakdown_items.append({
                "factor": "Crop Agronomic Alignment",
                "detail": "AgriSmart project-defined alignment rule: Crop data not available in database; standard practice scoring applied without crop adjustment",
                "points": "0",
            })
        else:
            # A. Crop water demand vs regional water availability (AgriSmart project-defined alignment rule)
            if crop_requirement.agrismart_water_need_category == "Very_High" and water_availability == "Scarce" and irrigation_method == "Flood":
                current_score -= 10.0
                adjustments.append("AgriSmart project-defined alignment rule: High seasonal water requirement crop (Very_High) flood-irrigated in water-scarce basin risks excessive aquifer drawdown (-10 pts)")
                breakdown_items.append({
                    "factor": "Crop Water Demand Alignment",
                    "detail": "AgriSmart project-defined alignment rule: High water-need crop flood-irrigated in water-scarce basin",
                    "points": "-10",
                })
            elif crop_requirement.agrismart_water_need_category == "Low" and water_availability == "Scarce":
                current_score += 5.0
                adjustments.append("AgriSmart project-defined alignment rule: Drought-resilient crop selection (Low water need) aligns with regional water scarcity (+5 pts)")
                breakdown_items.append({
                    "factor": "Crop Water Demand Alignment",
                    "detail": "AgriSmart project-defined alignment rule: Drought-resilient crop selection matches water scarcity",
                    "points": "+5",
                })
            elif crop_requirement.agrismart_water_need_category == "Very_High" and water_availability == "Scarce" and irrigation_method == "Drip":
                current_score += 5.0
                adjustments.append("AgriSmart project-defined alignment rule: High water-need crop in water-scarce basin managed with precision micro-irrigation stewardship (+5 pts)")
                breakdown_items.append({
                    "factor": "Crop Water Demand Alignment",
                    "detail": "AgriSmart project-defined alignment rule: High water-need crop managed with precision micro-irrigation",
                    "points": "+5",
                })

            # B. Growth stage sensitivity alignment (AgriSmart project-defined alignment rule)
            # Only evaluated when source data explicitly supports critical stage sensitivity (Ky > 1.0)
            if crop_requirement.is_stage_critical(growth_stage) and soil_moisture < 30.0:
                current_score -= 5.0
                adjustments.append(f"AgriSmart project-defined alignment rule: Field moisture deficit ({soil_moisture:.1f}%) during documented critical stage ({growth_stage}) threatens yield formation (-5 pts)")
                breakdown_items.append({
                    "factor": "Growth Stage Sensitivity Alignment",
                    "detail": f"AgriSmart project-defined alignment rule: Moisture deficit during documented critical stage ({growth_stage})",
                    "points": "-5",
                })
            elif growth_stage and growth_stage.lower() in ("maturity", "ripening", "harvest"):
                if soil_moisture < 30.0:
                    adjustments.append(f"AgriSmart project-defined alignment rule: Low soil moisture during ripening/maturity stage ({growth_stage}) is agronomically acceptable for dry-down (0 pts)")
                    breakdown_items.append({
                        "factor": "Growth Stage Sensitivity Alignment",
                        "detail": f"AgriSmart project-defined alignment rule: Ripening/maturity dry-down stage alignment",
                        "points": "0",
                    })

            # C. Soil water depletion sensitivity context (p)
            # Stored and presented strictly as documented FAO 56 context; does not adjust points without field TAW data
            if crop_requirement.fao_depletion_fraction_p > 0.0:
                breakdown_items.append({
                    "factor": "Crop Depletion Sensitivity Context",
                    "detail": (
                        f"FAO depletion fraction p={crop_requirement.fao_depletion_fraction_p:.2f} "
                        f"(documented soil water depletion fraction before stress per FAO 56 Table 22)"
                    ),
                    "points": "0",
                })

    # 5. Clamp score
    final_score = int(max(0.0, min(100.0, round(current_score))))

    # 5. Determine qualitative status
    if final_score >= 85:
        status = "EXCELLENT"
        severity = "success"
    elif final_score >= 70:
        status = "GOOD"
        severity = "info"
    elif final_score >= 50:
        status = "MODERATE"
        severity = "warning"
    else:
        status = "POOR"
        severity = "danger"

    # 6. Construct narrative explanation
    delivery_desc = {
        "Drip": "High-efficiency drip micro-irrigation maximizes application uniformity directly to the root zone.",
        "Sprinkler": "Overhead sprinkler irrigation provides moderate efficiency with some evaporative exposure.",
        "Flood": "Surface/flood irrigation involves higher deep percolation and evaporative losses.",
        "Rainfed": "Rainfed cultivation avoids supplemental water extraction and energy consumption.",
    }.get(irrigation_method, "Standard irrigation system utilized.")

    reason = (
        f"Water efficiency score is {final_score}/100 ({status}). "
        f"{delivery_desc} Manual soil moisture is {soil_moisture:.1f}% under '{water_availability}' water availability. "
        + (" ".join(adjustments) if adjustments else "Standard moisture alignment observed.")
    )

    return {
        "component": "water_efficiency",
        "title": "Water Efficiency",
        "score": final_score,
        "base_weight": SustainabilityConfig.WEIGHT_WATER_EFFICIENCY,
        "status": status,
        "severity": severity,
        "is_assessed": True,
        "reason": reason,
        "breakdown": breakdown_items,
        "inputs": {
            "soil_moisture_pct": soil_moisture,
            "irrigation_method": irrigation_method,
            "water_availability": water_availability,
            "weather_considered": bool(weather_summary),
            "crop": crop_requirement.display_name if crop_requirement else None,
            "growth_stage": growth_stage,
        },
    }


def evaluate_resource_use(
    nutrient_practice: str,
    soil_cover: str = "Bare_Soil",
) -> dict[str, Any]:
    """
    Evaluate agricultural resource use based on nutrient management practices
    and soil conservation / mulching practices.
    """
    # 1. Base score from nutrient practice
    base_score = SustainabilityConfig.NUTRIENT_BASE_SCORES.get(nutrient_practice, 70)
    adjustments: list[str] = []
    current_score = float(base_score)
    breakdown_items: list[dict[str, Any]] = [
        {
            "factor": "Nutrient Management",
            "detail": f"{nutrient_practice.replace('_', ' ')} Practice",
            "points": base_score,
        }
    ]

    # 2. Mulching / soil conservation practice
    cover_pts = SustainabilityConfig.SOIL_COVER_SCORES.get(soil_cover, 0)
    if cover_pts > 0:
        current_score += float(cover_pts)
        adjustments.append(f"Soil conservation practice ({soil_cover.replace('_', ' ')}) practiced (+{cover_pts} pts)")
        breakdown_items.append({
            "factor": "Soil Conservation & Cover",
            "detail": f"{soil_cover.replace('_', ' ')} Layer",
            "points": f"+{cover_pts}",
        })
    else:
        adjustments.append("Bare soil management without protective mulching or cover crop layer (0 pts)")
        breakdown_items.append({
            "factor": "Soil Conservation & Cover",
            "detail": "Conventional bare soil tillage",
            "points": "0",
        })

    # 3. Clamp score
    final_score = int(max(0.0, min(100.0, round(current_score))))

    # 4. Determine qualitative status
    if final_score >= 85:
        status = "EXCELLENT"
        severity = "success"
    elif final_score >= 70:
        status = "GOOD"
        severity = "info"
    elif final_score >= 50:
        status = "MODERATE"
        severity = "warning"
    else:
        status = "POOR"
        severity = "danger"

    # 5. Construct narrative explanation
    nutrient_desc = {
        "Organic": "Organic / bio-fertilizer practices preserve soil microbial diversity and minimize chemical runoff.",
        "Integrated": "Integrated Nutrient Management (INM) balances targeted synthetic inputs with organic amendments.",
        "Moderate_Chemical": "Moderate synthetic chemical fertilizer application with split timing.",
        "Intensive_Chemical": "Intensive synthetic chemical fertilization carries elevated risks of soil acidification and nutrient runoff.",
        "Synthetic_Heavy": "Heavy synthetic chemical fertilization carries elevated risks of soil acidification and nutrient runoff.",
    }.get(nutrient_practice, "Standard nutrient management practiced.")

    reason = (
        f"Resource-use score is {final_score}/100 ({status}). "
        f"{nutrient_desc} "
        + " ".join(adjustments)
    )

    return {
        "component": "resource_use",
        "title": "Resource Use",
        "score": final_score,
        "base_weight": SustainabilityConfig.WEIGHT_RESOURCE_USE,
        "status": status,
        "severity": severity,
        "is_assessed": True,
        "reason": reason,
        "breakdown": breakdown_items,
        "inputs": {
            "nutrient_practice": nutrient_practice,
            "soil_cover": soil_cover,
        },
    }


def evaluate_crop_health(
    crop_health_status: str | None,
) -> dict[str, Any]:
    """
    Evaluate crop health contribution based on available crop health signals
    or image-based disease screening.
    Supports missing-data handling (returns is_assessed=False, score=None).
    """
    clean_status = (crop_health_status or "").strip()

    if not clean_status or clean_status == "Not_Assessed":
        return {
            "component": "crop_health",
            "title": "Crop Health",
            "score": None,
            "base_weight": SustainabilityConfig.WEIGHT_CROP_HEALTH,
            "status": "NOT_ASSESSED",
            "severity": "neutral",
            "is_assessed": False,
            "reason": (
                "Crop Health: Not assessed — no field crop-health inspection or disease diagnosis "
                "was provided. This component is excluded from the score denominator, and the "
                "remaining components are renormalized."
            ),
            "breakdown": [],
            "inputs": {
                "crop_health_status": "Not_Assessed",
            },
        }

    raw_score = SustainabilityConfig.CROP_HEALTH_SCORES.get(clean_status)
    if raw_score is None:
        return {
            "component": "crop_health",
            "title": "Crop Health",
            "score": None,
            "base_weight": SustainabilityConfig.WEIGHT_CROP_HEALTH,
            "status": "NOT_ASSESSED",
            "severity": "neutral",
            "is_assessed": False,
            "reason": "Crop Health: Not assessed — invalid or omitted observation signal.",
            "breakdown": [],
            "inputs": {"crop_health_status": clean_status},
        }

    final_score = raw_score

    if final_score >= 85:
        status = "EXCELLENT"
        severity = "success"
        explanation = "Crop health signal indicates healthy foliage with no active foliar disease observed."
    elif final_score >= 70:
        status = "GOOD"
        severity = "info"
        explanation = "Minor physiological stress or leaf discoloration noted under field monitoring."
    elif final_score >= 40:
        status = "MODERATE"
        severity = "warning"
        explanation = "Active foliar disease or fungal lesion symptoms detected; prompt management advised."
    else:
        status = "POOR"
        severity = "danger"
        explanation = "Significant foliar disease damage observed; active cultural/therapeutic intervention required."

    reason = (
        f"Crop health score is {final_score}/100 ({status}). "
        f"{explanation} Note: Crop health component is based on the available crop-health observation signal."
    )

    return {
        "component": "crop_health",
        "title": "Crop Health",
        "score": final_score,
        "base_weight": SustainabilityConfig.WEIGHT_CROP_HEALTH,
        "status": status,
        "severity": severity,
        "is_assessed": True,
        "reason": reason,
        "breakdown": [
            {
                "factor": "Foliar Diagnostic Observation",
                "detail": clean_status.replace("_", " "),
                "points": final_score,
            }
        ],
        "inputs": {
            "crop_health_status": clean_status,
        },
    }


from model.sustainability.recommendations import generate_sustainability_recommendations
