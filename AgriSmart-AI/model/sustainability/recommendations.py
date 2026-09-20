"""
AgriSmart AI — Sustainability Recommendations Engine
===================================================
Generates prioritized, input-sensitive, deterministic agronomic recommendations.

NOTICE:
- All recommendations are derived from deterministic agronomic rules and actual field inputs.
- No machine learning, no random selections, no fake confidence values.
- Zero-IoT compliance: all inputs are farmer-provided field observations.
- Never makes unsupported causal claims regarding exact water savings, guaranteed yields,
  or unverified environmental outcomes.
- "Critical", "High", "Medium", and "Low" are AgriSmart project-defined recommendation
  priorities, not official international agricultural classifications.
"""

from __future__ import annotations

from typing import Any
from model.sustainability.crop_requirements import CropRequirement


PRIORITY_WEIGHTS = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}


def _build_water_recommendation(
    soil_moisture: float,
    irrigation_method: str,
    water_availability: str,
    weather_summary: dict[str, Any] | None = None,
    weather_status: str = "UNAVAILABLE",
    crop_requirement: CropRequirement | None = None,
    growth_stage: str | None = None,
) -> dict[str, str]:
    """
    Generate an input-sensitive water stewardship recommendation based on
    moisture condition bands, irrigation delivery method, water availability,
    weather intelligence, and crop stage context.
    """
    clean_stage = (growth_stage or "Vegetative").strip() or "Vegetative"
    clean_method = irrigation_method.strip() if irrigation_method else "Flood"
    clean_water = water_availability.strip() if water_availability else "Moderate"

    # 1. Weather Context Analysis
    weather_note = ""
    if weather_summary and weather_status in ("LIVE", "DEMO"):
        max_rain_prob = float(weather_summary.get("max_rain_prob", 0.0))
        expected_rain_mm = float(weather_summary.get("expected_rain_mm", 0.0))
        is_demo = weather_status == "DEMO" or bool(weather_summary.get("is_mock", False))
        prefix = "[DEMO Weather Context] " if is_demo else "[Live Weather Context] "

        # Distinguish significant rain from high probability with negligible rain
        if expected_rain_mm >= 4.0 and max_rain_prob >= 40.0:
            if soil_moisture > 65.0:
                weather_note = (
                    f"{prefix}Significant near-term rainfall is forecast ({expected_rain_mm:.1f} mm, "
                    f"max prob {max_rain_prob:.0f}%). Consider delaying any supplemental irrigation and reassess after rain."
                )
            elif soil_moisture < 25.0:
                weather_note = (
                    f"{prefix}Significant rainfall is in the forecast ({expected_rain_mm:.1f} mm, "
                    f"max prob {max_rain_prob:.0f}%). Review irrigation timing against expected rainfall and crop stress before applying water."
                )
            else:
                weather_note = (
                    f"{prefix}Near-term rainfall expected ({expected_rain_mm:.1f} mm, "
                    f"max prob {max_rain_prob:.0f}%). Consider taking advantage of rainfall to supplement field moisture."
                )
        elif max_rain_prob >= 50.0 and expected_rain_mm < 2.0:
            weather_note = (
                f"{prefix}Rainfall probability is {max_rain_prob:.0f}%, but expected accumulation is negligible "
                f"({expected_rain_mm:.1f} mm). Do not rely solely on precipitation to recharge root-zone moisture."
            )
    elif weather_status == "UNAVAILABLE" or not weather_summary:
        weather_note = "Weather data unavailable — base irrigation decisions on manual moisture checks and local forecast information."

    # 2. Crop & Growth Stage Context
    stage_note = ""
    is_critical_stage = False
    if crop_requirement and crop_requirement.is_available:
        is_critical_stage = crop_requirement.is_stage_critical(clean_stage)
        if is_critical_stage:
            stage_note = (
                f"The selected crop/stage ({crop_requirement.display_name} at {clean_stage}) is identified as "
                f"moisture-sensitive in the available crop data."
            )
        elif clean_stage.lower() in ("maturity", "ripening", "harvest"):
            stage_note = (
                f"The crop is in the {clean_stage} stage; lower soil moisture is agronomically acceptable "
                f"for natural pre-harvest dry-down."
            )
    elif crop_requirement and not crop_requirement.is_available:
        stage_note = "Crop-specific requirement data is unavailable, so this recommendation is based on the reported field condition."

    # 3. Moisture Bands Evaluation
    # Band A: Very High / Saturated (> 80%)
    if soil_moisture > 80.0:
        priority = "Critical" if clean_method == "Flood" and clean_water == "Scarce" else "High"

        if clean_method == "Flood":
            title = "Hold Unnecessary Flood Irrigation on Saturated Soil"
            method_advice = (
                "Surface flood application on saturated soil risks waterlogging and nutrient leaching; "
                "inspect drainage channels."
            )
        elif clean_method == "Drip":
            title = "Hold Micro-Irrigation Cycles on Saturated Soil"
            method_advice = "Hold scheduled micro-irrigation cycles while soil moisture remains high and monitor root-zone aeration."
        elif clean_method == "Sprinkler":
            title = "Suspend Overhead Sprinkler Application"
            method_advice = "Suspend overhead sprinkling to avoid excess water accumulation and prolonged canopy wetness."
        else:  # Rainfed
            title = "Monitor Field Drainage and Root-Zone Aeration"
            method_advice = "High moisture was supplied by precipitation; ensure field drainage outlets remain unobstructed."

        action_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}%. Hold unnecessary irrigation while soil moisture remains very high "
            f"and monitor drainage/root-zone conditions before the next irrigation event.",
            method_advice,
        ]
        if weather_note:
            action_parts.append(weather_note)

        why_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}% under {clean_method} irrigation.",
            "Applying additional water under saturated conditions is unnecessary, risks root-zone oxygen stress, and consumes avoidable pumping energy.",
        ]
        if clean_water == "Scarce":
            why_parts.append("Water availability is reported as scarce. Conserving water while soil moisture is abundant preserves limited farm reserves.")
        elif clean_water == "Abundant":
            why_parts.append("Even with abundant water availability, holding irrigation when soil is saturated supports healthy root development.")

        return {
            "priority": priority,
            "category": "Water Stewardship",
            "title": title,
            "action": " ".join(action_parts),
            "rationale": " ".join(why_parts),
            "why": " ".join(why_parts),
        }

    # Band B: High (65%–80%)
    if soil_moisture > 65.0:
        priority = "Medium"
        title = "Hold Irrigation Under High Soil Moisture"

        if clean_method == "Flood":
            method_advice = "Avoid unnecessary surface flood irrigation; allow the root zone to aerate naturally."
        elif clean_method == "Drip":
            method_advice = "Pause scheduled drip runs until root-zone moisture recedes toward the optimal band (35–65%)."
        elif clean_method == "Sprinkler":
            method_advice = "Hold overhead sprinkler runs until soil depletion warrants supplemental water."
        else:
            method_advice = "Adequate moisture reserve is present from rainfall; continue periodic checks."

        action_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}%. Avoid unnecessary irrigation events and monitor drainage "
            f"before scheduling the next cycle.",
            method_advice,
        ]
        if weather_note:
            action_parts.append(weather_note)

        why_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}%, which is above normal maintenance requirements.",
            "Holding irrigation allows healthy root aeration and reduces water waste.",
        ]
        return {
            "priority": priority,
            "category": "Water Stewardship",
            "title": title,
            "action": " ".join(action_parts),
            "rationale": " ".join(why_parts),
            "why": " ".join(why_parts),
        }

    # Band C: Optimal (35%–65%)
    if soil_moisture >= 35.0:
        priority = "Low"
        title = "Maintain Optimal Soil Moisture Balance"

        if clean_method == "Drip":
            method_advice = "Continue precision micro-irrigation scheduling aligned with seasonal crop stage demand."
        elif clean_method == "Flood":
            method_advice = "Maintain current moisture level and avoid unneeded flood cycles."
        elif clean_method == "Sprinkler":
            method_advice = "Operate sprinklers only when required by evaporative demand during low-wind hours."
        else:
            method_advice = "Soil moisture is well-balanced from natural precipitation; continue periodic scouting."

        action_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}%. Maintain current moisture management, avoid unnecessary irrigation, "
            f"and continue periodic manual field checks.",
            method_advice,
        ]
        if weather_note and "Significant near-term rainfall" in weather_note:
            action_parts.append(weather_note)

        why_parts = [
            f"Reported soil moisture ({soil_moisture:.0f}%) is within the balanced agronomic range (35–65%) "
            f"for general crop transpiration and nutrient uptake.",
        ]
        return {
            "priority": priority,
            "category": "Water Stewardship",
            "title": title,
            "action": " ".join(action_parts),
            "rationale": " ".join(why_parts),
            "why": " ".join(why_parts),
        }

    # Band D: Low / Depleted (25%–35%)
    if soil_moisture >= 25.0:
        priority = "High" if (clean_water == "Scarce" or is_critical_stage) else "Medium"
        title = "Review Irrigation Timing for Depleted Moisture"

        if clean_method == "Drip":
            method_advice = "Calibrate micro-irrigation runtime to deliver targeted root-zone replenishment without over-application."
        elif clean_method == "Flood":
            method_advice = "Plan controlled surface distribution rather than excessive single-event flood recharge."
        elif clean_method == "Sprinkler":
            method_advice = "Schedule sprinkler cycles during early dawn or late evening to minimize evaporative drift."
        else:
            method_advice = "Depleted moisture observed under rainfed conditions; track local rainfall forecasts and monitor foliage for moisture stress."

        action_parts = [
            f"Reported soil moisture is {soil_moisture:.0f}%. Monitor moisture closely, review irrigation timing, "
            f"and consider irrigation based on weather forecasts and crop stage.",
            method_advice,
        ]
        if clean_water == "Scarce":
            action_parts.append("Water availability is reported as scarce: prioritize irrigation only when field conditions and crop/weather context indicate a need.")
        elif clean_water == "Abundant":
            action_parts.append("Water availability is abundant: apply measured irrigation to restore moisture without over-saturating.")

        if stage_note:
            action_parts.append(stage_note)
        if weather_note:
            action_parts.append(weather_note)

        why_parts = [
            f"Soil moisture ({soil_moisture:.0f}%) has fallen into the depleted range (25–35%).",
            "Timely review helps avoid progressive crop water stress.",
        ]
        if clean_water == "Scarce":
            why_parts.append("Water availability is reported as scarce. Prioritize irrigation only when field conditions and crop/weather context indicate a need.")
        elif clean_water == "Abundant":
            why_parts.append("Water availability is abundant: apply measured irrigation to restore moisture without over-saturating.")

        return {
            "priority": priority,
            "category": "Water Stewardship",
            "title": title,
            "action": " ".join(action_parts),
            "rationale": " ".join(why_parts),
            "why": " ".join(why_parts),
        }

    # Band E: Critical Low Moisture (< 25%)
    priority = "Critical" if (clean_water == "Scarce" or is_critical_stage or clean_method == "Flood") else "High"

    # Check if flood upgrade recommendation title is appropriate
    if clean_method == "Flood":
        title = "Upgrade to Micro-Irrigation & Manage Moisture Deficit"
        method_advice = (
            "Review application timing and efficiency; avoid heavy over-application to reduce deep percolation losses. "
            "Transitioning from surface flood to localized drip or micro-sprinklers can improve delivery efficiency."
        )
    elif clean_method == "Drip":
        title = "Manage Critical Moisture Deficit with Targeted Micro-Irrigation"
        method_advice = (
            "Review irrigation timing and crop water requirements; precision micro-irrigation can deliver targeted "
            "recharge directly to active roots."
        )
    elif clean_method == "Sprinkler":
        title = "Manage Critical Moisture Deficit via Low-Evaporation Sprinkling"
        method_advice = "Schedule overhead sprinkler operations during early morning hours to maximize infiltration and reduce evaporation."
    else:  # Rainfed
        title = "Manage Severe Rainfed Moisture Deficit"
        method_advice = "Severe moisture deficit under rainfed conditions; monitor crop water stress and assess local weather forecasts."

    action_parts = [
        f"Reported soil moisture is {soil_moisture:.0f}%. Check crop water stress, review irrigation timing, "
        f"and consider replenishing root-zone moisture if weather does not provide sufficient rainfall.",
        method_advice,
    ]
    if stage_note:
        action_parts.append(stage_note)
    if weather_note:
        action_parts.append(weather_note)

    why_parts = [
        f"Soil moisture is critically low at {soil_moisture:.0f}% under {clean_method} irrigation and '{clean_water}' water availability.",
        "Prolonged moisture deficit at this level can impair plant cell turgor and vegetative development.",
    ]
    if clean_water == "Scarce":
        why_parts.append("Water availability is reported as scarce. Prioritize water stewardship and direct moisture to high-stress areas while avoiding runoff.")
    elif clean_water == "Abundant":
        why_parts.append("Water is abundant: apply measured irrigation to restore optimal moisture without over-saturating.")

    return {
        "priority": priority,
        "category": "Water Stewardship",
        "title": title,
        "action": " ".join(action_parts),
        "rationale": " ".join(why_parts),
        "why": " ".join(why_parts),
    }


def _build_resource_recommendations(
    nutrient_practice: str,
    soil_cover: str,
) -> list[dict[str, str]]:
    """
    Generate input-sensitive resource conservation recommendations
    based on nutrient stewardship and soil cover practices.
    Returns 1 or 2 targeted recommendations.
    """
    clean_nutrient = (nutrient_practice or "Integrated").strip()
    clean_cover = (soil_cover or "Bare_Soil").strip()
    cover_display = clean_cover.replace("_", " ")

    recs: list[dict[str, str]] = []

    # 1. Organic with protective cover
    if clean_nutrient == "Organic" and clean_cover in ("Mulch", "Cover_Crops", "Minimum_Tillage"):
        recs.append({
            "priority": "Low",
            "category": "Resource Conservation",
            "title": "Maintain Organic Nutrient & Soil Conservation Practices",
            "action": f"Continue current organic amendments (compost, bio-fertilizers) alongside protective soil cover ({cover_display}).",
            "rationale": (
                f"Field practices combine Organic nutrient management with {cover_display}, "
                f"supporting biological soil health, soil organic carbon retention, and natural moisture conservation."
            ),
            "why": (
                f"Field practices combine Organic nutrient management with {cover_display}, "
                f"supporting biological soil health, soil organic carbon retention, and natural moisture conservation."
            ),
        })
        return recs

    # 2. Organic with Bare Soil
    if clean_nutrient == "Organic" and clean_cover in ("Bare_Soil", "Bare", "None", ""):
        recs.append({
            "priority": "Medium",
            "category": "Resource Conservation",
            "title": "Establish Protective Soil Cover & Mulch",
            "action": "Apply organic residue mulching (crop straw or leaf litter) or sow leguminous cover crops over bare soil.",
            "rationale": (
                "While Organic nutrient practices support soil biology, bare soil remains exposed to solar radiation, "
                "accelerated surface evaporation, and erosion."
            ),
            "why": (
                "While Organic nutrient practices support soil biology, bare soil remains exposed to solar radiation, "
                "accelerated surface evaporation, and erosion."
            ),
        })
        return recs

    # 3. Integrated Nutrient Management with protective cover
    if clean_nutrient == "Integrated" and clean_cover in ("Mulch", "Cover_Crops", "Minimum_Tillage"):
        recs.append({
            "priority": "Low",
            "category": "Resource Conservation",
            "title": "Maintain Balanced Integrated Nutrient Management",
            "action": f"Maintain balanced Integrated Nutrient Management (INM) combining organic amendments with targeted mineral fertilizer, alongside {cover_display}.",
            "rationale": (
                f"Integrated nutrient management paired with {cover_display} provides balanced crop nutrition "
                f"while preserving soil structure and moisture."
            ),
            "why": (
                f"Integrated nutrient management paired with {cover_display} provides balanced crop nutrition "
                f"while preserving soil structure and moisture."
            ),
        })
        return recs

    # 4. Integrated Nutrient Management with Bare Soil
    if clean_nutrient == "Integrated" and clean_cover in ("Bare_Soil", "Bare", "None", ""):
        recs.append({
            "priority": "Medium",
            "category": "Resource Conservation",
            "title": "Establish Protective Soil Cover & Mulch",
            "action": "Apply protective crop residue mulching or establish cover crops to protect bare soil between crop rows.",
            "rationale": (
                "Integrated Nutrient Management provides balanced fertility, but bare soil remains vulnerable to "
                "surface runoff and evaporative moisture loss."
            ),
            "why": (
                "Integrated Nutrient Management provides balanced fertility, but bare soil remains vulnerable to "
                "surface runoff and evaporative moisture loss."
            ),
        })
        return recs

    # 5. Moderate Chemical with protective cover
    if clean_nutrient == "Moderate_Chemical" and clean_cover in ("Mulch", "Cover_Crops", "Minimum_Tillage"):
        recs.append({
            "priority": "Medium",
            "category": "Resource Conservation",
            "title": "Adopt Organic or Integrated Nutrient Management (INM)",
            "action": f"Incorporate organic amendments (compost, bio-fertilizers) alongside split synthetic applications, while maintaining {cover_display}.",
            "rationale": (
                f"Protective soil cover ({cover_display}) provides erosion control; introducing organic amendments "
                f"supports long-term soil microbial activity and nutrient retention."
            ),
            "why": (
                f"Protective soil cover ({cover_display}) provides erosion control; introducing organic amendments "
                f"supports long-term soil microbial activity and nutrient retention."
            ),
        })
        return recs

    # 6. Moderate Chemical with Bare Soil
    if clean_nutrient == "Moderate_Chemical" and clean_cover in ("Bare_Soil", "Bare", "None", ""):
        recs.append({
            "priority": "Medium",
            "category": "Resource Conservation",
            "title": "Adopt Organic or Integrated Nutrient Management & Soil Cover",
            "action": "Incorporate organic amendments alongside chemical fertilizers and establish protective soil cover (mulching or cover crops).",
            "rationale": "Moderate synthetic chemical fertilization on bare soil increases exposure to nutrient runoff and surface compaction.",
            "why": "Moderate synthetic chemical fertilization on bare soil increases exposure to nutrient runoff and surface compaction.",
        })
        return recs

    # 7. Intensive Chemical / Synthetic Heavy with Bare Soil
    if clean_nutrient in ("Intensive_Chemical", "Synthetic_Heavy") and clean_cover in ("Bare_Soil", "Bare", "None", ""):
        recs.append({
            "priority": "High",
            "category": "Resource Conservation",
            "title": "Adopt Organic or Integrated Nutrient Management (INM)",
            "action": "Incorporate well-rotted farmyard manure, compost, or bio-fertilizers alongside targeted mineral fertilizers.",
            "rationale": "Intensive synthetic chemical fertilization carries elevated risks of soil acidification and nutrient runoff.",
            "why": "Intensive synthetic chemical fertilization carries elevated risks of soil acidification and nutrient runoff.",
        })
        recs.append({
            "priority": "High",
            "category": "Resource Conservation",
            "title": "Establish Protective Soil Cover & Mulch",
            "action": "Apply organic residue mulching (crop straw or leaf litter) or sow leguminous cover crops.",
            "rationale": "Bare soil accelerates surface runoff, moisture loss, and weed competition.",
            "why": "Bare soil accelerates surface runoff, moisture loss, and weed competition.",
        })
        return recs

    # 8. Intensive Chemical / Synthetic Heavy with protective cover
    if clean_nutrient in ("Intensive_Chemical", "Synthetic_Heavy"):
        recs.append({
            "priority": "High",
            "category": "Resource Conservation",
            "title": "Adopt Organic or Integrated Nutrient Management (INM)",
            "action": f"Incorporate organic amendments (compost, green manure) to balance intensive synthetic inputs while maintaining {cover_display}.",
            "rationale": f"Although {cover_display} helps reduce erosion, intensive synthetic inputs risk soil microbial degradation over time.",
            "why": f"Although {cover_display} helps reduce erosion, intensive synthetic inputs risk soil microbial degradation over time.",
        })
        return recs

    # Fallback
    recs.append({
        "priority": "Low",
        "category": "Resource Conservation",
        "title": "Maintain Balanced Soil & Nutrient Practices",
        "action": "Continue annual soil monitoring and balanced nutrient additions.",
        "rationale": "Sustains soil structure and balanced nutrient management.",
        "why": "Sustains soil structure and balanced nutrient management.",
    })
    return recs


def _build_crop_health_recommendation(
    crop_health_status: str | None,
    is_assessed: bool,
    health_score: int | float | None,
) -> dict[str, str]:
    """
    Generate input-sensitive crop health recommendations based on
    reported foliar observation or image-based screening.
    """
    clean_status = (crop_health_status or "Not_Assessed").strip()

    if not is_assessed or clean_status in ("Not_Assessed", "none", "None", ""):
        return {
            "priority": "Medium",
            "category": "Crop Health",
            "title": "Screen Foliar Health for Complete Scoring",
            "action": "No crop-health observation was provided. Consider inspecting representative plants or using the Disease Detection module.",
            "rationale": "Crop health was not assessed during this evaluation. Assessing foliar condition enables the full 3-component sustainability evaluation.",
            "why": "Crop health was not assessed during this evaluation. Assessing foliar condition enables the full 3-component sustainability evaluation.",
        }

    if clean_status == "Healthy":
        return {
            "priority": "Low",
            "category": "Crop Health",
            "title": "Continue Regular Crop-Health Monitoring",
            "action": "Maintain periodic scouting of crop foliage and monitor representative plants across the plot.",
            "rationale": "Field foliage is currently reported as Healthy. Routine scouting supports early detection before issues can spread.",
            "why": "Field foliage is currently reported as Healthy. Routine scouting supports early detection before issues can spread.",
        }

    if clean_status in ("Minor_Stress", "Mild_Stress"):
        return {
            "priority": "Medium",
            "category": "Crop Health",
            "title": "Inspect Field for Physiological or Moisture Stress",
            "action": "Inspect representative plants showing mild symptoms or discoloration to determine whether stress is physiological, nutritional, or moisture-related.",
            "rationale": "Minor crop stress was observed. Early field inspection can resolve symptoms before they develop into yield-impacting problems.",
            "why": "Minor crop stress was observed. Early field inspection can resolve symptoms before they develop into yield-impacting problems.",
        }

    if clean_status in ("Disease_Detected", "Moderate_Disease"):
        return {
            "priority": "High",
            "category": "Crop Health",
            "title": "Targeted Foliar Disease Management & Screening",
            "action": "Perform targeted symptom screening using AgriSmart AI Crop Disease Detection and consult local agronomic guidance for approved cultural or therapeutic interventions.",
            "rationale": "Active foliar disease symptoms were reported. Timely diagnosis and localized management can help contain disease spread across the canopy.",
            "why": "Active foliar disease symptoms were reported. Timely diagnosis and localized management can help contain disease spread across the canopy.",
        }

    if clean_status in ("Severe_Damage", "Severe_Disease"):
        return {
            "priority": "Critical",
            "category": "Crop Health",
            "title": "Targeted Foliar Disease Management & Urgent Field Inspection",
            "action": "Conduct an immediate comprehensive field inspection and consult a qualified local plant pathologist or agricultural extension officer.",
            "rationale": "Severe foliar damage was reported, representing an acute risk to crop health and productivity that warrants professional on-site evaluation.",
            "why": "Severe foliar damage was reported, representing an acute risk to crop health and productivity that warrants professional on-site evaluation.",
        }

    # Fallback if score-based
    if health_score is not None and health_score < 70:
        return {
            "priority": "High",
            "category": "Crop Health",
            "title": "Targeted Foliar Disease Management & Screening",
            "action": "Perform symptom screening and consult local agronomic guidance.",
            "rationale": "Sub-optimal crop health score warrants targeted foliar inspection.",
            "why": "Sub-optimal crop health score warrants targeted foliar inspection.",
        }

    return {
        "priority": "Low",
        "category": "Crop Health",
        "title": "Continue Regular Crop-Health Monitoring",
        "action": "Continue regular crop-health monitoring and periodic scouting.",
        "rationale": "Crop foliage is currently in satisfactory condition.",
        "why": "Crop foliage is currently in satisfactory condition.",
    }


def generate_sustainability_recommendations(
    water_eval: dict[str, Any],
    resource_eval: dict[str, Any],
    health_eval: dict[str, Any],
    field_inputs: dict[str, Any] | None = None,
    crop_requirement: CropRequirement | None = None,
    weather_summary: dict[str, Any] | None = None,
    weather_status: str = "UNAVAILABLE",
) -> list[dict[str, Any]]:
    """
    Generate prioritized, input-sensitive agronomic recommendations.
    Deterministically evaluates conditions across water stewardship, resource conservation,
    and crop health without artificial padding.

    Returns 1 to 4 actionable recommendations sorted by project-defined priority.
    """
    # Extract inputs from field_inputs or fallback to component evaluation dicts
    water_inputs = water_eval.get("inputs", {})
    resource_inputs = resource_eval.get("inputs", {})
    health_inputs = health_eval.get("inputs", {})

    inputs = field_inputs or {}

    soil_moisture = float(
        inputs.get("soil_moisture_pct")
        if inputs.get("soil_moisture_pct") is not None
        else water_inputs.get("soil_moisture_pct", 50.0)
    )

    irrigation_method = str(
        inputs.get("irrigation_method")
        or water_inputs.get("irrigation_method", "Flood")
    )

    water_availability = str(
        inputs.get("water_availability")
        or water_inputs.get("water_availability", "Moderate")
    )

    growth_stage = str(
        inputs.get("growth_stage")
        or water_inputs.get("growth_stage", "Vegetative")
    )

    nutrient_practice = str(
        inputs.get("nutrient_practice")
        or resource_inputs.get("nutrient_practice", "Integrated")
    )

    soil_cover = str(
        inputs.get("soil_cover")
        or resource_inputs.get("soil_cover", "Bare_Soil")
    )

    is_health_assessed = bool(health_eval.get("is_assessed", False))
    health_score = health_eval.get("score")

    raw_health_status = inputs.get("crop_health_status") or health_inputs.get("crop_health_status")
    if not raw_health_status and is_health_assessed:
        if health_score is not None:
            if health_score >= 85:
                crop_health_status = "Healthy"
            elif health_score >= 70:
                crop_health_status = "Minor_Stress"
            elif health_score >= 40:
                crop_health_status = "Disease_Detected"
            else:
                crop_health_status = "Severe_Damage"
        else:
            crop_health_status = "Healthy"
    elif not raw_health_status:
        crop_health_status = "Not_Assessed"
    else:
        crop_health_status = str(raw_health_status)

    # 1. Water Recommendation
    water_rec = _build_water_recommendation(
        soil_moisture=soil_moisture,
        irrigation_method=irrigation_method,
        water_availability=water_availability,
        weather_summary=weather_summary,
        weather_status=weather_status,
        crop_requirement=crop_requirement,
        growth_stage=growth_stage,
    )

    # 2. Resource Recommendations
    resource_recs = _build_resource_recommendations(
        nutrient_practice=nutrient_practice,
        soil_cover=soil_cover,
    )

    # 3. Crop Health Recommendation
    health_rec = _build_crop_health_recommendation(
        crop_health_status=crop_health_status,
        is_assessed=is_health_assessed,
        health_score=health_score,
    )

    # Assemble all recommendations
    all_recs: list[dict[str, Any]] = [water_rec] + resource_recs + [health_rec]

    # Sort deterministically by AgriSmart project-defined priority
    all_recs.sort(key=lambda r: PRIORITY_WEIGHTS.get(r.get("priority", "Low"), 3))

    # Cap at 4 actionable recommendations (1 to 4 returned)
    return all_recs[:4]
