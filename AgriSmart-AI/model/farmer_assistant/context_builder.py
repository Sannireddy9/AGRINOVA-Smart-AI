"""
AgriSmart AI — Farmer Assistant Context Builder & Intent Classifier
==================================================================
Builds strictly grounded context snapshots from currently available
AgriSmart module results and user inputs. Never invents missing data.
"""

from __future__ import annotations

import re
from typing import Any


def classify_intent(query: str) -> str:
    """
    Deterministic query intent classification.
    Categorizes the farmer's question into one of the 6 core intents:
    IRRIGATION, WEATHER, SUSTAINABILITY, DISEASE, CROP, or GENERAL.
    Handles compound multi-module queries by classifying them as GENERAL to synthesize all active modules.
    Uses regex word boundaries to prevent false substring collisions (e.g. 'rog' in 'nitrogen').
    """
    q = str(query or "").lower().strip()

    # 1. Sustainability (Specific phrases checked before generic 'water')
    sustainability_patterns = [
        r"\bsustainability\b",
        r"\bsustainable\b",
        r"\bsustainability\s+score\b",
        r"\besg\b",
        r"\bwater\s+efficiency\b",
        r"\bresource\s+use\b",
        r"\bfoliar\s+health\b",
        r"\binm\b",
        r"\bsoil\s+cover\b",
        r"\borganic\s+practice\b",
        r"\bsustainability\s+(?:ka|no|nu)\s+score\b",
    ]

    # 2. Crop Recommendation
    crop_patterns = [
        r"\brecommend\b.*?\bcrop\b",
        r"\bcrop\b.*?\brecommend",
        r"\bwhat\s+crop\b",
        r"\bwhich\s+crop\b",
        r"\bwhich\s+crop\s+should\s+i\s+grow\b",
        r"\bwhat\s+to\s+grow\b",
        r"\bwhich\s+to\s+grow\b",
        r"\bwhat\s+to\s+plant\b",
        r"\bsow\b",
        r"\bsowing\b",
        r"\bsoil\s+suitability\b",
        r"\bfertilizer\s+need\b",
        r"\bnpk\b",
        r"\bkaun\s+si\s+fasal\b",
        r"\bfasal\s+ki\s+sifarish\b",
        r"\bkai\s+fasal\b",
        r"\bkai\s+pak\b",
        r"\bkaun\s+sa\s+pak\b",
    ]

    # 3. Disease Intent
    disease_patterns = [
        r"\bdisease\b",
        r"\bdiseased\b",
        r"\bblight\b",
        r"\bspot\b",
        r"\binfection\b",
        r"\bfungus\b",
        r"\bfungal\b",
        r"\blesion\b",
        r"\bmildew\b",
        r"\byellow\s+leaf\b",
        r"\bbrown\s+spot\b",
        r"\brot\b",
        r"\bleaf\s+symptom\b",
        r"\bcrop\s+health\b",
        r"\bsick\s+plant\b",
        r"\brog\b",
        r"\bbimari\b",
    ]

    # 4. Irrigation Intent
    irrigation_patterns = [
        r"\birrigate\b",
        r"\birrigation\b",
        r"\bwatering\b",
        r"\bwater\s+my\b",
        r"\bwater\s+the\b",
        r"\bwater\s+to\b",
        r"\bshould\s+i\s+water\b",
        r"\bwhen\s+to\s+water\b",
        r"\bneed\s+to\s+water\b",
        r"\bmoisture\b",
        r"\bsoil\s+moisture\b",
        r"\bdrip\b",
        r"\bsprinkler\b",
        r"\bflood\b",
        r"\bdry\s+soil\b",
        r"\bsoil\s+dry\b",
        r"\bwet\s+soil\b",
        r"\bpaani\b",
        r"\bpani\b",
        r"\bsinchai\b",
        r"\bpiyat\b",
    ]

    # 5. Weather Intent
    weather_patterns = [
        r"\bweather\b",
        r"\brain\b",
        r"\brainfall\b",
        r"\btemperature\b",
        r"\bforecast\b",
        r"\bhumidity\b",
        r"\bprecipitation\b",
        r"\bcloud\b",
        r"\bmonsoon\b",
        r"\bwind\b",
        r"\bmausam\b",
        r"\bbarish\b",
        r"\bhavaman\b",
        r"\bvarsad\b",
        r"\bvarsaad\b",
    ]

    matches = {
        "SUSTAINABILITY": any(re.search(pat, q) for pat in sustainability_patterns),
        "CROP": any(re.search(pat, q) for pat in crop_patterns),
        "DISEASE": any(re.search(pat, q) for pat in disease_patterns),
        "IRRIGATION": any(re.search(pat, q) for pat in irrigation_patterns),
        "WEATHER": any(re.search(pat, q) for pat in weather_patterns),
    }

    matched_intents = [intent for intent, matched in matches.items() if matched]

    # If multiple distinct module categories are matched in the same query, treat as compound/integrated GENERAL query
    if len(matched_intents) >= 2:
        return "GENERAL"
    elif len(matched_intents) == 1:
        return matched_intents[0]

    return "GENERAL"


def extract_soil_moisture_from_text(query: str) -> float | None:
    """
    Extract stated soil moisture percentage from conversational input.
    E.g., "My soil moisture is 25%" -> 25.0
    """
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", query)
    if match:
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 100.0:
                return val
        except ValueError:
            pass

    match2 = re.search(r"moisture(?:\s+is|\s*:\s*|\s+level\s+is)?\s+(\d{1,3}(?:\.\d+)?)", query, re.IGNORECASE)
    if match2:
        try:
            val = float(match2.group(1))
            if 0.0 <= val <= 100.0:
                return val
        except ValueError:
            pass

    return None


def extract_farm_profile_from_text(query: str) -> dict[str, Any]:
    """
    Extract field observations stated conversationally by the farmer.
    E.g.: 'I am growing tomato at the flowering stage. My soil moisture is 28%, my farm is located in Ahmedabad, Gujarat, and I use drip irrigation.'
    Returns dict with extracted scalar fields: crop, growth_stage, soil_moisture, location, irrigation_method.
    """
    extracted: dict[str, Any] = {}
    q = str(query or "").strip()
    q_lower = q.lower()

    # 1. Soil Moisture
    moisture = extract_soil_moisture_from_text(q)
    if moisture is not None:
        extracted["soil_moisture"] = moisture

    # 2. Cultivated Crop
    known_crops = [
        "tomato", "wheat", "rice", "maize", "cotton", "potato", "sugarcane",
        "chickpea", "mustard", "onion", "soybean", "groundnut", "banana",
        "mango", "grapes", "apple", "orange", "papaya", "pomegranate",
        "watermelon", "muskmelon", "coffee", "jute", "lentil", "blackgram",
        "pigeonpeas", "mungbean", "mothbeans"
    ]
    for c in known_crops:
        if re.search(r"\b" + re.escape(c) + r"\b", q_lower):
            extracted["crop"] = c.capitalize()
            break

    # 3. Growth Stage
    known_stages = {
        "flowering": "Flowering",
        "vegetative": "Vegetative",
        "fruiting": "Fruiting",
        "seedling": "Seedling",
        "germination": "Germination",
        "maturity": "Maturity",
        "ripening": "Ripening",
        "harvest": "Harvest",
        "harvesting": "Harvest",
    }
    for st_key, st_name in known_stages.items():
        if re.search(r"\b" + re.escape(st_key) + r"\b", q_lower):
            extracted["growth_stage"] = st_name
            break

    # 4. Irrigation System
    known_irrigation = {
        "drip": "Drip",
        "sprinkler": "Sprinkler",
        "flood": "Flood",
        "furrow": "Furrow",
        "rainfed": "Rainfed",
        "subsurface": "Drip",
    }
    for ir_key, ir_name in known_irrigation.items():
        if re.search(r"\b" + re.escape(ir_key) + r"\b", q_lower):
            extracted["irrigation_method"] = ir_name
            break

    # 5. Farm Location
    loc_match = re.search(
        r"(?:located\s+in|farm\s+(?:is\s+)?in|location\s+(?:is|:)?)\s+([A-Za-z\s,]+?)(?:\.|\sand\s|\susing|,?\s*and\s|,?\s*my|\n|$)",
        q,
        re.IGNORECASE,
    )
    if loc_match:
        cand_loc = loc_match.group(1).strip().strip(",").strip()
        if cand_loc and len(cand_loc) >= 3 and not any(cand_loc.lower().startswith(w) for w in ["drip", "flowering", "tomato", "the", "a"]):
            extracted["location"] = cand_loc

    return extracted


def build_farm_context(
    farm_state: dict[str, Any] | None = None,
    weather_data: dict[str, Any] | None = None,
    irrigation_data: dict[str, Any] | None = None,
    sustainability_data: dict[str, Any] | None = None,
    disease_data: dict[str, Any] | None = None,
    crop_recommendation_data: dict[str, Any] | None = None,
    intent: str = "GENERAL",
) -> dict[str, Any]:
    """
    Synthesize all currently available AgriSmart context.
    Strictly omits missing data and tags operational states accurately.
    """
    farm = farm_state or {}
    context: dict[str, Any] = {
        "intent": intent,
        "farm_profile": {},
        "modules": {},
        "sources_used": [],
    }

    # 1. Farm Profile (Farmer-entered inputs)
    profile: dict[str, Any] = {}
    if farm.get("crop"):
        profile["crop"] = str(farm["crop"])
    if farm.get("growth_stage"):
        profile["growth_stage"] = str(farm["growth_stage"])
    if farm.get("location"):
        profile["location"] = str(farm["location"])
    if farm.get("soil_moisture") is not None:
        try:
            profile["soil_moisture_pct"] = float(farm["soil_moisture"])
        except (ValueError, TypeError):
            pass
    if farm.get("irrigation_method"):
        profile["irrigation_method"] = str(farm["irrigation_method"])
    if farm.get("water_availability"):
        profile["water_availability"] = str(farm["water_availability"])
    if farm.get("nutrient_practice"):
        profile["nutrient_practice"] = str(farm["nutrient_practice"])
    if farm.get("soil_cover"):
        profile["soil_cover"] = str(farm["soil_cover"])

    context["farm_profile"] = profile

    # 2. Weather Intelligence Context
    if weather_data:
        w_status = "DEMO" if weather_data.get("is_mock") else "LIVE"
        cur = weather_data.get("current_weather") or {}
        temp = cur.get("temperature") if cur.get("temperature") is not None else weather_data.get("temperature_c")
        hum = cur.get("relative_humidity") if cur.get("relative_humidity") is not None else weather_data.get("relative_humidity_pct")
        desc = cur.get("condition") if cur.get("condition") is not None else weather_data.get("weather_description")

        max_prob = float(weather_data.get("precipitation_probability_max", 0.0))
        expected_mm = float(weather_data.get("precipitation_sum", 0.0))

        # Check forecast array if available
        forecast = weather_data.get("forecast", [])
        if forecast and (max_prob == 0.0 or expected_mm == 0.0):
            near_term = forecast[:3]
            max_prob = max([float(d.get("precipitation_probability_max", 0.0)) for d in near_term] or [0.0])
            expected_mm = sum([float(d.get("precipitation_sum", 0.0)) for d in near_term])

        loc = weather_data.get("resolved_location") or weather_data.get("location") or profile.get("location", "Unknown")

        context["modules"]["weather"] = {
            "status": w_status,
            "location": loc,
            "temperature_c": temp,
            "relative_humidity_pct": hum,
            "weather_description": desc,
            "max_rain_probability_pct": max_prob,
            "expected_rain_mm": expected_mm,
            "is_significant_rain": bool(expected_mm >= 4.0 and max_prob >= 40.0),
            "is_negligible_rain": bool(expected_mm < 2.0 and max_prob >= 50.0),
        }
    else:
        context["modules"]["weather"] = {"status": "UNAVAILABLE"}

    # 3. Smart Irrigation Context
    if irrigation_data and irrigation_data.get("decision"):
        context["modules"]["irrigation"] = {
            "status": "LIVE" if irrigation_data.get("is_live", True) else "DEMO",
            "decision": irrigation_data.get("decision"),
            "priority": irrigation_data.get("priority"),
            "headline": irrigation_data.get("headline"),
            "primary_action": irrigation_data.get("primary_action"),
            "reasoning": irrigation_data.get("reasoning", []),
        }
    else:
        context["modules"]["irrigation"] = {"status": "UNAVAILABLE"}

    # 4. Sustainability Score Context
    if sustainability_data and sustainability_data.get("score") is not None:
        comps = sustainability_data.get("components", {})
        context["modules"]["sustainability"] = {
            "status": "ACTIVE",
            "score": sustainability_data.get("score"),
            "category": sustainability_data.get("category"),
            "water_score": comps.get("water_efficiency", {}).get("score"),
            "resource_score": comps.get("resource_use", {}).get("score"),
            "crop_health_score": comps.get("crop_health", {}).get("score"),
            "recommendations": [
                {
                    "title": r.get("title"),
                    "action": r.get("action"),
                    "why": r.get("why") or r.get("rationale"),
                    "priority": r.get("priority"),
                }
                for r in sustainability_data.get("recommendations", [])[:3]
            ],
        }
    else:
        context["modules"]["sustainability"] = {"status": "UNAVAILABLE"}

    # 5. Crop Disease AI Context
    if disease_data and (disease_data.get("prediction") or disease_data.get("disease_name")):
        pred_name = disease_data.get("prediction") or disease_data.get("disease_name")
        is_mock = bool(disease_data.get("is_mock", False))
        context["modules"]["disease"] = {
            "status": "DEMO" if is_mock else "LIVE",
            "prediction": pred_name,
            "confidence": disease_data.get("confidence") if not is_mock else None,
            "treatment": disease_data.get("treatment"),
            "prevention": disease_data.get("prevention"),
        }
    else:
        context["modules"]["disease"] = {"status": "UNAVAILABLE"}

    # 6. Crop Recommendation Context
    if crop_recommendation_data and (crop_recommendation_data.get("recommended_crop") or crop_recommendation_data.get("crop")):
        rec_crop = crop_recommendation_data.get("recommended_crop") or crop_recommendation_data.get("crop")
        context["modules"]["crop_recommendation"] = {
            "status": "LIVE",
            "recommended_crop": rec_crop,
            "confidence": crop_recommendation_data.get("confidence"),
        }
    else:
        context["modules"]["crop_recommendation"] = {"status": "UNAVAILABLE"}

    # 7. Identify Active Sources for the specific intent
    sources = []
    if intent in ("IRRIGATION", "GENERAL"):
        if context["modules"]["irrigation"]["status"] != "UNAVAILABLE":
            sources.append("Smart Irrigation")
        if context["modules"]["weather"]["status"] != "UNAVAILABLE":
            sources.append("Weather Intelligence")
    if intent in ("WEATHER", "GENERAL"):
        if context["modules"]["weather"]["status"] != "UNAVAILABLE" and "Weather Intelligence" not in sources:
            sources.append("Weather Intelligence")
    if intent in ("SUSTAINABILITY", "GENERAL"):
        if context["modules"]["sustainability"]["status"] != "UNAVAILABLE":
            sources.append("Sustainability Score")
    if intent in ("DISEASE", "GENERAL"):
        if context["modules"]["disease"]["status"] != "UNAVAILABLE":
            sources.append("Crop Disease AI")
    if intent in ("CROP", "GENERAL"):
        if context["modules"]["crop_recommendation"]["status"] != "UNAVAILABLE":
            sources.append("Crop Recommendation Model")

    context["sources_used"] = sources
    return context


def format_context_prompt(context: dict[str, Any]) -> str:
    """
    Format structured context into an explainable, grounded text block
    prioritized by the classified intent.
    """
    intent = context.get("intent", "GENERAL")
    profile = context.get("farm_profile", {})
    mods = context.get("modules", {})

    lines: list[str] = [f"=== ACTIVE QUERY INTENT: {intent} ==="]

    # Farm Profile
    lines.append("--- REPORTED FARM PROFILE ---")
    if profile:
        for k, v in profile.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (No field practices reported yet)")

    # Relevant Modules based on Intent
    lines.append("--- MODULE EVALUATIONS ---")

    # Smart Irrigation
    irrig = mods.get("irrigation", {})
    if irrig.get("status") != "UNAVAILABLE":
        lines.append(f"[Smart Irrigation - {irrig.get('status')}]:")
        lines.append(f"  Decision: {irrig.get('decision')}")
        lines.append(f"  Priority: {irrig.get('priority')}")
        lines.append(f"  Headline: {irrig.get('headline')}")
        lines.append(f"  Action: {irrig.get('primary_action')}")
    else:
        lines.append("[Smart Irrigation]: UNAVAILABLE (No irrigation assessment conducted)")

    # Weather
    w = mods.get("weather", {})
    if w.get("status") != "UNAVAILABLE":
        lines.append(f"[Weather Intelligence - {w.get('status')}]:")
        lines.append(f"  Location: {w.get('location')}")
        lines.append(f"  Temperature: {w.get('temperature_c')}°C, Humidity: {w.get('relative_humidity_pct')}%")
        lines.append(f"  Max Rain Prob: {w.get('max_rain_probability_pct')}%, Expected Rain: {w.get('expected_rain_mm')} mm")
        if w.get("is_significant_rain"):
            lines.append("  Rain Significance: Meaningful rainfall opportunity forecast")
        elif w.get("is_negligible_rain"):
            lines.append("  Rain Significance: High probability but negligible accumulation (<2mm)")
    else:
        lines.append("[Weather Intelligence]: UNAVAILABLE (No weather query performed)")

    # Sustainability
    sust = mods.get("sustainability", {})
    if sust.get("status") != "UNAVAILABLE":
        lines.append(f"[Sustainability Score - {sust.get('status')}]:")
        lines.append(f"  Overall Score: {sust.get('score')}/100 ({sust.get('category')})")
        lines.append(f"  Water Efficiency: {sust.get('water_score')}, Resource Use: {sust.get('resource_score')}, Crop Health: {sust.get('crop_health_score')}")
        for r in sust.get("recommendations", []):
            lines.append(f"  Top Action: {r.get('title')} ({r.get('action')})")
    else:
        lines.append("[Sustainability Score]: UNAVAILABLE (No score computed)")

    # Disease
    dis = mods.get("disease", {})
    if dis.get("status") != "UNAVAILABLE":
        lines.append(f"[Crop Disease AI - {dis.get('status')}]:")
        lines.append(f"  Identified Condition: {dis.get('prediction')}")
        if dis.get("status") == "DEMO":
            lines.append("  Notice: This result is from DEMO mode and is not a confirmed field diagnosis")
    else:
        lines.append("[Crop Disease AI]: UNAVAILABLE (No image uploaded for diagnosis)")

    # Crop Recommendation
    cr = mods.get("crop_recommendation", {})
    if cr.get("status") != "UNAVAILABLE":
        lines.append(f"[Crop Recommendation Model - {cr.get('status')}]:")
        lines.append(f"  Recommended Crop: {cr.get('recommended_crop')}")
    else:
        lines.append("[Crop Recommendation Model]: UNAVAILABLE (No crop recommendation requested)")

    return "\n".join(lines)


def build_structured_openai_context(
    context: dict[str, Any],
    intent: str = "GENERAL",
) -> dict[str, Any]:
    """
    Build a clean, structured JSON context payload for the OpenAI provider.
    Applies intent-based relevance filtering: only includes fields that
    actually exist (omits empty/unassessed blocks). Never fabricates data.
    """
    profile = context.get("farm_profile", {})
    mods = context.get("modules", {})
    structured: dict[str, Any] = {}

    # 1. Farmer Inputs (always included when present)
    farmer_inputs: dict[str, Any] = {}
    if profile.get("crop"):
        farmer_inputs["crop"] = profile["crop"]
    if profile.get("growth_stage"):
        farmer_inputs["growth_stage"] = profile["growth_stage"]
    if profile.get("location"):
        farmer_inputs["location"] = profile["location"]
    if profile.get("soil_moisture_pct") is not None:
        farmer_inputs["soil_moisture_pct"] = profile["soil_moisture_pct"]
    if profile.get("irrigation_method"):
        farmer_inputs["irrigation_method"] = profile["irrigation_method"]
    if profile.get("water_availability"):
        farmer_inputs["water_availability"] = profile["water_availability"]
    if profile.get("nutrient_practice"):
        farmer_inputs["nutrient_practice"] = profile["nutrient_practice"]
    if profile.get("soil_cover"):
        farmer_inputs["soil_cover"] = profile["soil_cover"]
    if farmer_inputs:
        structured["farmer_inputs"] = farmer_inputs

    # 2. Weather (relevant for IRRIGATION, WEATHER, GENERAL)
    if intent in ("IRRIGATION", "WEATHER", "GENERAL"):
        w = mods.get("weather", {})
        if w.get("status") not in ("UNAVAILABLE", None):
            weather_block: dict[str, Any] = {"status": w["status"]}
            if w.get("location"):
                weather_block["location"] = w["location"]
            if w.get("temperature_c") is not None:
                weather_block["temperature_c"] = w["temperature_c"]
            if w.get("relative_humidity_pct") is not None:
                weather_block["relative_humidity_pct"] = w["relative_humidity_pct"]
            if w.get("weather_description"):
                weather_block["weather_description"] = w["weather_description"]
            weather_block["max_rain_probability_pct"] = w.get("max_rain_probability_pct", 0.0)
            weather_block["expected_rain_mm"] = w.get("expected_rain_mm", 0.0)
            if w.get("is_significant_rain"):
                weather_block["rain_significance"] = "significant"
            elif w.get("is_negligible_rain"):
                weather_block["rain_significance"] = "negligible"
            structured["weather"] = weather_block

    # 3. Smart Irrigation (relevant for IRRIGATION, GENERAL)
    if intent in ("IRRIGATION", "GENERAL"):
        irrig = mods.get("irrigation", {})
        if irrig.get("status") not in ("UNAVAILABLE", None):
            irrig_block: dict[str, Any] = {"status": irrig["status"]}
            if irrig.get("decision"):
                irrig_block["decision"] = irrig["decision"]
            if irrig.get("priority"):
                irrig_block["priority"] = irrig["priority"]
            if irrig.get("headline"):
                irrig_block["headline"] = irrig["headline"]
            if irrig.get("primary_action"):
                irrig_block["primary_action"] = irrig["primary_action"]
            structured["smart_irrigation"] = irrig_block

    # 4. Sustainability (relevant for SUSTAINABILITY, GENERAL)
    if intent in ("SUSTAINABILITY", "GENERAL"):
        sust = mods.get("sustainability", {})
        if sust.get("status") not in ("UNAVAILABLE", None):
            sust_block: dict[str, Any] = {"status": sust["status"]}
            if sust.get("score") is not None:
                sust_block["score"] = sust["score"]
            if sust.get("category"):
                sust_block["category"] = sust["category"]
            if sust.get("water_score") is not None:
                sust_block["water_efficiency_score"] = sust["water_score"]
            if sust.get("resource_score") is not None:
                sust_block["resource_use_score"] = sust["resource_score"]
            if sust.get("crop_health_score") is not None:
                sust_block["crop_health_score"] = sust["crop_health_score"]
            recs = sust.get("recommendations", [])
            if recs:
                sust_block["top_actions"] = [
                    {"title": r.get("title"), "action": r.get("action")}
                    for r in recs[:3] if r.get("title")
                ]
            structured["sustainability"] = sust_block

    # 5. Disease (relevant for DISEASE, GENERAL)
    if intent in ("DISEASE", "GENERAL"):
        dis = mods.get("disease", {})
        if dis.get("status") not in ("UNAVAILABLE", None):
            dis_block: dict[str, Any] = {"status": dis["status"]}
            if dis.get("prediction"):
                dis_block["prediction"] = dis["prediction"]
            if dis.get("treatment"):
                dis_block["treatment"] = dis["treatment"]
            if dis.get("prevention"):
                dis_block["prevention"] = dis["prevention"]
            structured["disease"] = dis_block

    # 6. Crop Recommendation (relevant for CROP, GENERAL)
    if intent in ("CROP", "GENERAL"):
        cr = mods.get("crop_recommendation", {})
        if cr.get("status") not in ("UNAVAILABLE", None):
            cr_block: dict[str, Any] = {"status": cr["status"]}
            if cr.get("recommended_crop"):
                cr_block["recommended_crop"] = cr["recommended_crop"]
            structured["crop_recommendation"] = cr_block

    return structured


# Universal alias for all grounded LLM providers (OpenAI, Gemini)
build_structured_context = build_structured_openai_context
