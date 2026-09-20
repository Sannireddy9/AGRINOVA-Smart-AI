"""
AgriSmart AI — Farmer Assistant Service Layer
=============================================
Orchestrates conversational decision support, coordinates live module
context retrieval, and manages compact session memory.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.disease_service import DiseaseDetectionService
from app.services.crop_service import CropRecommendationService
from app.services.weather_service import WeatherIntelligenceService
from app.services.irrigation_service import SmartIrrigationService
from app.services.sustainability_service import SustainabilityService
from model.farmer_assistant.assistant import FarmerAssistant, get_mode_notice
from model.farmer_assistant.config import FarmerAssistantConfig

logger = logging.getLogger(__name__)


class FarmerAssistantService:
    """Service boundary for conversational agricultural assistant."""

    def __init__(
        self,
        assistant: FarmerAssistant | None = None,
        weather_service: WeatherIntelligenceService | None = None,
        irrigation_service: SmartIrrigationService | None = None,
        sustainability_service: SustainabilityService | None = None,
        crop_service: CropRecommendationService | None = None,
        disease_service: DiseaseDetectionService | None = None,
    ) -> None:
        self.assistant = assistant or FarmerAssistant()
        self.weather_service = weather_service or WeatherIntelligenceService()
        self.irrigation_service = irrigation_service or SmartIrrigationService(weather_service=self.weather_service)
        self.sustainability_service = sustainability_service or SustainabilityService(weather_service=self.weather_service)
        self.crop_service = crop_service or CropRecommendationService()
        self.disease_service = disease_service or DiseaseDetectionService()

    def get_context_summary(self, session_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate an authoritative, concise current farm context summary for UI presentation."""
        ctx = session_context or {}
        moisture = ctx.get("soil_moisture")
        crop = ctx.get("crop")
        growth_stage = ctx.get("growth_stage")
        location = ctx.get("location")
        irrigation_method = ctx.get("irrigation_method")
        water_availability = ctx.get("water_availability")
        nutrient_practice = ctx.get("nutrient_practice")
        soil_cover = ctx.get("soil_cover")

        # Sustainability score: check session first; if not present but moisture is, compute on-the-fly
        sust_score = ctx.get("sustainability_score")
        if sust_score is None and moisture is not None:
            try:
                sust_res = self.sustainability_service.calculate_score(
                    soil_moisture=float(moisture),
                    irrigation_method=str(irrigation_method or "Drip"),
                    water_availability=str(water_availability or "Moderate"),
                    nutrient_practice=str(nutrient_practice or "Integrated"),
                    soil_cover=str(soil_cover or "Mulch"),
                    crop=str(crop or "Tomato"),
                    growth_stage=str(growth_stage or "Vegetative"),
                    location=str(location) if location else None,
                )
                if sust_res and sust_res.get("score") is not None:
                    sust_score = sust_res.get("score")
            except Exception as exc:
                logger.debug("On-the-fly sustainability evaluation in get_context_summary: %s", exc)
                sust_score = None

        # Weather status: LIVE, DEMO, or UNAVAILABLE
        weather_status = "UNAVAILABLE"
        if location and str(location).strip():
            try:
                w_res = self.weather_service.analyze(location=str(location).strip(), force_refresh=False)
                if w_res:
                    weather_status = "DEMO" if w_res.get("is_mock") else "LIVE"
            except Exception:
                weather_status = "UNAVAILABLE"
        else:
            weather_status = "UNAVAILABLE"

        disease_status = "LIVE" if self.disease_service.is_checkpoint_available() else "DEMO"
        crop_model_status = "LIVE" if self.crop_service.is_model_available() else "DEMO"
        is_local = isinstance(getattr(self.assistant, "provider", None), type(getattr(self.assistant, "local_provider", None)))

        return {
            "crop": str(crop).strip() if crop and str(crop).strip() else "Not provided",
            "growth_stage": str(growth_stage).strip() if growth_stage and str(growth_stage).strip() else "Not provided",
            "soil_moisture": f"{float(moisture):.0f}%" if moisture is not None else "Not provided",
            "location": str(location).strip() if location and str(location).strip() else "Not provided",
            "irrigation_method": str(irrigation_method).strip() if irrigation_method and str(irrigation_method).strip() else "Not provided",
            "water_availability": str(water_availability).strip() if water_availability and str(water_availability).strip() else "Not provided",
            "nutrient_practice": str(nutrient_practice).strip() if nutrient_practice and str(nutrient_practice).strip() else "Not provided",
            "soil_cover": str(soil_cover).strip() if soil_cover and str(soil_cover).strip() else "Not provided",
            "sustainability_score": f"{round(float(sust_score))}/100" if sust_score is not None else "Not provided",
            "weather_status": weather_status,
            "disease_status": disease_status,
            "crop_model_status": crop_model_status,
            "disease_prediction": ctx.get("disease_prediction"),
            "recommended_crop": ctx.get("recommended_crop"),
            "mode_notice": get_mode_notice(getattr(self.assistant, "provider", None), is_local),
        }

    def answer_farmer_query(
        self,
        query: str,
        language: str = "en",
        session_context: dict[str, Any] | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Synthesize available context and answer farmer question.
        Returns clean structured response.
        """
        # 1. Merge compact session context with optional request context
        farm_state = dict(session_context or {})
        if request_context:
            for k in (
                "crop", "growth_stage", "location", "soil_moisture",
                "irrigation_method", "water_availability", "nutrient_practice",
                "soil_cover", "sustainability_score", "disease_prediction", "recommended_crop"
            ):
                if request_context.get(k) is not None:
                    farm_state[k] = request_context[k]

        # 2. Retrieve optional module contexts based on compact profile
        weather_data = None
        loc = farm_state.get("location")
        if loc and str(loc).strip():
            try:
                weather_data = self.weather_service.analyze(location=str(loc).strip(), force_refresh=False)
            except Exception as exc:
                logger.warning("Assistant weather retrieval failed: %s", exc)
                weather_data = None

        irrigation_data = None
        moisture_val = farm_state.get("soil_moisture")
        if moisture_val is not None:
            try:
                irrigation_data = self.irrigation_service.get_advisory(
                    location=str(loc or "Ahmedabad, Gujarat"),
                    crop=farm_state.get("crop", "Tomato"),
                    growth_stage=farm_state.get("growth_stage", "Vegetative"),
                    soil_moisture=float(moisture_val),
                    irrigation_method=farm_state.get("irrigation_method", "Drip"),
                )
            except Exception as exc:
                logger.warning("Assistant irrigation evaluation failed: %s", exc)
                irrigation_data = None

        sustainability_data = None
        if farm_state.get("sustainability_score") is not None:
            sustainability_data = {
                "score": farm_state.get("sustainability_score"),
                "category": "Assessed in Sustainability Module",
                "components": {},
                "recommendations": [],
            }
        elif moisture_val is not None:
            try:
                sustainability_data = self.sustainability_service.calculate_score(
                    soil_moisture=float(moisture_val),
                    irrigation_method=farm_state.get("irrigation_method", "Drip"),
                    water_availability=farm_state.get("water_availability", "Moderate"),
                    nutrient_practice=farm_state.get("nutrient_practice", "Integrated"),
                    soil_cover=farm_state.get("soil_cover", "Mulch"),
                    crop=farm_state.get("crop", "Tomato"),
                    growth_stage=farm_state.get("growth_stage", "Vegetative"),
                    location=str(loc) if loc else None,
                )
            except Exception as exc:
                logger.warning("Assistant sustainability evaluation failed: %s", exc)
                sustainability_data = None

        disease_data = None
        if request_context and request_context.get("disease_prediction"):
            disease_data = {
                "prediction": request_context.get("disease_prediction"),
                "is_mock": not self.disease_service.is_checkpoint_available(),
                "treatment": request_context.get("disease_treatment"),
                "prevention": request_context.get("disease_prevention"),
            }
        elif farm_state.get("disease_prediction"):
            disease_data = {
                "prediction": farm_state.get("disease_prediction"),
                "is_mock": not self.disease_service.is_checkpoint_available(),
            }

        crop_rec_data = None
        if request_context and request_context.get("recommended_crop"):
            crop_rec_data = {
                "recommended_crop": request_context.get("recommended_crop"),
                "confidence": request_context.get("crop_confidence"),
            }
        elif farm_state.get("recommended_crop"):
            crop_rec_data = {
                "recommended_crop": farm_state.get("recommended_crop"),
            }

        # 3. Execute query with assistant
        result = self.assistant.ask(
            query=query,
            language=language,
            farm_state=farm_state,
            weather_data=weather_data,
            irrigation_data=irrigation_data,
            sustainability_data=sustainability_data,
            disease_data=disease_data,
            crop_recommendation_data=crop_rec_data,
        )

        return result
