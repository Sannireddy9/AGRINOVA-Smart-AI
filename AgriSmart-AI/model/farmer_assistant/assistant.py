"""
AgriSmart AI — Farmer Assistant Coordinator
===========================================
Coordinates intent classification, grounded context synthesis,
guardrail enforcement, provider execution, and safe local fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from model.farmer_assistant.config import FarmerAssistantConfig
from model.farmer_assistant.context_builder import (
    classify_intent,
    build_farm_context,
    format_context_prompt,
    extract_soil_moisture_from_text,
    extract_farm_profile_from_text,
)
from model.farmer_assistant.guardrails import (
    check_prompt_injection,
    check_hazardous_chemicals,
    sanitize_text,
)
from model.farmer_assistant.providers import (
    BaseFarmerAssistantProvider,
    LocalRuleProvider,
    OpenAIProvider,
    GeminiProvider,
    ProviderFactory,
)

logger = logging.getLogger(__name__)


def get_mode_notice(provider: BaseFarmerAssistantProvider | None, is_local: bool) -> str:
    """Return honest, provider-specific mode notice string."""
    if is_local:
        return FarmerAssistantConfig.LOCAL_MODE_NOTICE
    if isinstance(provider, OpenAIProvider):
        return FarmerAssistantConfig.OPENAI_MODE_NOTICE
    if isinstance(provider, GeminiProvider):
        return FarmerAssistantConfig.GEMINI_MODE_NOTICE
    return FarmerAssistantConfig.GENAI_MODE_NOTICE


_get_mode_notice = get_mode_notice


class FarmerAssistant:
    """Conversational agricultural assistant grounded in AgriSmart platform context."""

    def __init__(
        self,
        provider: BaseFarmerAssistantProvider | None = None,
    ) -> None:
        self.provider = provider or ProviderFactory.get_provider()
        self.local_provider = LocalRuleProvider()

    def ask(
        self,
        query: str,
        language: str = "en",
        farm_state: dict[str, Any] | None = None,
        weather_data: dict[str, Any] | None = None,
        irrigation_data: dict[str, Any] | None = None,
        sustainability_data: dict[str, Any] | None = None,
        disease_data: dict[str, Any] | None = None,
        crop_recommendation_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a farmer's question with strict grounding and safety guardrails.
        """
        # 1. Input Validation
        clean_q = str(query or "").strip()
        if not clean_q:
            raise ValueError("Farmer query cannot be empty. Please ask a farming-related question.")

        lang = language.lower().strip()
        if lang not in FarmerAssistantConfig.SUPPORTED_LANGUAGES:
            lang = FarmerAssistantConfig.DEFAULT_LANGUAGE

        # 2. Prompt Injection Guardrail
        is_injection, injection_refusal = check_prompt_injection(clean_q)
        if is_injection:
            is_local = isinstance(self.provider, LocalRuleProvider)
            provider_name = "local" if is_local else ("openai" if isinstance(self.provider, OpenAIProvider) else "gemini")
            return {
                "answer": injection_refusal,
                "language": lang,
                "intent": "RESTRICTED",
                "sources": [],
                "data_status": "RESTRICTED",
                "provider_mode": "GUARDRAIL",
                "provider": provider_name,
                "fallback_used": False,
                "fallback_reason": None,
                "mode_notice": _get_mode_notice(self.provider, is_local),
                "disclaimer": FarmerAssistantConfig.DISCLAIMER,
                "context_summary": {},
                "diagnostics": {
                    "provider_configured": FarmerAssistantConfig.DEFAULT_PROVIDER,
                    "api_key_present": "YES" if bool(FarmerAssistantConfig.API_KEY) else "NO",
                    "selected_provider_class": self.provider.__class__.__name__,
                    "request_attempted_provider": "GUARDRAIL",
                    "fallback_triggered": False,
                    "fallback_reason_type": None,
                    "status_code": None,
                },
            }

        # 3. Hazardous Chemicals Guardrail
        is_hazard, hazard_refusal = check_hazardous_chemicals(clean_q)
        if is_hazard:
            is_local = isinstance(self.provider, LocalRuleProvider)
            provider_name = "local" if is_local else ("openai" if isinstance(self.provider, OpenAIProvider) else "gemini")
            return {
                "answer": hazard_refusal,
                "language": lang,
                "intent": "RESTRICTED",
                "sources": [],
                "data_status": "RESTRICTED",
                "provider_mode": "GUARDRAIL",
                "provider": provider_name,
                "fallback_used": False,
                "fallback_reason": None,
                "mode_notice": _get_mode_notice(self.provider, is_local),
                "disclaimer": FarmerAssistantConfig.DISCLAIMER,
                "context_summary": {},
                "diagnostics": {
                    "provider_configured": FarmerAssistantConfig.DEFAULT_PROVIDER,
                    "api_key_present": "YES" if bool(FarmerAssistantConfig.API_KEY) else "NO",
                    "selected_provider_class": self.provider.__class__.__name__,
                    "request_attempted_provider": "GUARDRAIL",
                    "fallback_triggered": False,
                    "fallback_reason_type": None,
                    "status_code": None,
                },
            }

        # 4. Extract Conversational Updates (Short-term context memory)
        active_farm = dict(farm_state or {})
        extracted_profile = extract_farm_profile_from_text(clean_q)
        for k, v in extracted_profile.items():
            active_farm[k] = v

        # 5. Intent Classification
        intent = classify_intent(clean_q)

        # 6. Build Grounded Context
        context = build_farm_context(
            farm_state=active_farm,
            weather_data=weather_data,
            irrigation_data=irrigation_data,
            sustainability_data=sustainability_data,
            disease_data=disease_data,
            crop_recommendation_data=crop_recommendation_data,
            intent=intent,
        )

        # 7. Format System Instructions and User Prompt
        system_instruction = (
            "You are the AgriSmart AI Farmer Assistant, an empathetic, explainable agricultural advisory tool. "
            "Your role is to help farmers interpret their farm conditions using strictly the provided AgriSmart context.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Answer in simple, farmer-friendly language. Avoid complex jargon without explanation.\n"
            "2. Never fabricate farm observations (soil moisture, rainfall, crop score, disease diagnosis, or water amounts).\n"
            "3. If information is unavailable or marked UNAVAILABLE, state clearly that it is not provided.\n"
            "4. If a module result is marked DEMO, explicitly inform the farmer that it is DEMO data and not a confirmed field reality.\n"
            "5. Never invent chemical dosages or off-label pesticide recipes. Advise consulting registered labels or local extension officers.\n"
            "6. Never claim certifier agronomist authority; provide decision support only.\n"
            f"7. Respond in {FarmerAssistantConfig.SUPPORTED_LANGUAGES.get(lang, 'English')}.\n"
        )

        context_prompt_text = format_context_prompt(context)
        full_user_prompt = f"{context_prompt_text}\n\nFARMER QUESTION: {clean_q}"

        # 8. Provider Execution with Safe Fallback
        initial_is_local = isinstance(self.provider, LocalRuleProvider)
        is_local = initial_is_local
        attempted_provider = self.provider.__class__.__name__
        fallback_used = False
        fallback_reason = None
        status_code = None

        provider_resp = self.provider.generate(
            system_instruction=system_instruction,
            user_prompt=full_user_prompt,
            language=lang,
            context=context,
        )

        # If external provider failed, fallback safely to LocalRuleProvider
        if not provider_resp.get("success"):
            fallback_used = not initial_is_local
            fallback_reason = provider_resp.get("error") or "External provider failure"
            status_code = provider_resp.get("status_code")
            logger.warning("External provider failed (%s), falling back to LocalRuleProvider", fallback_reason)
            provider_resp = self.local_provider.generate(
                system_instruction=system_instruction,
                user_prompt=full_user_prompt,
                language=lang,
                context=context,
            )
            is_local = True

        raw_answer = provider_resp.get("answer", "")
        clean_answer = sanitize_text(raw_answer)

        # 9. Determine Overall Data Status
        mods = context.get("modules", {})
        data_status = "LIVE"
        if any(m.get("status") == "DEMO" for m in mods.values()):
            data_status = "DEMO"
        elif all(m.get("status") == "UNAVAILABLE" for m in mods.values()):
            data_status = "UNAVAILABLE"

        # 10. Assemble Context Summary for UI (authoritative single representation)
        profile = context.get("farm_profile", {})
        w = mods.get("weather", {})
        dis = mods.get("disease", {})
        sust = mods.get("sustainability", {})

        sust_val = sust.get("score") if sust.get("score") is not None else active_farm.get("sustainability_score")
        mode_notice = _get_mode_notice(self.provider, is_local)

        context_summary = {
            "crop": profile.get("crop", "Not provided"),
            "growth_stage": profile.get("growth_stage", "Not provided"),
            "soil_moisture": f"{profile.get('soil_moisture_pct'):.0f}%" if profile.get("soil_moisture_pct") is not None else "Not provided",
            "location": profile.get("location", "Not provided"),
            "irrigation_method": profile.get("irrigation_method", "Not provided"),
            "water_availability": profile.get("water_availability", "Not provided"),
            "nutrient_practice": profile.get("nutrient_practice", "Not provided"),
            "soil_cover": profile.get("soil_cover", "Not provided"),
            "sustainability_score": f"{round(float(sust_val))}/100" if sust_val is not None else "Not provided",
            "weather_status": w.get("status", "UNAVAILABLE"),
            "disease_status": dis.get("status", "UNAVAILABLE"),
            "crop_model_status": "LIVE" if (crop_recommendation_data or {}).get("recommended_crop") else "DEMO",
            "disease_prediction": dis.get("prediction"),
            "recommended_crop": (crop_recommendation_data or {}).get("recommended_crop"),
            "mode_notice": mode_notice,
        }

        provider_name = "local" if is_local else ("openai" if isinstance(self.provider, OpenAIProvider) else "gemini")

        # Prepare safe runtime diagnostics without exposing secrets
        is_gemini_attempt = attempted_provider == "GeminiProvider"
        is_openai_attempt = attempted_provider == "OpenAIProvider"

        diag = {
            "provider_configured": FarmerAssistantConfig.DEFAULT_PROVIDER,
            "api_key_present": "YES" if bool(FarmerAssistantConfig.API_KEY) else "NO",
            "selected_provider_class": self.provider.__class__.__name__,
            "request_attempted_provider": attempted_provider,
            "fallback_triggered": fallback_used,
            "fallback_reason_type": fallback_reason,
            "fallback_provider": "LocalRuleProvider" if fallback_used else None,
            "status_code": status_code,
        }
        if is_gemini_attempt:
            diag["gemini_request_started"] = "YES"
            diag["gemini_request_result"] = "SUCCESS" if not fallback_used else "FAILED"
        elif is_openai_attempt:
            diag["openai_request_started"] = "YES"
            diag["openai_request_result"] = "SUCCESS" if not fallback_used else "FAILED"

        return {
            "answer": clean_answer,
            "language": lang,
            "intent": intent,
            "sources": context.get("sources_used", []),
            "data_status": data_status,
            "provider_mode": "LOCAL_RULE" if is_local else "GENAI",
            "provider": provider_name,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "mode_notice": mode_notice,
            "disclaimer": FarmerAssistantConfig.DISCLAIMER,
            "context_summary": context_summary,
            "updated_farm_state": active_farm,
            "diagnostics": diag,
        }
