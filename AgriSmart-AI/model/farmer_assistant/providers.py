"""
AgriSmart AI — Farmer Assistant Provider Implementations
========================================================
Implements pluggable provider abstraction:
- BaseFarmerAssistantProvider (Abstract Base)
- OpenAIProvider (OpenAI REST API)
- GeminiProvider (Google Gemini Native REST API)
- LocalRuleProvider (Deterministic Grounded Local Fallback)
- ProviderFactory (Dynamic Provider Selection)
"""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from model.farmer_assistant.config import FarmerAssistantConfig
from model.farmer_assistant.context_builder import (
    build_structured_context,
    build_structured_openai_context,
)

logger = logging.getLogger(__name__)


class BaseFarmerAssistantProvider(ABC):
    """Abstract interface for Farmer Assistant LLM providers."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or ""
        self.model = model or ""

    @abstractmethod
    def generate(
        self,
        system_instruction: str,
        user_prompt: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute completion request.
        Must return standard dict with:
        answer, why, what_to_do, sources, provider, model, success, error.
        """
        raise NotImplementedError


class OpenAIProvider(BaseFarmerAssistantProvider):
    """OpenAI API implementation using standard HTTP REST interface."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key,
            model=model or (FarmerAssistantConfig.MODEL_NAME if FarmerAssistantConfig.DEFAULT_PROVIDER == "openai" else None) or FarmerAssistantConfig.DEFAULT_MODELS["openai"],
        )
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(
        self,
        system_instruction: str,
        user_prompt: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "error": "OpenAI API key not configured.",
                "answer": "",
                "provider": "openai",
                "model": self.model,
            }

        # Build structured context for grounded responses
        enriched_prompt = user_prompt
        if context:
            intent = context.get("intent", "GENERAL")
            structured = build_structured_openai_context(context, intent)
            if structured:
                context_json = json.dumps(structured, indent=2, default=str)
                enriched_prompt = (
                    f"AGRISMART MODULE CONTEXT (verified data — do not fabricate beyond this):\n"
                    f"{context_json}\n\n"
                    f"{user_prompt}"
                )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": enriched_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=FarmerAssistantConfig.REQUEST_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "").strip()
                return {
                    "success": True,
                    "answer": content,
                    "provider": "openai",
                    "model": self.model,
                    "error": None,
                }
        except urllib.error.HTTPError as exc:
            err_type = "HTTPError"
            err_code = None
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
                err_detail = err_body.get("error", {})
                err_code = err_detail.get("code")
                err_type = err_detail.get("type") or "HTTPError"
            except Exception:
                pass

            if exc.code == 401:
                category = "AuthenticationError (HTTP 401: Invalid or revoked OpenAI API key)"
            elif exc.code == 429:
                if err_code == "credit_balance_exhausted" or err_type == "insufficient_quota":
                    category = f"QuotaExceeded (HTTP {exc.code}: OpenAI account credit balance exhausted)"
                else:
                    category = f"RateLimitError (HTTP {exc.code}: OpenAI request rate limit exceeded)"
            elif exc.code == 404:
                category = f"ModelNotFoundError (HTTP {exc.code}: Model '{self.model}' not found on OpenAI)"
            elif exc.code in (500, 502, 503, 504):
                category = f"ServiceUnavailable (HTTP {exc.code}: OpenAI server error)"
            else:
                category = f"OpenAI provider error (HTTP {exc.code})"

            logger.warning("OpenAI HTTP error [%d]: %s", exc.code, category)
            return {
                "success": False,
                "error": category,
                "error_type": err_type,
                "error_code": err_code,
                "status_code": exc.code,
                "answer": "",
                "provider": "openai",
                "model": self.model,
            }
        except urllib.error.URLError as exc:
            is_timeout = isinstance(exc.reason, (socket.timeout, TimeoutError)) or "timed out" in str(exc.reason).lower()
            category = "TimeoutError (OpenAI request timed out)" if is_timeout else "NetworkError (Cannot connect to api.openai.com)"
            logger.warning("OpenAI network failure: %s", category)
            return {
                "success": False,
                "error": category,
                "error_type": "TimeoutError" if is_timeout else "NetworkError",
                "error_code": "network_failure",
                "status_code": None,
                "answer": "",
                "provider": "openai",
                "model": self.model,
            }
        except Exception as exc:
            logger.warning("OpenAI provider request failure: %s", type(exc).__name__)
            return {
                "success": False,
                "error": f"InternalProviderError ({type(exc).__name__})",
                "error_type": type(exc).__name__,
                "error_code": "internal_error",
                "status_code": None,
                "answer": "",
                "provider": "openai",
                "model": self.model,
            }


class GeminiProvider(BaseFarmerAssistantProvider):
    """Google Gemini API implementation using native REST API structure."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key,
            model=model or (FarmerAssistantConfig.MODEL_NAME if FarmerAssistantConfig.DEFAULT_PROVIDER == "gemini" else None) or FarmerAssistantConfig.DEFAULT_MODELS["gemini"],
        )
        self.endpoint_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate(
        self,
        system_instruction: str,
        user_prompt: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "error": "Gemini API key not configured.",
                "answer": "",
                "provider": "gemini",
                "model": self.model,
            }

        # Build structured context for grounded responses (strictly verified AgriSmart context)
        enriched_prompt = user_prompt
        if context:
            intent = context.get("intent", "GENERAL")
            structured = build_structured_context(context, intent)
            if structured:
                context_json = json.dumps(structured, indent=2, default=str)
                enriched_prompt = (
                    f"AGRISMART MODULE CONTEXT (verified data — do not fabricate beyond this):\n"
                    f"{context_json}\n\n"
                    f"{user_prompt}"
                )

        endpoint_url = f"{self.endpoint_template.format(model=self.model)}?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": enriched_prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 600,
            },
        }

        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=FarmerAssistantConfig.REQUEST_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates:
                    feedback = data.get("promptFeedback", {})
                    block_reason = feedback.get("blockReason")
                    category = f"ContentBlocked (Gemini prompt blocked: {block_reason})" if block_reason else "EmptyResponse (Gemini returned no candidates)"
                    return {
                        "success": False,
                        "error": category,
                        "error_type": "ContentBlocked" if block_reason else "EmptyResponse",
                        "error_code": block_reason or "empty_candidates",
                        "status_code": getattr(resp, "status", 200),
                        "answer": "",
                        "provider": "gemini",
                        "model": self.model,
                    }

                candidate = candidates[0]
                finish_reason = candidate.get("finishReason")
                parts = candidate.get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts).strip()

                if not content and finish_reason in ("SAFETY", "RECITATION", "OTHER"):
                    category = f"SafetyBlocked (Gemini candidate finish reason: {finish_reason})"
                    return {
                        "success": False,
                        "error": category,
                        "error_type": "SafetyBlocked",
                        "error_code": finish_reason,
                        "status_code": 200,
                        "answer": "",
                        "provider": "gemini",
                        "model": self.model,
                    }

                return {
                    "success": True,
                    "answer": content,
                    "provider": "gemini",
                    "model": self.model,
                    "error": None,
                }
        except urllib.error.HTTPError as exc:
            err_type = "HTTPError"
            err_code = None
            err_msg = ""
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
                err_detail = err_body.get("error", {})
                err_code = err_detail.get("status") or err_detail.get("code")
                err_type = err_detail.get("status") or "HTTPError"
                err_msg = str(err_detail.get("message", ""))
            except Exception:
                pass

            if exc.code in (400, 401, 403):
                if "api key" in err_msg.lower() or "api_key" in err_msg.lower() or exc.code in (401, 403):
                    category = f"AuthenticationError (HTTP {exc.code}: Invalid or unauthorized Gemini API key)"
                else:
                    category = f"BadRequestError (HTTP {exc.code}: Gemini API request invalid)"
            elif exc.code == 429:
                if "quota" in err_msg.lower() or "resource_exhausted" in str(err_code).lower():
                    category = f"QuotaExceeded (HTTP {exc.code}: Gemini API free-tier quota exhausted)"
                else:
                    category = f"RateLimitError (HTTP {exc.code}: Gemini request rate limit exceeded)"
            elif exc.code == 404:
                category = f"ModelNotFoundError (HTTP {exc.code}: Model '{self.model}' not found on Gemini API)"
            elif exc.code in (500, 502, 503, 504):
                category = f"ServiceUnavailable (HTTP {exc.code}: Google Gemini service temporarily unavailable)"
            else:
                category = f"Gemini provider error (HTTP {exc.code})"

            clean_category = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", category)
            clean_category = re.sub(r"key=[A-Za-z0-9_-]+", "key=[REDACTED_API_KEY]", clean_category)
            logger.warning("Gemini HTTP error [%d]: %s", exc.code, clean_category)
            return {
                "success": False,
                "error": clean_category,
                "error_type": err_type,
                "error_code": err_code,
                "status_code": exc.code,
                "answer": "",
                "provider": "gemini",
                "model": self.model,
            }
        except urllib.error.URLError as exc:
            is_timeout = isinstance(exc.reason, (socket.timeout, TimeoutError)) or "timed out" in str(exc.reason).lower()
            category = "TimeoutError (Gemini request timed out)" if is_timeout else "NetworkError (Cannot connect to generativelanguage.googleapis.com)"
            logger.warning("Gemini network failure: %s", category)
            return {
                "success": False,
                "error": category,
                "error_type": "TimeoutError" if is_timeout else "NetworkError",
                "error_code": "network_failure",
                "status_code": None,
                "answer": "",
                "provider": "gemini",
                "model": self.model,
            }
        except Exception as exc:
            logger.warning("Gemini provider request failure: %s", type(exc).__name__)
            return {
                "success": False,
                "error": f"InternalProviderError ({type(exc).__name__})",
                "error_type": type(exc).__name__,
                "error_code": "internal_error",
                "status_code": None,
                "answer": "",
                "provider": "gemini",
                "model": self.model,
            }


class LocalRuleProvider(BaseFarmerAssistantProvider):
    """
    Deterministic, grounded rule-based local provider.
    NOT a fake LLM: directly answers supported queries using actual AgriSmart results.
    Explicitly refuses unsupported general chat without an LLM provider.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(api_key="", model=FarmerAssistantConfig.DEFAULT_MODELS["local"])

    def generate(
        self,
        system_instruction: str,
        user_prompt: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = context or {}
        intent = ctx.get("intent", "GENERAL")
        profile = ctx.get("farm_profile", {})
        mods = ctx.get("modules", {})
        lang = language if language in ("en", "hi", "gu") else "en"

        # 1. Irrigation Queries
        if intent == "IRRIGATION":
            irrig = mods.get("irrigation", {})
            w = mods.get("weather", {})
            moisture = profile.get("soil_moisture_pct")

            if irrig.get("status") != "UNAVAILABLE":
                decision = irrig.get("decision", "REVIEW")
                headline = irrig.get("headline", "Smart Irrigation Advisory")
                action = irrig.get("primary_action", "Check soil moisture before watering.")

                moisture_str = f"Soil moisture is currently reported at {moisture:.0f}%." if moisture is not None else "No soil moisture reading was provided."
                weather_note = ""
                if w.get("status") != "UNAVAILABLE":
                    w_status = w.get("status")
                    prob = w.get("max_rain_probability_pct", 0.0)
                    rain_mm = w.get("expected_rain_mm", 0.0)
                    if w.get("is_significant_rain"):
                        weather_note = f"Weather forecast ({w_status}) predicts significant near-term rainfall ({rain_mm:.1f} mm, max prob {prob:.0f}%)."
                    elif w.get("is_negligible_rain"):
                        weather_note = f"Weather forecast ({w_status}) shows a {prob:.0f}% rain chance, but expected rainfall is negligible ({rain_mm:.1f} mm)."
                    else:
                        weather_note = f"Weather conditions ({w_status}) indicate temperature {w.get('temperature_c')}°C with no heavy rain forecast."

                if lang == "hi":
                    moisture_str_hi = f"वर्तमान मिट्टी की नमी {moisture:.0f}% दर्ज है।" if moisture is not None else ""
                    answer = (
                        f"स्मार्ट सिंचाई सलाह: {headline}। "
                        f"{moisture_str_hi} {action} {weather_note}".strip()
                    )
                elif lang == "gu":
                    moisture_str_gu = f"હાલમાં જમીનનો ભેજ {moisture:.0f}% નોંધાયેલ છે." if moisture is not None else ""
                    answer = (
                        f"સ્માર્ટ સિંચાઈ સલાહ: {headline}. "
                        f"{moisture_str_gu} {action} {weather_note}".strip()
                    )
                else:
                    answer = (
                        f"Smart Irrigation recommendation: {headline}. "
                        f"{moisture_str} {action} {weather_note}".strip()
                    )

                return {
                    "success": True,
                    "answer": answer,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            elif moisture is not None:
                if lang == "hi":
                    if moisture < 35:
                        advice = f"दर्ज की गई मिट्टी की नमी {moisture:.0f}% है। नमी कम है, इसलिए फसल में पानी की कमी से पहले सिंचाई की योजना बनाएं।"
                    elif moisture > 80:
                        advice = f"दर्ज की गई मिट्टी की नमी {moisture:.0f}% है। अत्यधिक नमी से बचें और जल निकासी सुनिश्चित करें।"
                    else:
                        advice = f"दर्ज की गई मिट्टी की नमी {moisture:.0f}% है। मिट्टी में नमी सामान्य है; नियमित पानी की निगरानी रखें।"
                elif lang == "gu":
                    if moisture < 35:
                        advice = f"નોંધાયેલ જમીનનો ભેજ {moisture:.0f}% છે. ભેજ ઓછો છે, તેથી પાકમાં પાણીની અછત આવે તે પહેલાં સિંચાઈ કરો."
                    elif moisture > 80:
                        advice = f"નોંધાયેલ જમીનનો ભેજ {moisture:.0f}% છે. ભેજ વધુ પડતો છે; વધારાની સિંચાઈ ટાળો અને ડ્રેનેજ તપાસો."
                    else:
                        advice = f"નોંધાયેલ જમીનનો ભેજ {moisture:.0f}% છે. જમીનમાં ભેજ યોગ્ય સ્તરે છે; નિયમિત દેખરેખ ચાલુ રાખો."
                else:
                    if moisture < 35:
                        advice = f"Reported soil moisture is {moisture:.0f}%. Soil moisture is low; schedule irrigation before crop water stress develops."
                    elif moisture > 80:
                        advice = f"Reported soil moisture is {moisture:.0f}%. Avoid additional irrigation and monitor field drainage."
                    else:
                        advice = f"Reported soil moisture is {moisture:.0f}%. Moisture is in an adequate range (35–65%); continue regular checks."

                return {
                    "success": True,
                    "answer": advice,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            else:
                if lang == "hi":
                    no_moisture_ans = (
                        "मेरे पास वर्तमान मिट्टी की नमी या सक्रिय स्मार्ट सिंचाई मूल्यांकन नहीं है। "
                        "कृपया स्मार्ट सिंचाई मॉड्यूल में नमी की जानकारी दर्ज करें।"
                    )
                elif lang == "gu":
                    no_moisture_ans = (
                        "મારી પાસે હાલના જમીનના ભેજ અથવા સક્રિય સ્માર્ટ સિંચાઈ મૂલ્યાંકનની માહિતી નથી. "
                        "કૃપા કરીને સ્માર્ટ સિંચાઈ મોડ્યુલમાં ભેજ નોંધાવો."
                    )
                else:
                    no_moisture_ans = (
                        "I don't have a current soil-moisture reading or active Smart Irrigation assessment. "
                        "Please enter your manual moisture estimate in the Smart Irrigation module so I can interpret it for you."
                    )
                return {
                    "success": True,
                    "answer": no_moisture_ans,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }

        # 2. Weather Queries
        if intent == "WEATHER":
            w = mods.get("weather", {})
            if w.get("status") != "UNAVAILABLE":
                w_status = w.get("status")
                loc = w.get("location", "your area")
                temp = w.get("temperature_c")
                hum = w.get("relative_humidity_pct")
                prob = w.get("max_rain_probability_pct", 0.0)
                rain_mm = w.get("expected_rain_mm", 0.0)
                desc = w.get("weather_description") or "Partly cloudy"

                temp_str = f"{temp:.1f}°C" if temp is not None else "seasonal"
                hum_str = f"{hum:.0f}%" if hum is not None else "normal"

                demo_tag = " (DEMO data — not for actual field decisions)" if w_status == "DEMO" else " (LIVE Open-Meteo data)"

                if lang == "hi":
                    answer = (
                        f"{loc} के लिए मौसम{demo_tag}: तापमान {temp_str}, आर्द्रता {hum_str}, स्थिति: {desc}। "
                        f"बारिश की संभावना: {prob:.0f}%, अनुमानित वर्षा: {rain_mm:.1f} मिमी।"
                    )
                elif lang == "gu":
                    answer = (
                        f"{loc} માટે હવામાન{demo_tag}: તાપમાન {temp_str}, ભેજ {hum_str}, સ્થિતિ: {desc}. "
                        f"વરસાદની સંભાવના: {prob:.0f}%, અપેક્ષિત વરસાદ: {rain_mm:.1f} મીમી."
                    )
                else:
                    rain_significance = ""
                    if w.get("is_significant_rain"):
                        rain_significance = f" Significant rainfall is expected ({rain_mm:.1f} mm); factor this into irrigation planning."
                    elif w.get("is_negligible_rain"):
                        rain_significance = f" Rainfall probability is {prob:.0f}%, but expected amount is negligible ({rain_mm:.1f} mm)."

                    answer = (
                        f"Weather Intelligence for {loc}{demo_tag}: Temperature is {temp_str} with {hum_str} relative humidity ({desc}). "
                        f"Near-term rain probability is {prob:.0f}% with {rain_mm:.1f} mm expected precipitation.{rain_significance}"
                    )

                return {
                    "success": True,
                    "answer": answer,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            else:
                return {
                    "success": True,
                    "answer": "Weather data is currently unavailable. Enter your district in the Weather Intelligence module to retrieve a forecast.",
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }

        # 3. Sustainability Queries
        if intent == "SUSTAINABILITY":
            sust = mods.get("sustainability", {})
            if sust.get("status") != "UNAVAILABLE":
                score = sust.get("score")
                cat = sust.get("category")
                w_score = sust.get("water_score")
                r_score = sust.get("resource_score")
                h_score = sust.get("crop_health_score")
                recs = sust.get("recommendations", [])
                top_rec = recs[0]["title"] if recs else "Maintain current practices"

                h_display = f"{h_score}/100" if h_score is not None else "Not Assessed"

                if lang == "hi":
                    answer = (
                        f"आपका फॉर्म सस्टेनेबिलिटी स्कोर {score}/100 ({cat}) है। "
                        f"घटक: जल दक्षता {w_score}/100, संसाधन उपयोग {r_score}/100, फसल स्वास्थ्य {h_display}। "
                        f"मुख्य सुधार: {top_rec}।"
                    )
                elif lang == "gu":
                    answer = (
                        f"તમારો ફાર્મ સસ્ટેનેબિલિટી સ્કોર {score}/100 ({cat}) છે. "
                        f"વિભાગો: જળ કાર્યક્ષમતા {w_score}/100, સંસાધન વપરાશ {r_score}/100, પાક આરોગ્ય {h_display}. "
                        f"મુખ્ય પગલું: {top_rec}."
                    )
                else:
                    answer = (
                        f"Your Sustainability Score is {score}/100 ({cat}).\n"
                        f"- Water Efficiency: {w_score}/100\n"
                        f"- Resource Use: {r_score}/100\n"
                        f"- Crop Foliar Health: {h_display}\n\n"
                        f"Top recommended action: {top_rec}."
                    )

                return {
                    "success": True,
                    "answer": answer,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            else:
                return {
                    "success": True,
                    "answer": "Sustainability score has not been calculated yet. Please evaluate your practices in the Sustainability module.",
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }

        # 4. Disease Queries
        if intent == "DISEASE":
            dis = mods.get("disease", {})
            if dis.get("status") != "UNAVAILABLE":
                condition = dis.get("prediction", "Unknown condition")
                d_status = dis.get("status")
                conf = dis.get("confidence")

                if d_status == "DEMO":
                    status_note = "NOTICE: This result was generated in DEMO mode and is not a confirmed field diagnosis."
                else:
                    conf_str = f" with {conf*100:.1f}% model confidence" if conf is not None else ""
                    status_note = f"The screening model identified features consistent with this condition{conf_str}."

                treatment = dis.get("treatment") or "Inspect representative plants across the canopy and consult local extension officers."

                if lang == "hi":
                    answer = (
                        f"फसल रोग स्क्रीनिंग परिणाम: {condition}। {status_note} "
                        f"सलाह: प्रतिनिधि पौधों का निरीक्षण करें। {treatment}"
                    )
                elif lang == "gu":
                    answer = (
                        f"પાક રોગ સ્ક્રીનીંગ પરિણામ: {condition}. {status_note} "
                        f"સલાહ: મુખ્ય છોડનું નિરીક્ષણ કરો. {treatment}"
                    )
                else:
                    answer = (
                        f"Crop Disease Screening result: {condition}.\n"
                        f"{status_note}\n\n"
                        f"Next steps: Inspect representative plants for lesion spread before applying treatments. {treatment}"
                    )

                return {
                    "success": True,
                    "answer": answer,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            else:
                return {
                    "success": True,
                    "answer": "No crop disease analysis is active. Upload a leaf image on the Disease Detection page for foliar screening.",
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }

        # 5. Crop Queries
        if intent == "CROP":
            cr = mods.get("crop_recommendation", {})
            if cr.get("status") != "UNAVAILABLE":
                rec = cr.get("recommended_crop", "Unknown")
                conf = cr.get("confidence")
                conf_text = f" (model confidence: {conf*100:.1f}%)" if conf is not None else ""

                if lang == "hi":
                    answer = f"फसल सिफारिश मॉडल ने आपके खेत के लिए '{rec}'{conf_text} की सिफारिश की है।"
                elif lang == "gu":
                    answer = f"પાક ભલામણ મોડેલે તમારા ખેતર માટે '{rec}'{conf_text} ની ભલામણ કરી છે."
                else:
                    answer = (
                        f"The Crop Recommendation model recommended: {rec}{conf_text}. "
                        "This recommendation is derived from your soil N-P-K levels, temperature, humidity, pH, and rainfall inputs."
                    )

                return {
                    "success": True,
                    "answer": answer,
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }
            else:
                return {
                    "success": True,
                    "answer": "No crop recommendation has been computed yet. Enter your soil N-P-K and climate parameters in Crop Recommendation.",
                    "provider": "local",
                    "model": self.model,
                    "error": None,
                }

        # 6. General / Compound / Integrated Queries
        has_active_modules = (
            any(m.get("status") not in ("UNAVAILABLE", None) for m in mods.values())
            or bool(profile.get("crop") or profile.get("soil_moisture_pct") is not None or profile.get("location"))
        )

        if has_active_modules:
            crop = profile.get("crop", "crop")
            stage = profile.get("growth_stage", "")
            stage_str = f" ({stage} stage)" if stage else ""
            moisture = profile.get("soil_moisture_pct")
            moisture_str = f"{moisture:.0f}%" if moisture is not None else "not specified"
            irrig_sys = profile.get("irrigation_method", "standard irrigation")
            loc = profile.get("location") or mods.get("weather", {}).get("location", "your area")

            # A. Irrigation Summary
            irrig = mods.get("irrigation", {})
            if irrig.get("status") != "UNAVAILABLE":
                irrig_summary = f"{irrig.get('headline', 'Advisory active')}: {irrig.get('primary_action', 'Check moisture.')}"
            elif moisture is not None:
                if moisture < 35:
                    irrig_summary = f"Soil moisture is low ({moisture_str}). Schedule irrigation before water stress develops."
                elif moisture > 70:
                    irrig_summary = f"Soil moisture is high ({moisture_str}). Hold further irrigation to avoid root waterlogging."
                else:
                    irrig_summary = f"Soil moisture is adequate ({moisture_str}). Maintain current schedule."
            else:
                irrig_summary = "Provide a soil moisture estimate for customized watering guidance."

            # B. Weather Summary
            w = mods.get("weather", {})
            if w.get("status") != "UNAVAILABLE":
                w_status = w.get("status")
                w_temp = f"{w.get('temperature_c')}°C" if w.get("temperature_c") is not None else "seasonal"
                w_hum = f"{w.get('relative_humidity_pct')}%" if w.get("relative_humidity_pct") is not None else "normal"
                w_rain = f"{w.get('max_rain_probability_pct', 0):.0f}% chance, {w.get('expected_rain_mm', 0):.1f} mm"
                w_tag = " (DEMO data)" if w_status == "DEMO" else " (LIVE Open-Meteo)"
                weather_summary = f"Current conditions in {loc}{w_tag}: {w_temp}, {w_hum} humidity, rain: {w_rain}."
            else:
                weather_summary = "Weather forecast is currently unavailable. Enter your district in Weather Intelligence to load forecast data."

            # C. Crop Health & Disease Summary
            dis = mods.get("disease", {})
            if dis.get("status") != "UNAVAILABLE":
                dis_cond = dis.get("prediction", "Identified symptoms")
                dis_note = " (DEMO mode — not confirmed diagnosis)" if dis.get("status") == "DEMO" else ""
                disease_summary = f"Disease screening indicates: {dis_cond}{dis_note}. Inspect representative plants across the field."
            else:
                disease_summary = f"No active foliar disease recorded. Regularly scout your {crop} field for leaf spots or fungal symptoms."

            # D. Sustainability Summary
            sust = mods.get("sustainability", {})
            if sust.get("status") != "UNAVAILABLE":
                s_score = sust.get("score")
                s_cat = sust.get("category", "")
                recs = sust.get("recommendations", [])
                top_r = recs[0]["title"] if recs else "maintain balanced nutrient and water practices"
                sust_summary = f"Sustainability Score is {s_score}/100 ({s_cat}). Top action: {top_r}."
            else:
                sust_summary = "Sustainability Score has not been computed yet. Evaluate your practices in the Sustainability module."

            if lang == "hi":
                answer = (
                    f"आपके खेत संदर्भ के अनुसार एकीकृत कृषि परामर्श ({crop}{stage_str}):\n\n"
                    f"1. सिंचाई सलाह: {irrig_summary}\n"
                    f"2. मौसम की स्थिति: {weather_summary}\n"
                    f"3. फसल स्वास्थ्य: {disease_summary}\n"
                    f"4. सस्टेनेबिलिटी सुधार: {sust_summary}"
                )
            elif lang == "gu":
                answer = (
                    f"તમારા ખેતર સંદર્ભ મુજબ એકીકૃત કૃષિ સલાહ ({crop}{stage_str}):\n\n"
                    f"1. સિંચાઈ સલાહ: {irrig_summary}\n"
                    f"2. હવામાન સ્થિતિ: {weather_summary}\n"
                    f"3. પાક આરોગ્ય: {disease_summary}\n"
                    f"4. સસ્ટેનેબિલિટી સુધારો: {sust_summary}"
                )
            else:
                answer = (
                    f"Integrated Farm Advisory based on your active context ({crop}{stage_str}, {irrig_sys}):\n\n"
                    f"• Irrigation Advisory: {irrig_summary}\n"
                    f"• Weather Conditions: {weather_summary}\n"
                    f"• Crop Health & Disease: {disease_summary}\n"
                    f"• Sustainability Actions: {sust_summary}"
                )

            return {
                "success": True,
                "answer": answer,
                "provider": "local",
                "model": self.model,
                "error": None,
            }

        # Fallback for generic unsupported open-domain conversation without module data
        if lang == "hi":
            refusal = (
                "असिस्टेंट डेमो / लोकल रूल मोड वर्तमान में एग्रीस्मार्ट मॉड्यूल परिणामों "
                "(सिंचाई, मौसम, सस्टेनेबिलिटी, रोग और फसल सिफारिश) की व्याख्या कर सकता है। "
                "व्यापक बातचीत के लिए कृपया GenAI प्रदाता कॉन्फ़िगर करें।"
            )
        elif lang == "gu":
            refusal = (
                "આસિસ્ટન્ટ ડેમો / લોકલ રૂલ મોડ હાલમાં એગ્રીસ્માર્ટ મોડ્યુલ પરિણામો "
                "(સિંચાઈ, હવામાન, સસ્ટેનેબિલિટી, રોગ અને પાક ભલામણ) સમજાવી શકે છે. "
                "વધુ વાતચીત માટે કૃપા કરીને GenAI પ્રદાતા ગોઠવો."
            )
        else:
            refusal = (
                "Assistant Demo / Local Rule Mode can currently explain AgriSmart module results "
                "(Irrigation, Weather, Sustainability, Disease, and Crop Recommendation). "
                "Configure a GenAI provider (OpenAI or Gemini) for broader conversational assistance."
            )

        return {
            "success": True,
            "answer": refusal,
            "provider": "local",
            "model": self.model,
            "error": None,
        }


class ProviderFactory:
    """Factory to instantiate the configured LLM provider."""

    @staticmethod
    def get_provider(
        provider_type: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> BaseFarmerAssistantProvider:
        p_type = (provider_type if provider_type is not None else FarmerAssistantConfig.DEFAULT_PROVIDER).lower().strip()
        key = api_key.strip() if api_key is not None else FarmerAssistantConfig.API_KEY

        if p_type == "openai" and key:
            return OpenAIProvider(api_key=key, model=model)
        elif p_type == "gemini" and key:
            return GeminiProvider(api_key=key, model=model)
        else:
            # Fallback to local rule mode
            return LocalRuleProvider(api_key=None, model=model)
