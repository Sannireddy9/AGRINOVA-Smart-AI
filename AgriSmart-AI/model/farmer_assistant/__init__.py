"""
AgriSmart AI — Farmer Assistant (GenAI) Package
===============================================
Provides conversational, grounded agricultural decision support
anchored strictly to AgriSmart module outputs and farmer field context.
"""

from __future__ import annotations

from model.farmer_assistant.config import FarmerAssistantConfig
from model.farmer_assistant.assistant import FarmerAssistant
from model.farmer_assistant.context_builder import (
    classify_intent,
    build_farm_context,
)
from model.farmer_assistant.guardrails import (
    check_prompt_injection,
    check_hazardous_chemicals,
)
from model.farmer_assistant.providers import (
    BaseFarmerAssistantProvider,
    OpenAIProvider,
    GeminiProvider,
    LocalRuleProvider,
    ProviderFactory,
)

__all__ = [
    "FarmerAssistantConfig",
    "FarmerAssistant",
    "classify_intent",
    "build_farm_context",
    "check_prompt_injection",
    "check_hazardous_chemicals",
    "BaseFarmerAssistantProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "LocalRuleProvider",
    "ProviderFactory",
]
