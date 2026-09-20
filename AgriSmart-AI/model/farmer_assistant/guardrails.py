"""
AgriSmart AI — Farmer Assistant Safety & Grounding Guardrails
============================================================
Enforces strict input validation, prompt injection blocking,
hazardous chemical prevention, and data confidentiality.
"""

from __future__ import annotations

import re
from typing import Any
from model.farmer_assistant.config import FarmerAssistantConfig


def check_prompt_injection(query: str) -> tuple[bool, str | None]:
    """
    Detect adversarial attempts to extract system prompts, API keys,
    environment variables, or force hallucination/mode spoofing.
    """
    clean_q = str(query or "").lower().strip()

    for pattern in FarmerAssistantConfig.PROMPT_INJECTION_PATTERNS:
        if pattern in clean_q:
            return (
                True,
                "I cannot fulfill requests to reveal system instructions, API keys, environment settings, "
                "or alter platform operational data modes. How can I assist you with your farming decisions?",
            )

    return False, None


def check_hazardous_chemicals(query: str) -> tuple[bool, str | None]:
    """
    Detect requests for hazardous chemical manufacturing, dangerous off-label
    pesticide cocktails, or harmful substance creation.
    """
    clean_q = str(query or "").lower().strip()

    for pattern in FarmerAssistantConfig.HAZARDOUS_CHEMICAL_PATTERNS:
        if pattern in clean_q:
            return (
                True,
                "For safety and regulatory reasons, I cannot provide chemical synthesis, off-label pesticide mixing, "
                "or unverified chemical dosage instructions. Please refer to registered product labels and consult your "
                "local Krishi Vigyan Kendra (KVK) or certified agricultural extension officer.",
            )

    return False, None


def sanitize_text(text: str) -> str:
    """
    Sanitize generated text to prevent accidental exposure of tokens,
    keys, or internal environment traces.
    """
    if not text:
        return ""

    # Remove potential key formats like sk-..., AIza..., etc.
    sanitized = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_API_KEY]", text)
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", sanitized)
    return sanitized
