"""
AgriSmart AI — Farmer AI Assistant Service (Future Bonus Module)
=================================================================
Interface placeholder for localized conversational agronomy advice.
"""

from __future__ import annotations
from typing import Any


class FarmerAssistantService:
    """Service boundary for conversational agronomy assistant."""

    def answer_query(
        self,
        query: str,
        farmer_language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate conversational agricultural advisory response."""
        raise NotImplementedError("Farmer AI Assistant module is planned for a future phase.")
