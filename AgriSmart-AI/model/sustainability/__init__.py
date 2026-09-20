"""
AgriSmart AI — Sustainability Score Module
===========================================
Deterministic, rule-based agricultural sustainability scoring index
evaluating Water Efficiency (40%), Resource Use (30%), and Crop Health (30%)
with transparent missing-data weight renormalization and Zero-IoT architecture.
"""

from model.sustainability.config import SustainabilityConfig
from model.sustainability.rules import (
    evaluate_water_efficiency,
    evaluate_resource_use,
    evaluate_crop_health,
    generate_sustainability_recommendations,
)
from model.sustainability.analyzer import analyze_sustainability_score

__all__ = [
    "SustainabilityConfig",
    "evaluate_water_efficiency",
    "evaluate_resource_use",
    "evaluate_crop_health",
    "generate_sustainability_recommendations",
    "analyze_sustainability_score",
]
