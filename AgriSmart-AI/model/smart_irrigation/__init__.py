"""
AgriSmart AI — Smart Irrigation Advisor Engine
==============================================
A transparent, deterministic, rule-based decision-support system
for agricultural irrigation timing based on manual soil moisture,
meteorological forecasts, crop type, and phenological growth stage.
"""

from model.smart_irrigation.config import SmartIrrigationConfig
from model.smart_irrigation.rules import (
    evaluate_smart_irrigation,
    detect_future_rain_notice,
)
from model.smart_irrigation.analyzer import analyze_irrigation_advisory

__all__ = [
    "SmartIrrigationConfig",
    "evaluate_smart_irrigation",
    "detect_future_rain_notice",
    "analyze_irrigation_advisory",
]
