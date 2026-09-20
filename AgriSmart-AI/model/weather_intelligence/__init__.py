"""
AgriSmart AI — Weather-Based Intelligence Module
=================================================
Rule-based agricultural weather intelligence combining meteorological
data with farm context to deliver actionable agronomic advice.
"""

from model.weather_intelligence.config import WeatherIntelligenceConfig
from model.weather_intelligence.analyzer import analyze_farm_weather

__all__ = ["WeatherIntelligenceConfig", "analyze_farm_weather"]
