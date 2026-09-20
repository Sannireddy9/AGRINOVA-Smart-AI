"""
AgriSmart AI — Weather Intelligence Configuration
=================================================
Centralized heuristic thresholds, WMO code mappings, and provider endpoints.

NOTICE ON THRESHOLDS:
Configurable rule-based heuristic thresholds for demonstration and advisory purposes.
Actual thresholds vary by crop, soil type, growth stage, climate, irrigation system,
and measurement method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class WeatherIntelligenceConfig:
    """Central configuration for weather-based agricultural intelligence."""

    # ── Provider & API Endpoints (Verified Open-Meteo) ───────────────────
    PROVIDER_NAME: ClassVar[str] = "Open-Meteo"
    PROVIDER_DOCS_URL: ClassVar[str] = "https://open-meteo.com/en/docs"
    GEOCODING_API_URL: ClassVar[str] = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_API_URL: ClassVar[str] = "https://api.open-meteo.com/v1/forecast"

    # Default request timeout in seconds
    REQUEST_TIMEOUT_SECONDS: ClassVar[int] = 8

    # In-memory cache time-to-live in minutes to prevent rate-limiting
    CACHE_TTL_MINUTES: ClassVar[int] = 15

    # ── Configurable Agronomic Heuristic Thresholds ───────────────────────
    # Precipitation: Rain probability >= 60% or rain amount >= 10 mm triggers delay
    RAIN_LIKELY_PROB_THRESHOLD: ClassVar[float] = 60.0  # %
    RAIN_HEAVY_AMOUNT_THRESHOLD: ClassVar[float] = 10.0  # mm
    SIGNIFICANT_RAIN_AMOUNT_THRESHOLD: ClassVar[float] = 8.0  # mm (meaningful rain)
    LIGHT_RAIN_AMOUNT_THRESHOLD: ClassVar[float] = 2.0  # mm (light shower vs soaking rain)
    HEAVY_FUTURE_RAIN_THRESHOLD: ClassVar[float] = 15.0  # mm (advance 7-day warning trigger)

    # Fungal & Bacterial Foliar Risk: Sustained humidity >= 75% triggers elevated disease alert
    HIGH_HUMIDITY_THRESHOLD: ClassVar[float] = 75.0  # %
    MODERATE_HUMIDITY_THRESHOLD: ClassVar[float] = 60.0  # %

    # Thermal & Atmospheric Stress
    HEAT_STRESS_THRESHOLD: ClassVar[float] = 35.0  # °C
    HIGH_WIND_SPRAY_THRESHOLD: ClassVar[float] = 20.0  # km/h (avoid pesticide drift)

    # Soil Moisture Depletion (Farmer-provided manual percentage)
    SOIL_MOISTURE_CRITICAL_LOW: ClassVar[float] = 25.0  # % (wilting risk)
    SOIL_MOISTURE_DEPLETED: ClassVar[float] = 35.0  # % (irrigation consideration trigger)
    SOIL_MOISTURE_OPTIMAL_HIGH: ClassVar[float] = 70.0  # % (adequate soil moisture)

    # Disclaimers
    SPRAY_PRODUCT_LABEL_DISCLAIMER: ClassVar[str] = (
        "Weather conditions are evaluated for spray drift and wash-off suitability only. "
        "This is not a chemical application recommendation. Always consult the pesticide or fertilizer "
        "product label, local agricultural extension guidance, and statutory regulations."
    )
    DISEASE_WEATHER_DISCLAIMER: ClassVar[str] = (
        "This is an agrometeorological disease-weather risk index, not a plant diagnosis. "
        "Weather conditions indicate favorable environments for spore germination, not infection presence. "
        "Inspect your crops visually and confirm symptoms using the Computer Vision foliar diagnosis module."
    )

    # ── WMO Weather Interpretation Codes (WMO 0-99) ──────────────────────
    WMO_CODE_MAP: ClassVar[dict[int, tuple[str, str]]] = {
        0: ("Clear sky", "☀️"),
        1: ("Mainly clear", "🌤️"),
        2: ("Partly cloudy", "⛅"),
        3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"),
        48: ("Depositing rime fog", "🌫️"),
        51: ("Light drizzle", "🌦️"),
        53: ("Moderate drizzle", "🌦️"),
        55: ("Dense drizzle", "🌧️"),
        61: ("Slight rain", "🌦️"),
        63: ("Moderate rain", "🌧️"),
        65: ("Heavy rain", "🌧️"),
        71: ("Slight snow", "🌨️"),
        73: ("Moderate snow", "🌨️"),
        75: ("Heavy snow", "🌨️"),
        80: ("Slight rain showers", "🌦️"),
        81: ("Moderate rain showers", "🌧️"),
        82: ("Violent rain showers", "⛈️"),
        95: ("Thunderstorm", "⛈️"),
        96: ("Thunderstorm with slight hail", "⛈️"),
        99: ("Thunderstorm with heavy hail", "⛈️"),
    }

    @classmethod
    def get_wmo_condition(cls, code: int) -> tuple[str, str]:
        """Return (description, icon) for a given WMO weather code."""
        return cls.WMO_CODE_MAP.get(code, ("Variable conditions", "🌤️"))
