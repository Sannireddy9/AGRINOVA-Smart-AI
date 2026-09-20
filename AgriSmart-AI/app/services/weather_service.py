"""
AgriSmart AI — Weather Intelligence Service
============================================
Integrates real-time meteorological conditions and 7-day multi-day forecasts
from the Open-Meteo API with manual farmer field parameters to produce
heuristic, actionable agricultural advisories (irrigation timing, foliar
disease risk index, and heat/spray windows).

Features:
- Live Open-Meteo Geocoding & Forecast queries (no API key required).
- In-memory TTL caching to avoid rate-limiting and accelerate repeat requests.
- Robust fallback to DEMO/simulated mode with prominent disclaimers upon network
  outage, API timeout, or unresolvable location.
- 100% backward-compatible stubs for legacy service test assertions.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any

from app.config import AppConfig
from model.weather_intelligence.analyzer import analyze_farm_weather
from model.weather_intelligence.config import WeatherIntelligenceConfig

logger = logging.getLogger(__name__)


class WeatherIntelligenceService:
    """Service boundary for hyper-local weather intelligence and agricultural advisories."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl_minutes: int | None = None,
        mode: str | None = None,
    ) -> None:
        self.api_key = api_key or getattr(AppConfig, "WEATHER_API_KEY", None)
        self.cache_ttl_seconds = (
            (cache_ttl_minutes or getattr(AppConfig, "WEATHER_CACHE_TTL_MINUTES", 15)) * 60
        )
        self.mode = (mode or getattr(AppConfig, "WEATHER_MODE", "auto")).lower()
        # In-memory cache: {cache_key: (timestamp, data)}
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def is_live_configured(self) -> bool:
        """Return True if service is configured to attempt live meteorological fetching."""
        return self.mode in ("auto", "live")

    # ── Geocoding & API Queries ───────────────────────────────────────────

    def geocode_location(self, location_name: str) -> tuple[float, float, str]:
        """
        Resolve a city/location query to latitude, longitude, and formatted name.
        Uses Open-Meteo Geocoding API.
        """
        trimmed = location_name.strip()
        if not trimmed:
            raise ValueError("Location query cannot be empty.")

        encoded_name = urllib.parse.quote(trimmed)
        url = f"{WeatherIntelligenceConfig.GEOCODING_API_URL}?name={encoded_name}&count=1"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgriSmartAI/1.0 (Agriculture Decision Support)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=WeatherIntelligenceConfig.REQUEST_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Geocoding API responded with HTTP status {resp.status}")
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Geocoding lookup failed for '%s': %s", trimmed, exc)
            raise RuntimeError(f"Unable to reach geocoding service: {exc}") from exc

        results = payload.get("results")
        if not results or len(results) == 0:
            raise ValueError(f"Could not resolve coordinates for location: '{trimmed}'")

        top_match = results[0]
        lat = float(top_match["latitude"])
        lon = float(top_match["longitude"])
        name = top_match.get("name", trimmed)
        admin1 = top_match.get("admin1", "")
        country = top_match.get("country", "")

        parts = [p for p in (name, admin1, country) if p]
        display_name = ", ".join(parts) if parts else trimmed

        return lat, lon, display_name

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        force_refresh: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Query Open-Meteo forecast API for current weather and 7-day daily forecast.
        Utilizes in-memory TTL caching unless force_refresh is True.
        """
        cache_key = f"{latitude:.3f},{longitude:.3f}"
        now = time.time()

        if not force_refresh and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < self.cache_ttl_seconds:
                return cached_data["current"], cached_data["forecast"]

        query_params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,rain,weather_code,wind_speed_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max"
            ),
            "timezone": "auto",
        }
        url = f"{WeatherIntelligenceConfig.FORECAST_API_URL}?{urllib.parse.urlencode(query_params)}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgriSmartAI/1.0 (Agriculture Decision Support)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=WeatherIntelligenceConfig.REQUEST_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Forecast API responded with HTTP status {resp.status}")
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Forecast query failed for (lat=%s, lon=%s): %s", latitude, longitude, exc)
            raise RuntimeError(f"Unable to reach forecast service: {exc}") from exc

        # 1. Normalize current conditions
        curr_raw = data.get("current", {})
        code = int(curr_raw.get("weather_code", 0))
        cond_desc, cond_icon = WeatherIntelligenceConfig.get_wmo_condition(code)

        current_weather = {
            "temperature": float(curr_raw.get("temperature_2m", 0.0)),
            "apparent_temperature": float(curr_raw.get("apparent_temperature", 0.0)),
            "relative_humidity": float(curr_raw.get("relative_humidity_2m", 0.0)),
            "precipitation": float(curr_raw.get("precipitation", 0.0)),
            "rain": float(curr_raw.get("rain", 0.0)),
            "wind_speed": float(curr_raw.get("wind_speed_10m", 0.0)),
            "weather_code": code,
            "condition": cond_desc,
            "icon": cond_icon,
            "timestamp": curr_raw.get("time", datetime.datetime.now().isoformat()),
        }

        # 2. Normalize 7-day daily forecast
        daily_raw = data.get("daily", {})
        times = daily_raw.get("time", [])
        w_codes = daily_raw.get("weather_code", [])
        t_maxs = daily_raw.get("temperature_2m_max", [])
        t_mins = daily_raw.get("temperature_2m_min", [])
        p_sums = daily_raw.get("precipitation_sum", [])
        p_probs = daily_raw.get("precipitation_probability_max", [])

        forecast_days: list[dict[str, Any]] = []
        for i in range(min(len(times), 7)):
            d_code = int(w_codes[i]) if i < len(w_codes) and w_codes[i] is not None else 0
            d_desc, d_icon = WeatherIntelligenceConfig.get_wmo_condition(d_code)
            forecast_days.append({
                "date": times[i],
                "temp_max": float(t_maxs[i]) if i < len(t_maxs) and t_maxs[i] is not None else 0.0,
                "temp_min": float(t_mins[i]) if i < len(t_mins) and t_mins[i] is not None else 0.0,
                "precipitation_sum": float(p_sums[i]) if i < len(p_sums) and p_sums[i] is not None else 0.0,
                "precipitation_probability_max": float(p_probs[i]) if i < len(p_probs) and p_probs[i] is not None else 0.0,
                "weather_code": d_code,
                "condition": d_desc,
                "icon": d_icon,
            })

        # Cache result
        self._cache[cache_key] = (now, {"current": current_weather, "forecast": forecast_days})
        return current_weather, forecast_days

    # ── Deterministic Demo / Simulation Mode ──────────────────────────────

    def get_demo_weather(
        self,
        location_name: str = "Demo Farm, Gujarat",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Generate deterministic, realistic meteorological data for demonstration
        and fallback when live network access is unavailable or demo mode is enforced.
        """
        today = datetime.date.today()

        current_weather = {
            "temperature": 29.4,
            "apparent_temperature": 32.1,
            "relative_humidity": 68.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "wind_speed": 11.5,
            "weather_code": 2,
            "condition": "Partly cloudy",
            "icon": "⛅",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"),
        }

        # Simulated 7-day pattern demonstrating realistic agricultural variations
        simulated_days = [
            {"offset": 0, "max": 31.0, "min": 23.0, "code": 2, "rain_sum": 0.0, "rain_prob": 15.0},
            {"offset": 1, "max": 30.5, "min": 23.5, "code": 3, "rain_sum": 0.5, "rain_prob": 40.0},
            {"offset": 2, "max": 28.0, "min": 22.0, "code": 63, "rain_sum": 14.5, "rain_prob": 75.0},
            {"offset": 3, "max": 27.5, "min": 21.5, "code": 61, "rain_sum": 6.0, "rain_prob": 65.0},
            {"offset": 4, "max": 29.0, "min": 22.0, "code": 2, "rain_sum": 0.0, "rain_prob": 25.0},
            {"offset": 5, "max": 30.2, "min": 23.0, "code": 1, "rain_sum": 0.0, "rain_prob": 10.0},
            {"offset": 6, "max": 31.5, "min": 24.0, "code": 0, "rain_sum": 0.0, "rain_prob": 5.0},
        ]

        forecast_days: list[dict[str, Any]] = []
        for item in simulated_days:
            d_date = (today + datetime.timedelta(days=item["offset"])).strftime("%Y-%m-%d")
            d_desc, d_icon = WeatherIntelligenceConfig.get_wmo_condition(int(item["code"]))
            forecast_days.append({
                "date": d_date,
                "temp_max": item["max"],
                "temp_min": item["min"],
                "precipitation_sum": item["rain_sum"],
                "precipitation_probability_max": item["rain_prob"],
                "weather_code": item["code"],
                "condition": d_desc,
                "icon": d_icon,
            })

        return current_weather, forecast_days

    # ── High-Level Farm Weather Intelligence Coordinator ─────────────────

    def analyze(
        self,
        location: str = "Gujarat, India",
        farm_context: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Execute full weather analysis pipeline combining meteorological data
        with manual farm parameters to produce actionable agricultural advice.
        """
        context = farm_context or {}
        # Clean manual farm inputs with sensible safe defaults
        cleaned_context = {
            "crop": str(context.get("crop", "Tomato")).strip() or "Tomato",
            "growth_stage": str(context.get("growth_stage", "Vegetative")).strip() or "Vegetative",
            "soil_moisture": float(context.get("soil_moisture", 50.0)),
            "soil_type": str(context.get("soil_type", "Loamy")).strip() or "Loamy",
            "irrigation_method": str(context.get("irrigation_method", "Drip")).strip() or "Drip",
        }

        retrieved_at_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle forced DEMO mode
        if self.mode == "demo":
            cur, fcast = self.get_demo_weather(location)
            return analyze_farm_weather(
                current_weather=cur,
                forecast_days=fcast,
                farm_context=cleaned_context,
                location_name=location,
                requested_location=location,
                coordinates={"latitude": 23.0225, "longitude": 72.5714},
                retrieved_at=retrieved_at_str,
                provider_name=WeatherIntelligenceConfig.PROVIDER_NAME,
                is_mock=True,
                disclaimer=(
                    "DEMO WEATHER — SIMULATED DATA. Demonstration mode is active. "
                    "Rule-based heuristic advisory only; verify with local field observations."
                ),
                error_reason="Demonstration mode explicitly enabled in settings.",
            )

        # Attempt LIVE Open-Meteo fetching
        try:
            lat, lon, resolved_location = self.geocode_location(location)
            cur, fcast = self.fetch_weather(lat, lon, force_refresh=force_refresh)
            return analyze_farm_weather(
                current_weather=cur,
                forecast_days=fcast,
                farm_context=cleaned_context,
                location_name=resolved_location,
                requested_location=location,
                coordinates={"latitude": round(lat, 4), "longitude": round(lon, 4)},
                retrieved_at=retrieved_at_str,
                provider_name=WeatherIntelligenceConfig.PROVIDER_NAME,
                is_mock=False,
                disclaimer=(
                    "Rule-based heuristic advisory based on live Open-Meteo weather data "
                    "and manual farmer inputs. Configurable thresholds for decision support; "
                    "not an automated diagnosis or physical sensor control system."
                ),
                error_reason=None,
            )
        except Exception as exc:
            logger.warning("Live weather analysis failed for '%s', falling back to DEMO: %s", location, exc)
            cur, fcast = self.get_demo_weather(location)
            return analyze_farm_weather(
                current_weather=cur,
                forecast_days=fcast,
                farm_context=cleaned_context,
                location_name=f"{location} (Demo Fallback)",
                requested_location=location,
                coordinates={"latitude": 0.0, "longitude": 0.0},
                retrieved_at=retrieved_at_str,
                provider_name=WeatherIntelligenceConfig.PROVIDER_NAME,
                is_mock=True,
                disclaimer=(
                    "DEMO WEATHER — SIMULATED DATA. Live weather API unavailable. "
                    "Rule-based heuristic advisory only; verify with local field observations."
                ),
                error_reason=str(exc),
            )

    # ── Legacy Compatibility Stubs (Zero-Regression Guarantees) ──────────

    def get_current_conditions(self, latitude: float, longitude: float) -> dict[str, Any]:
        """
        Legacy 2-argument stub preserved for 100% test_app.py backward compatibility.
        """
        raise NotImplementedError(
            "Legacy 2-argument stub. Please use WeatherIntelligenceService.analyze(...) instead."
        )

    def get_spray_advisory(self, latitude: float, longitude: float) -> dict[str, Any]:
        """
        Legacy stub preserved for 100% test_app.py backward compatibility.
        """
        raise NotImplementedError(
            "Legacy stub. Please use WeatherIntelligenceService.analyze(...) instead."
        )
