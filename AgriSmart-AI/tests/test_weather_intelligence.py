"""
AgriSmart AI — Weather Intelligence Test Suite
================================================
Comprehensive unit, integration, and service tests for the Weather-Based
Intelligence module (SIH 2026 Phase 3 & Phase 3.1).

Coverage:
1. Open-Meteo Configuration & WMO Code Dictionary
2. Pure Agronomic Heuristic Rules (Irrigation, Disease Risk, Spray & Heat Windows)
3. Context-Aware Irrigation Priorities (Critical Moisture, Rain Prob vs Amount, 7-Day Advance Notice)
4. Forecast Extremes & Farm Advisory Summary Synthesis
5. "Why This Recommendation?" Explainability
6. Zero-IoT Manual Farmer Context Handling
7. WeatherIntelligenceService (Geocoding, Cache & Force-Refresh, Fallback & Demo Mode)
8. Non-Diagnosis & Transparency Disclaimers
9. Legacy Compatibility Stubs (Zero Regression)
10. Flask Web Routes (GET /weather, POST /analyze-weather)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from app.main import create_app
from app.services.weather_service import WeatherIntelligenceService
from model.weather_intelligence.config import WeatherIntelligenceConfig
from model.weather_intelligence.rules import (
    compute_forecast_extremes,
    detect_future_rainfall_event,
    evaluate_disease_weather_risk,
    evaluate_heat_and_spray_advisory,
    evaluate_irrigation_rule,
    generate_farm_advisory_summary,
)
from model.weather_intelligence.analyzer import analyze_farm_weather


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & WMO DICTIONARY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherConfig:
    """Verify provider settings, heuristic thresholds, and WMO mappings."""

    def test_provider_settings(self):
        assert WeatherIntelligenceConfig.PROVIDER_NAME == "Open-Meteo"
        assert "open-meteo.com" in WeatherIntelligenceConfig.PROVIDER_DOCS_URL
        assert "geocoding-api.open-meteo.com" in WeatherIntelligenceConfig.GEOCODING_API_URL
        assert "api.open-meteo.com" in WeatherIntelligenceConfig.FORECAST_API_URL

    def test_heuristic_thresholds(self):
        assert WeatherIntelligenceConfig.RAIN_LIKELY_PROB_THRESHOLD == 60.0
        assert WeatherIntelligenceConfig.RAIN_HEAVY_AMOUNT_THRESHOLD == 10.0
        assert WeatherIntelligenceConfig.SIGNIFICANT_RAIN_AMOUNT_THRESHOLD == 8.0
        assert WeatherIntelligenceConfig.LIGHT_RAIN_AMOUNT_THRESHOLD == 2.0
        assert WeatherIntelligenceConfig.HIGH_HUMIDITY_THRESHOLD == 75.0
        assert WeatherIntelligenceConfig.HEAT_STRESS_THRESHOLD == 35.0
        assert WeatherIntelligenceConfig.HIGH_WIND_SPRAY_THRESHOLD == 20.0
        assert WeatherIntelligenceConfig.SOIL_MOISTURE_CRITICAL_LOW == 25.0
        assert WeatherIntelligenceConfig.SOIL_MOISTURE_DEPLETED == 35.0

    def test_wmo_code_resolution(self):
        desc_0, icon_0 = WeatherIntelligenceConfig.get_wmo_condition(0)
        assert desc_0 == "Clear sky"
        assert "☀️" in icon_0

        desc_65, icon_65 = WeatherIntelligenceConfig.get_wmo_condition(65)
        assert desc_65 == "Heavy rain"

        desc_95, icon_95 = WeatherIntelligenceConfig.get_wmo_condition(95)
        assert desc_95 == "Thunderstorm"

        # Unknown code fallback
        desc_unk, icon_unk = WeatherIntelligenceConfig.get_wmo_condition(9999)
        assert desc_unk == "Variable conditions"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORECAST EXTREMES & ADVISORY SUMMARY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastExtremesAndSummary:
    """Verify calculation of forecast extremes and advisory summary."""

    def test_compute_forecast_extremes(self):
        forecast_days = [
            {"date": "2026-09-11", "precipitation_probability_max": 25.0, "precipitation_sum": 0.5},
            {"date": "2026-09-12", "precipitation_probability_max": 85.0, "precipitation_sum": 4.0},
            {"date": "2026-09-13", "precipitation_probability_max": 70.0, "precipitation_sum": 18.5},
            {"date": "2026-09-14", "precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
        ]
        res = compute_forecast_extremes(forecast_days)
        assert res["highest_rain_probability"]["probability"] == 85.0
        assert res["highest_rain_probability"]["date"] == "2026-09-12"
        assert res["highest_expected_rainfall"]["rainfall_mm"] == 18.5
        assert res["highest_expected_rainfall"]["date"] == "2026-09-13"
        assert "85%" in res["highest_rain_probability"]["formatted"]
        assert "18.5 mm" in res["highest_expected_rainfall"]["formatted"]

    def test_compute_forecast_extremes_empty(self):
        res = compute_forecast_extremes([])
        assert res["highest_rain_probability"]["probability"] == 0.0
        assert res["highest_expected_rainfall"]["rainfall_mm"] == 0.0

    def test_detect_future_rainfall_event(self):
        forecast_days = [
            {"date": "Day 0", "precipitation_sum": 0.0},
            {"date": "Day 1", "precipitation_sum": 0.0},
            {"date": "Day 2", "precipitation_sum": 1.0},
            {"date": "Day 3", "precipitation_sum": 2.0},
            {"date": "Day 4", "precipitation_sum": 24.5},
            {"date": "Day 5", "precipitation_sum": 3.0},
        ]
        event = detect_future_rainfall_event(forecast_days)
        assert event is not None
        assert event["is_advance_notice"] is True
        assert event["amount_mm"] == 24.5
        assert event["date"] == "Day 4"
        assert "Heavy rainfall is expected later this week" in event["warning"]

    def test_generate_farm_advisory_summary(self):
        irr_advice = {"action": "DELAY_IRRIGATION", "future_rain_warning": "Heavy rain on Friday"}
        dis_risk = {"risk_level": "ELEVATED"}
        hs_adv = {"heat_advisory": {"severity": "success"}}
        extremes = {
            "highest_rain_probability": {"formatted": "80% on Friday"},
            "highest_expected_rainfall": {"rainfall_mm": 14.0, "date": "Friday"},
        }
        context = {"soil_moisture": 52.0}

        summary = generate_farm_advisory_summary(irr_advice, dis_risk, hs_adv, extremes, context)
        assert "Review irrigation timing" in summary["primary_action"]
        assert "Heavy rain on Friday" in summary["watch_condition"]
        assert "above the critical threshold" in summary["field_condition"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGRONOMIC RULES ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIrrigationRules:
    """Verify deterministic irrigation decision tree and priority handling."""

    def test_delay_irrigation_when_heavy_rain_forecast(self):
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 75.0, "precipitation_sum": 12.0},
            {"precipitation_probability_max": 40.0, "precipitation_sum": 2.0},
            {"precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 55.0, "soil_type": "Loamy", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "DELAY_IRRIGATION"
        assert res["severity"] == "warning"
        assert "Rain is likely" in res["headline"]
        assert "DELAY IRRIGATION" in res["badge_text"]
        assert "why_recommendation" in res
        assert len(res["why_recommendation"]["inputs_evaluated"]) >= 3

    def test_distinguishes_high_prob_with_light_rain_amount(self):
        # 63% probability but only 0.2 mm expected rain, soil moisture 58%
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 63.0, "precipitation_sum": 0.2},
            {"precipitation_probability_max": 30.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 58.0, "soil_type": "Loamy", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "DELAY_IRRIGATION"
        assert "63%" in res["reason"]
        assert "0.2 mm" in res["reason"]
        assert "58.0%" in res["reason"]
        assert "Likely shower with adequate soil moisture" in res["why_recommendation"]["rule_triggered"]

    def test_critical_moisture_prioritizes_irrigation_over_light_rain(self):
        # Soil moisture is critically low (20%), rain prob is 65% but expected rain is only 0.5 mm
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 65.0, "precipitation_sum": 0.5},
            {"precipitation_probability_max": 20.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 20.0, "soil_type": "Loamy", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        # Should NOT blindly delay when moisture is critical and rain is just a light shower!
        assert res["action"] == "CONSIDER_IRRIGATION"
        assert res["severity"] == "primary"
        assert "critical deficit" in res["headline"].lower()
        assert "light shower" in res["reason"].lower()

    def test_critical_moisture_with_imminent_heavy_rain(self):
        # Soil moisture is 20%, but imminent heavy rain (15 mm) is expected
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 90.0, "precipitation_sum": 15.0},
            {"precipitation_probability_max": 50.0, "precipitation_sum": 3.0},
        ]
        context = {"soil_moisture": 20.0, "soil_type": "Loamy", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "DELAY_IRRIGATION"
        assert "imminent" in res["headline"].lower()

    def test_future_rainfall_does_not_override_immediate_action(self):
        # Near term (days 0-2) is dry, day 4 has 20 mm rain. Soil moisture is 32% (depleted)
        current = {"precipitation": 0.0}
        forecast = [
            {"date": "Day 0", "precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
            {"date": "Day 1", "precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
            {"date": "Day 2", "precipitation_probability_max": 15.0, "precipitation_sum": 0.0},
            {"date": "Day 3", "precipitation_probability_max": 20.0, "precipitation_sum": 1.0},
            {"date": "Day 4", "precipitation_sum": 20.0, "precipitation_probability_max": 80.0},
        ]
        context = {"soil_moisture": 32.0, "soil_type": "Clay", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        # Immediate action must still be CONSIDER_IRRIGATION because near term is dry and soil is depleted
        assert res["action"] == "CONSIDER_IRRIGATION"
        # Advance warning is attached without overriding the immediate recommendation
        assert res["future_rain_warning"] is not None
        assert "Heavy rainfall is expected later this week" in res["future_rain_warning"]

    def test_consider_irrigation_when_depleted_and_dry(self):
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 15.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 5.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 28.0, "soil_type": "Clay", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "CONSIDER_IRRIGATION"
        assert res["severity"] == "primary"
        assert "depleted" in res["headline"].lower()

    def test_monitor_and_reassess_on_moderate_rain(self):
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 45.0, "precipitation_sum": 3.0},
            {"precipitation_probability_max": 35.0, "precipitation_sum": 1.0},
            {"precipitation_probability_max": 20.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 50.0, "soil_type": "Loamy", "irrigation_method": "Sprinkler"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "MONITOR_AND_REASSESS"
        assert res["severity"] == "info"

    def test_maintain_normal_cycle_stable(self):
        current = {"precipitation": 0.0}
        forecast = [
            {"precipitation_probability_max": 10.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 5.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 0.0, "precipitation_sum": 0.0},
        ]
        context = {"soil_moisture": 60.0, "soil_type": "Loamy", "irrigation_method": "Drip"}

        res = evaluate_irrigation_rule(current, forecast, context)
        assert res["action"] == "MAINTAIN_NORMAL_CYCLE"
        assert res["severity"] == "success"


class TestDiseaseWeatherRiskRules:
    """Verify disease weather risk and strict non-diagnosis policy."""

    def test_elevated_risk_under_high_humidity_and_rain(self):
        current = {"relative_humidity": 82.0, "temperature": 26.0, "precipitation": 1.0}
        forecast = [{"precipitation_probability_max": 70.0, "precipitation_sum": 8.0}]
        context = {"crop": "Tomato"}

        res = evaluate_disease_weather_risk(current, forecast, context)
        assert res["risk_level"] == "ELEVATED"
        assert res["severity"] == "warning"
        assert "not a plant diagnosis" in res["disclaimer"].lower()
        assert "Tomato" in res["monitoring_guidance"]
        assert "Computer Vision" in res["monitoring_guidance"]
        assert "why_recommendation" in res

    def test_moderate_risk_moderate_humidity(self):
        current = {"relative_humidity": 65.0, "temperature": 24.0, "precipitation": 0.0}
        forecast = [{"precipitation_probability_max": 20.0, "precipitation_sum": 0.0}]
        context = {"crop": "Wheat"}

        res = evaluate_disease_weather_risk(current, forecast, context)
        assert res["risk_level"] == "MODERATE"
        assert res["severity"] == "info"

    def test_low_risk_dry_conditions(self):
        current = {"relative_humidity": 45.0, "temperature": 28.0, "precipitation": 0.0}
        forecast = [{"precipitation_probability_max": 10.0, "precipitation_sum": 0.0}]
        context = {"crop": "Cotton"}

        res = evaluate_disease_weather_risk(current, forecast, context)
        assert res["risk_level"] == "LOW"
        assert res["severity"] == "success"


class TestHeatAndSprayAdvisoryRules:
    """Verify spray drift, active rain, product label disclaimer, and heat stress."""

    def test_high_wind_spray_warning(self):
        current = {"temperature": 27.0, "wind_speed": 24.5, "precipitation": 0.0}
        forecast = []
        context = {"growth_stage": "Flowering", "crop": "Cotton"}

        res = evaluate_heat_and_spray_advisory(current, forecast, context)
        assert "UNFAVORABLE" in res["spray_advisory"]["status"]
        assert res["spray_advisory"]["severity"] == "warning"
        assert "drift" in res["spray_advisory"]["description"].lower()
        assert "pesticide" in res["spray_advisory"]["disclaimer"].lower()

    def test_active_rain_spray_warning(self):
        current = {"temperature": 24.0, "wind_speed": 8.0, "precipitation": 2.0}
        forecast = []
        context = {"growth_stage": "Vegetative", "crop": "Tomato"}

        res = evaluate_heat_and_spray_advisory(current, forecast, context)
        assert "UNFAVORABLE — ACTIVE RAIN" in res["spray_advisory"]["status"]
        assert res["spray_advisory"]["severity"] == "warning"

    def test_favorable_spray_window(self):
        current = {"temperature": 26.0, "wind_speed": 8.0, "precipitation": 0.0}
        forecast = []
        context = {"growth_stage": "Vegetative", "crop": "Tomato"}

        res = evaluate_heat_and_spray_advisory(current, forecast, context)
        assert res["spray_advisory"]["status"] == "FAVORABLE SPRAY WINDOW"
        assert res["spray_advisory"]["severity"] == "success"

    def test_heat_stress_warning(self):
        current = {"temperature": 37.5, "wind_speed": 10.0, "precipitation": 0.0}
        forecast = []
        context = {"growth_stage": "Flowering", "crop": "Tomato"}

        res = evaluate_heat_and_spray_advisory(current, forecast, context)
        assert res["heat_advisory"]["status"] == "HEAT STRESS WARNING"
        assert res["heat_advisory"]["severity"] == "warning"
        assert "Flowering" in res["heat_advisory"]["description"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEATHER SERVICE, CACHING & FORCE-REFRESH TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherIntelligenceService:
    """Test service caching, force-refresh, demo fallback, coordinates, and legacy stubs."""

    def test_legacy_stubs_raise_not_implemented(self):
        svc = WeatherIntelligenceService()
        with pytest.raises(NotImplementedError):
            svc.get_current_conditions(28.61, 77.20)
        with pytest.raises(NotImplementedError):
            svc.get_spray_advisory(28.61, 77.20)

    def test_demo_mode_generates_mock_analysis(self):
        svc = WeatherIntelligenceService(mode="demo")
        res = svc.analyze("Ahmedabad, Gujarat")

        assert res["mode"] == "DEMO"
        assert res["is_mock"] is True
        assert res["error_reason"] is not None
        assert "DEMO WEATHER" in res["disclaimer"]
        assert len(res["forecast"]) == 7
        assert "forecast_extremes" in res
        assert "farm_advisory_summary" in res
        assert "coordinates" in res
        assert "retrieved_at" in res
        assert "irrigation_advice" in res
        assert "disease_weather_risk" in res
        assert "heat_and_spray_advisory" in res
        assert len(res["primary_actions"]) == 3

    def test_live_fallback_on_invalid_location(self):
        svc = WeatherIntelligenceService(mode="auto")
        # Query that triggers geocoding resolution failure or timeout
        res = svc.analyze("XzyNonExistentPlace999888777")

        assert res["mode"] == "DEMO"
        assert res["is_mock"] is True
        assert res["error_reason"] is not None
        assert "Demo Fallback" in res["location"]

    def test_deterministic_fallback_on_mocked_empty_geocoding(self):
        svc = WeatherIntelligenceService(mode="auto")
        # Mock empty geocoding response
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"results": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = svc.analyze("FakeLocationWithoutCoords")
            assert res["mode"] == "DEMO"
            assert res["is_mock"] is True
            assert "Could not resolve" in res["error_reason"]
            assert "Demo Fallback" in res["location"]

    def test_in_memory_ttl_caching(self):
        svc = WeatherIntelligenceService()
        lat, lon = 23.0225, 72.5714
        cache_key = f"{lat:.3f},{lon:.3f}"
        dummy_data = {
            "current": {"temperature": 25.0, "relative_humidity": 60.0, "apparent_temperature": 26.0, "precipitation": 0.0, "rain": 0.0, "wind_speed": 10.0, "weather_code": 0, "condition": "Clear sky", "icon": "☀️", "timestamp": "2026-09-10T12:00"},
            "forecast": [],
        }
        svc._cache[cache_key] = (1e12, dummy_data)  # future timestamp

        cur, fcast = svc.fetch_weather(lat, lon, force_refresh=False)
        assert cur["temperature"] == 25.0

    def test_force_refresh_bypasses_cache(self):
        svc = WeatherIntelligenceService()
        lat, lon = 23.0225, 72.5714
        cache_key = f"{lat:.3f},{lon:.3f}"
        old_data = {
            "current": {"temperature": 18.0, "relative_humidity": 50.0, "apparent_temperature": 18.0, "precipitation": 0.0, "rain": 0.0, "wind_speed": 5.0, "weather_code": 0, "condition": "Clear sky", "icon": "☀️", "timestamp": "2026-09-10T08:00"},
            "forecast": [],
        }
        svc._cache[cache_key] = (1e12, old_data)

        # Mock fresh API payload
        fresh_payload = {
            "current": {
                "time": "2026-09-10T12:00",
                "temperature_2m": 31.5,
                "relative_humidity_2m": 62.0,
                "apparent_temperature": 34.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 12.0,
            },
            "daily": {
                "time": ["2026-09-10"],
                "weather_code": [1],
                "temperature_2m_max": [33.0],
                "temperature_2m_min": [24.0],
                "precipitation_sum": [0.0],
                "precipitation_probability_max": [10.0],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(fresh_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            cur, fcast = svc.fetch_weather(lat, lon, force_refresh=True)
            assert cur["temperature"] == 31.5


# ─────────────────────────────────────────────────────────────────────────────
# 5. FLASK ROUTE INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherRoutes:
    """Verify HTTP GET and POST endpoints for Weather Intelligence."""

    @pytest.fixture()
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_get_weather_page(self, client):
        resp = client.get("/weather")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Weather-Based Agricultural Intelligence" in html
        assert "Open-Meteo" in html
        assert "TODAY'S FARM ADVISORY" in html
        assert "Refresh Weather" in html
        assert "WHY THIS RECOMMENDATION?" in html

    def test_get_weather_intelligence_alias(self, client):
        resp = client.get("/weather-intelligence")
        assert resp.status_code == 200

    def test_post_analyze_weather_json(self, client):
        payload = {
            "location": "Ahmedabad, Gujarat",
            "crop": "Tomato",
            "growth_stage": "Vegetative",
            "soil_moisture": 40.0,
            "soil_type": "Loamy",
            "irrigation_method": "Drip",
        }
        resp = client.post("/analyze-weather", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["status"] == "success"
        assert data["mode"] in ("LIVE", "DEMO")
        assert "current_weather" in data
        assert "forecast" in data
        assert "forecast_extremes" in data
        assert "farm_advisory_summary" in data
        assert "coordinates" in data
        assert "retrieved_at" in data
        assert "irrigation_advice" in data
        assert "why_recommendation" in data["irrigation_advice"]
        assert "disease_weather_risk" in data
        assert "heat_and_spray_advisory" in data

    def test_post_analyze_weather_with_force_refresh(self, client):
        payload = {
            "location": "Ahmedabad, Gujarat",
            "soil_moisture": 35.0,
            "force_refresh": True,
        }
        resp = client.post("/analyze-weather", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["retrieved_at"] is not None

    def test_post_analyze_weather_form_encoded(self, client):
        form_data = {
            "location": "Pune, Maharashtra",
            "crop": "Wheat",
            "growth_stage": "Flowering",
            "soil_moisture": "25.0",
            "soil_type": "Clay",
            "irrigation_method": "Drip",
        }
        resp = client.post("/analyze-weather", data=form_data)
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Weather-Based Agricultural Intelligence" in html
        assert "TODAY'S FARM ADVISORY" in html
