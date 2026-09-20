"""
Unit and Integration Tests for Phase 5: Farm Sustainability Score
================================================================
Covers:
- Deterministic heuristic configuration and weights
- Component scoring rules (Water Efficiency, Resource Use, Crop Health)
- Dynamic missing-data weight renormalization
- Input validation and boundary safety
- Zero-IoT adherence (pure farmer observation)
- Zero hidden defaults (location optional, no fallback to Pune)
- Web routes (GET/POST, HTML/JSON, aliases, errors)
- Legacy compute_farm_score stub preservation
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.main import create_app
from model.sustainability.config import SustainabilityConfig
from model.sustainability.rules import (
    evaluate_water_efficiency,
    evaluate_resource_use,
    evaluate_crop_health,
    generate_sustainability_recommendations,
)
from model.sustainability.analyzer import analyze_sustainability_score
from app.services.sustainability_service import SustainabilityService


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ==========================================================================
# 1. Configuration and Constants Tests
# ==========================================================================

def test_sustainability_config_weights():
    """Verify base component weights sum to 1.0."""
    total = (
        SustainabilityConfig.WEIGHT_WATER_EFFICIENCY
        + SustainabilityConfig.WEIGHT_RESOURCE_USE
        + SustainabilityConfig.WEIGHT_CROP_HEALTH
    )
    assert abs(total - 1.0) < 1e-6
    assert SustainabilityConfig.WEIGHT_WATER_EFFICIENCY == 0.40
    assert SustainabilityConfig.WEIGHT_RESOURCE_USE == 0.30
    assert SustainabilityConfig.WEIGHT_CROP_HEALTH == 0.30


def test_sustainability_config_categories():
    """Verify scoring tier metadata thresholds."""
    strong = SustainabilityConfig.get_category_meta(85)
    assert strong["category"] == "Strong"
    assert strong["severity"] == "success"

    moderate = SustainabilityConfig.get_category_meta(65)
    assert moderate["category"] == "Moderate"
    assert moderate["severity"] == "info"

    needs_imp = SustainabilityConfig.get_category_meta(45)
    assert needs_imp["category"] == "Needs Improvement"
    assert needs_imp["severity"] == "warning"

    low = SustainabilityConfig.get_category_meta(20)
    assert low["category"] == "Low"
    assert low["severity"] == "danger"


# ==========================================================================
# 2. Water Efficiency Component Tests
# ==========================================================================

def test_water_efficiency_drip_optimal():
    """Drip irrigation with optimal moisture (35-65%) yields strong score."""
    eval_res = evaluate_water_efficiency(
        soil_moisture=50.0,
        irrigation_method="Drip",
        water_availability="Moderate",
        weather_summary=None,
    )
    assert eval_res["is_assessed"] is True
    assert eval_res["status"] == "EXCELLENT"
    # Drip base: 90, optimal moisture bonus: +5 -> 95
    assert eval_res["score"] == 95
    assert eval_res["base_weight"] == 0.40


def test_water_efficiency_flood_critical_moisture():
    """Flood irrigation with critical depletion and drought scarcity."""
    eval_res = evaluate_water_efficiency(
        soil_moisture=20.0,
        irrigation_method="Flood",
        water_availability="Scarce",
        weather_summary=None,
    )
    # Flood base: 50, critical moisture under scarce water: -20 -> 30
    assert eval_res["score"] == 30
    assert any("Flood" in item["detail"] for item in eval_res["breakdown"])
    assert any("Water Stress" in item["factor"] for item in eval_res["breakdown"])


def test_water_efficiency_rain_opportunity():
    """Rain forecast combined with flood vs drip systems."""
    rain_weather = {"has_rain_forecast": True, "max_rain_prob": 75.0, "expected_rain_mm": 12.0}

    # Drip with rain forecast gets conservation awareness bonus (+5)
    drip_eval = evaluate_water_efficiency(
        soil_moisture=50.0,
        irrigation_method="Drip",
        water_availability="Moderate",
        weather_summary=rain_weather,
    )
    # Base 90 + moisture bonus 5 + rain bonus 5 = 100
    assert drip_eval["score"] == 100

    # Flood with rain forecast on saturated soil (>70%) gets runoff/loss penalty (-10)
    flood_eval = evaluate_water_efficiency(
        soil_moisture=75.0,
        irrigation_method="Flood",
        water_availability="Moderate",
        weather_summary=rain_weather,
    )
    # Base 50 - rain penalty 10 = 40
    assert flood_eval["score"] == 40


def test_water_efficiency_clamping():
    """Verify water efficiency is strictly bounded to [0, 100]."""
    low_eval = evaluate_water_efficiency(
        soil_moisture=10.0,
        irrigation_method="Flood",
        water_availability="Scarce",
        weather_summary={"has_rain_forecast": True, "max_rain_prob": 80.0, "expected_rain_mm": 15.0},
    )
    assert 0 <= low_eval["score"] <= 100


# ==========================================================================
# 3. Resource Use Component Tests
# ==========================================================================

def test_resource_use_organic_cover_crops():
    """Organic inputs (95) + Cover crops (+15) = 100 (clamped)."""
    res = evaluate_resource_use(
        nutrient_practice="Organic",
        soil_cover="Cover_Crops",
    )
    assert res["is_assessed"] is True
    assert res["status"] == "EXCELLENT"
    assert res["score"] == 100
    assert res["base_weight"] == 0.30


def test_resource_use_integrated_mulch():
    """Integrated (85) + Mulch (+10) = 95."""
    res = evaluate_resource_use(
        nutrient_practice="Integrated",
        soil_cover="Mulch",
    )
    assert res["score"] == 95


def test_resource_use_synthetic_bare():
    """Synthetic heavy (45) + Bare soil (0) = 45."""
    res = evaluate_resource_use(
        nutrient_practice="Synthetic_Heavy",
        soil_cover="Bare_Soil",
    )
    assert res["score"] == 45


# ==========================================================================
# 4. Crop Health Component Tests
# ==========================================================================

def test_crop_health_assessed_healthy():
    """Healthy foliage gives 95."""
    health = evaluate_crop_health("Healthy")
    assert health["is_assessed"] is True
    assert health["status"] == "EXCELLENT"
    assert health["score"] == 95
    assert health["base_weight"] == 0.30


def test_crop_health_assessed_severe():
    """Severe disease gives 25."""
    health = evaluate_crop_health("Severe_Disease")
    assert health["is_assessed"] is True
    assert health["status"] == "POOR"
    assert health["score"] == 25


def test_crop_health_not_assessed():
    """Not_Assessed returns score None and status NOT_ASSESSED."""
    health = evaluate_crop_health("Not_Assessed")
    assert health["is_assessed"] is False
    assert health["status"] == "NOT_ASSESSED"
    assert health["score"] is None
    assert health["base_weight"] == 0.30


# ==========================================================================
# 5. Dynamic Weight Renormalization Tests
# ==========================================================================

def test_renormalization_when_crop_health_missing():
    """When Crop Health is omitted, weights renormalize across Water (40%) and Resource (30%)."""
    result = analyze_sustainability_score(
        soil_moisture=50.0,
        irrigation_method="Drip",             # score 95
        water_availability="Moderate",
        nutrient_practice="Organic",           # score 100
        soil_cover="Cover_Crops",
        crop_health_status="Not_Assessed",     # excluded
        crop="Tomato",
        growth_stage="Vegetative",
        weather_summary=None,
        weather_status="UNAVAILABLE",
    )

    # Water = 95, Resource = 100, Crop Health = Not Assessed
    # Available weight sum = 0.40 + 0.30 = 0.70
    # Weighted sum = (95 * 0.40) + (100 * 0.30) = 38.0 + 30.0 = 68.0
    # Final renormalized score = 68.0 / 0.70 = 97.14 -> 97
    assert result["score"] == 97
    assert result["weights"]["is_renormalized"] is True
    assert result["data_status"] == "PARTIAL_RENORMALIZED"
    assert result["components"]["crop_health"]["is_assessed"] is False
    assert result["components"]["crop_health"]["effective_weight"] == 0.0

    # Effective weight percentages: Water = 0.40/0.70 = ~57.14%, Resource = 0.30/0.70 = ~42.86%
    assert round(result["components"]["water_efficiency"]["effective_weight_pct"], 1) == 57.1
    assert round(result["components"]["resource_use"]["effective_weight_pct"], 1) == 42.9

    # Calculation string includes step-by-step division by 70%
    assert "/ 70%" in result["calculation"]
    assert "→ 97 / 100" in result["calculation"]


def test_standard_3_component_score_when_all_assessed():
    """When all 3 components are assessed, base weights 40/30/30 are used directly."""
    result = analyze_sustainability_score(
        soil_moisture=50.0,
        irrigation_method="Drip",             # score 95 * 0.40 = 38
        water_availability="Moderate",
        nutrient_practice="Organic",           # score 100 * 0.30 = 30
        soil_cover="Cover_Crops",
        crop_health_status="Healthy",          # score 95 * 0.30 = 28.5
        crop="Tomato",
        growth_stage="Vegetative",
        weather_summary=None,
        weather_status="UNAVAILABLE",
    )
    # Total = 38 + 30 + 28.5 = 96.5 -> 96 (Python half-to-even rounding)
    assert result["score"] == 96
    assert result["weights"]["is_renormalized"] is False
    assert result["data_status"] == "COMPLETE"
    assert result["components"]["crop_health"]["is_assessed"] is True
    assert result["components"]["crop_health"]["effective_weight_pct"] == 30.0


# ==========================================================================
# 6. Service Validation and Zero Hidden Default Tests
# ==========================================================================

def test_sustainability_service_invalid_moisture():
    service = SustainabilityService()
    with pytest.raises(ValueError, match="between 0% and 100%"):
        service.calculate_score(soil_moisture=-5.0, irrigation_method="Drip")

    with pytest.raises(ValueError, match="between 0% and 100%"):
        service.calculate_score(soil_moisture=105.0, irrigation_method="Drip")

    with pytest.raises(ValueError, match="valid number"):
        service.calculate_score(soil_moisture="invalid", irrigation_method="Drip")


def test_sustainability_service_invalid_enums():
    service = SustainabilityService()
    with pytest.raises(ValueError, match="Invalid irrigation method"):
        service.calculate_score(soil_moisture=40.0, irrigation_method="Subsurface_Trickle")

    with pytest.raises(ValueError, match="Invalid water availability"):
        service.calculate_score(
            soil_moisture=40.0,
            irrigation_method="Drip",
            water_availability="Unlimited",
        )

    with pytest.raises(ValueError, match="Invalid nutrient practice"):
        service.calculate_score(
            soil_moisture=40.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Nanotechnology",
        )


def test_sustainability_service_zero_default_location():
    """Verify that omitting location does NOT query Pune or fabricate weather data."""
    mock_weather = MagicMock()
    service = SustainabilityService(weather_service=mock_weather)

    result = service.calculate_score(
        soil_moisture=45.0,
        irrigation_method="Drip",
        location="",  # empty string
    )
    # Mock weather should NOT have been called!
    mock_weather.analyze.assert_not_called()
    assert result["weather_status"] == "UNAVAILABLE"

    # None location also does not call weather
    result_none = service.calculate_score(
        soil_moisture=45.0,
        irrigation_method="Drip",
        location=None,
    )
    mock_weather.analyze.assert_not_called()
    assert result_none["weather_status"] == "UNAVAILABLE"


def test_sustainability_service_with_valid_location():
    """When location is provided, weather service is queried."""
    mock_weather = MagicMock()
    mock_weather.analyze.return_value = {
        "is_mock": False,
        "location": "Nagpur, Maharashtra",
        "forecast": [
            {"precipitation_probability_max": 20.0, "precipitation_sum": 0.0},
            {"precipitation_probability_max": 65.0, "precipitation_sum": 8.0},
        ],
    }
    service = SustainabilityService(weather_service=mock_weather)

    result = service.calculate_score(
        soil_moisture=45.0,
        irrigation_method="Drip",
        location="Nagpur, Maharashtra",
    )
    mock_weather.analyze.assert_called_once_with(
        location="Nagpur, Maharashtra",
        force_refresh=False,
    )
    assert result["weather_status"] == "LIVE"


def test_legacy_compute_farm_score_preservation():
    """Verify the legacy boundary stub compute_farm_score continues to raise NotImplementedError."""
    service = SustainabilityService()
    with pytest.raises(NotImplementedError):
        service.compute_farm_score(50.0, 1000.0, True)


# ==========================================================================
# 7. Actionable Recommendations Generation Tests
# ==========================================================================

def test_recommendations_generation_rules():
    water_low = {"score": 30, "inputs": {"irrigation_method": "Flood", "soil_moisture_pct": 15.0}}
    resource_low = {"score": 35, "inputs": {"nutrient_practice": "Synthetic_Heavy", "soil_cover": "Bare_Soil"}}
    health_low = {"score": 25, "is_assessed": True}

    recs = generate_sustainability_recommendations(water_low, resource_low, health_low)
    titles = [r["title"] for r in recs]

    assert any("Micro-Irrigation" in t for t in titles)
    assert any("Organic or Integrated" in t for t in titles)
    assert any("Protective Soil Cover" in t for t in titles)
    assert any("Foliar Disease Management" in t for t in titles)
    assert all(r["priority"] in ("Critical", "High", "Medium", "Low") for r in recs)


# ==========================================================================
# 8. Web Route Tests
# ==========================================================================

def test_get_sustainability_page(client):
    """GET /sustainability returns 200 and renders template with active badge."""
    resp = client.get("/sustainability")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Farm Sustainability Score" in html
    assert "Sustainability Score: ACTIVE" in html


def test_get_sustainability_alias(client):
    """GET /sustainability-score alias returns 200."""
    resp = client.get("/sustainability-score")
    assert resp.status_code == 200


def test_post_sustainability_json(client):
    """POST /calculate-sustainability with JSON payload returns 200 and score result."""
    payload = {
        "soil_moisture": 45,
        "irrigation_method": "Drip",
        "water_availability": "Moderate",
        "nutrient_practice": "Integrated",
        "soil_cover": "Mulch",
        "crop_health_status": "Not_Assessed",
        "crop": "Tomato",
        "growth_stage": "Vegetative",
    }
    resp = client.post("/calculate-sustainability", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert 0 <= data["score"] <= 100
    assert data["weights"]["is_renormalized"] is True


def test_post_sustainability_form(client):
    """POST /calculate-sustainability via HTML form submission returns 200."""
    form_data = {
        "soil_moisture": "50",
        "irrigation_method": "Sprinkler",
        "water_availability": "Moderate",
        "nutrient_practice": "Organic",
        "soil_cover": "Cover_Crops",
        "crop_health_status": "Healthy",
        "crop": "Potato",
        "growth_stage": "Flowering",
    }
    resp = client.post("/calculate-sustainability", data=form_data)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "TRANSPARENT ARITHMETIC CALCULATION" in html


def test_post_sustainability_validation_errors(client):
    """POST with missing soil moisture or missing irrigation method returns 400."""
    # Missing soil moisture
    resp1 = client.post(
        "/calculate-sustainability",
        json={"irrigation_method": "Drip"},
    )
    assert resp1.status_code == 400
    assert "soil moisture" in resp1.get_json()["message"]

    # Missing irrigation method
    resp2 = client.post(
        "/calculate-sustainability",
        json={"soil_moisture": 45},
    )
    assert resp2.status_code == 400
    assert "irrigation system" in resp2.get_json()["message"]


# ==========================================================================
# 9. Strict SIH Constraints Verification
# ==========================================================================

def test_zero_fake_ml_and_zero_iot_claims():
    """Verify that results do not contain fake ML confidences or unmeasured carbon/liter figures."""
    result = analyze_sustainability_score(
        soil_moisture=45.0,
        irrigation_method="Drip",
        water_availability="Moderate",
        nutrient_practice="Organic",
        soil_cover="Cover_Crops",
        crop_health_status="Not_Assessed",
        crop="Tomato",
        growth_stage="Vegetative",
        weather_summary=None,
        weather_status="UNAVAILABLE",
    )

    result_str = str(result).lower()
    # Must NOT claim exact fake liters or percentages saved
    assert "liters saved" not in result_str
    assert "carbon credits" not in result_str
    assert "ml confidence" not in result_str
    # Must explicitly state it is an advisory index
    assert "advisory" in result["disclaimer"].lower()


# ==========================================================================
# 10. Sensitivity & Input Impact Tests (Anti-Regression Suite)
# ==========================================================================

class TestSustainabilitySensitivityAndEdgeCases:
    """Verifies that all farmer inputs genuinely and deterministically affect scoring."""

    def test_sensitivity_two_distinct_scenarios(self):
        """Scenario A (poor/stress) vs Scenario B (sustainable) produces distinct components and final scores."""
        service = SustainabilityService()

        # INPUT A: Poor practices & disease
        res_a = service.calculate_score(
            soil_moisture=20.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            nutrient_practice="Intensive_Chemical",
            soil_cover="None",
            crop_health_status="Disease_Detected",
        )

        # INPUT B: Precision stewardship & health
        res_b = service.calculate_score(
            soil_moisture=70.0,
            irrigation_method="Drip",
            water_availability="Abundant",
            nutrient_practice="Organic",
            soil_cover="Mulched",
            crop_health_status="Healthy",
        )

        # All 3 components and final score must differ
        assert res_a["components"]["water_efficiency"]["score"] != res_b["components"]["water_efficiency"]["score"]
        assert res_a["components"]["resource_use"]["score"] != res_b["components"]["resource_use"]["score"]
        assert res_a["components"]["crop_health"]["score"] != res_b["components"]["crop_health"]["score"]
        assert res_a["score"] != res_b["score"]

        # Water: 30 vs 90
        assert res_a["components"]["water_efficiency"]["score"] == 30
        assert res_b["components"]["water_efficiency"]["score"] == 90

        # Resource: 45 vs 100
        assert res_a["components"]["resource_use"]["score"] == 45
        assert res_b["components"]["resource_use"]["score"] == 100

        # Crop Health: 45 vs 95
        assert res_a["components"]["crop_health"]["score"] == 45
        assert res_b["components"]["crop_health"]["score"] == 95

        # Final Scores: 39 vs 95
        assert res_a["score"] < 50
        assert res_b["score"] >= 90

    def test_extreme_scenarios_poor_vs_strong(self):
        """Extreme poor practices (Flood, 90%, Scarce, Chemical, Bare, Severe) vs Strong practices."""
        service = SustainabilityService()

        poor = service.calculate_score(
            soil_moisture=90.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            nutrient_practice="Intensive_Chemical",
            soil_cover="Bare_Soil",
            crop_health_status="Severe_Damage",
        )

        strong = service.calculate_score(
            soil_moisture=55.0,
            irrigation_method="Drip",
            water_availability="Abundant",
            nutrient_practice="Organic",
            soil_cover="Cover_Crops",
            crop_health_status="Healthy",
        )

        # Poor should be in Low tier (< 40)
        assert poor["score"] <= 35
        assert poor["category"] == "Low"

        # Strong should be in Strong tier (>= 80)
        assert strong["score"] >= 90
        assert strong["category"] == "Strong"

        # Substantial difference (> 50 points)
        assert (strong["score"] - poor["score"]) >= 55

    def test_sensitivity_irrigation_method_alone_changes_water_score(self):
        """Holding all inputs constant, changing irrigation method changes Water Efficiency."""
        service = SustainabilityService()
        drip = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip")
        sprinkler = service.calculate_score(soil_moisture=50.0, irrigation_method="Sprinkler")
        flood = service.calculate_score(soil_moisture=50.0, irrigation_method="Flood")

        score_drip = drip["components"]["water_efficiency"]["score"]
        score_sprinkler = sprinkler["components"]["water_efficiency"]["score"]
        score_flood = flood["components"]["water_efficiency"]["score"]

        assert score_drip > score_sprinkler > score_flood
        assert score_drip == 95
        assert score_sprinkler == 75
        assert score_flood == 50

    def test_sensitivity_soil_moisture_alone_changes_water_score(self):
        """Holding all inputs constant, changing soil moisture alters water efficiency."""
        service = SustainabilityService()
        flood_optimal = service.calculate_score(soil_moisture=50.0, irrigation_method="Flood")
        flood_depleted = service.calculate_score(soil_moisture=30.0, irrigation_method="Flood")
        flood_critical = service.calculate_score(soil_moisture=15.0, irrigation_method="Flood")
        flood_waterlogged = service.calculate_score(soil_moisture=90.0, irrigation_method="Flood")

        s_optimal = flood_optimal["components"]["water_efficiency"]["score"]
        s_depleted = flood_depleted["components"]["water_efficiency"]["score"]
        s_critical = flood_critical["components"]["water_efficiency"]["score"]
        s_waterlogged = flood_waterlogged["components"]["water_efficiency"]["score"]

        assert s_optimal == 50
        assert s_depleted == 45
        assert s_critical == 40
        assert s_waterlogged == 35

    def test_sensitivity_water_availability_alone_changes_water_score(self):
        """Changing water availability from Moderate to Scarce adjusts stewardship scores."""
        service = SustainabilityService()
        drip_mod = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", water_availability="Moderate")
        drip_scarce = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", water_availability="Scarce")

        # Drip gets +5 stewardship bonus in water-scarce regions
        assert drip_scarce["components"]["water_efficiency"]["score"] > drip_mod["components"]["water_efficiency"]["score"]
        assert drip_scarce["components"]["water_efficiency"]["score"] == 100

        sprinkler_mod = service.calculate_score(soil_moisture=50.0, irrigation_method="Sprinkler", water_availability="Moderate")
        sprinkler_scarce = service.calculate_score(soil_moisture=50.0, irrigation_method="Sprinkler", water_availability="Scarce")
        # Sprinkler gets -5 penalty for evaporative losses in water-scarce regions
        assert sprinkler_scarce["components"]["water_efficiency"]["score"] < sprinkler_mod["components"]["water_efficiency"]["score"]

    def test_sensitivity_nutrient_practice_alone_changes_resource_score(self):
        """Changing nutrient practice changes resource use score from Organic (95) down to Chemical (45)."""
        service = SustainabilityService()
        org = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Organic")
        integ = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Integrated")
        chem = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Intensive_Chemical")

        s_org = org["components"]["resource_use"]["score"]
        s_integ = integ["components"]["resource_use"]["score"]
        s_chem = chem["components"]["resource_use"]["score"]

        assert s_org > s_integ > s_chem
        assert s_org == 95  # bare soil
        assert s_integ == 85
        assert s_chem == 45

    def test_sensitivity_soil_cover_alone_changes_resource_score(self):
        """Changing soil cover adjusts resource conservation points."""
        service = SustainabilityService()
        cover = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Integrated", soil_cover="Cover_Crops")
        mulch = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Integrated", soil_cover="Mulch")
        tillage = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Integrated", soil_cover="Minimum_Tillage")
        bare = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", nutrient_practice="Integrated", soil_cover="Bare_Soil")

        assert cover["components"]["resource_use"]["score"] == 100  # 85 + 15
        assert mulch["components"]["resource_use"]["score"] == 95   # 85 + 10
        assert tillage["components"]["resource_use"]["score"] == 90 # 85 + 5
        assert bare["components"]["resource_use"]["score"] == 85    # 85 + 0

    def test_sensitivity_crop_health_alone_changes_score(self):
        """Changing crop health status alters the crop health component score."""
        service = SustainabilityService()
        healthy = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", crop_health_status="Healthy")
        minor = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", crop_health_status="Minor_Stress")
        disease = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", crop_health_status="Disease_Detected")
        severe = service.calculate_score(soil_moisture=50.0, irrigation_method="Drip", crop_health_status="Severe_Damage")

        assert healthy["components"]["crop_health"]["score"] == 95
        assert minor["components"]["crop_health"]["score"] == 75
        assert disease["components"]["crop_health"]["score"] == 45
        assert severe["components"]["crop_health"]["score"] == 25

        # Final score also scales accordingly
        assert healthy["score"] > minor["score"] > disease["score"] > severe["score"]

    def test_sensitivity_not_assessed_to_healthy_changes_denominator(self):
        """Selecting Not_Assessed reallocates 30% weight to Water & Resource; Healthy uses all 3."""
        service = SustainabilityService()
        unassessed = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            crop_health_status="Not_Assessed",
        )
        assessed = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            crop_health_status="Healthy",
        )

        assert unassessed["weights"]["is_renormalized"] is True
        assert unassessed["weights"]["total_evaluated_weight"] == 0.70
        assert unassessed["components"]["crop_health"]["is_assessed"] is False
        assert unassessed["components"]["crop_health"]["score"] is None

        assert assessed["weights"]["is_renormalized"] is False
        assert assessed["weights"]["total_evaluated_weight"] == 1.00
        assert assessed["components"]["crop_health"]["is_assessed"] is True
        assert assessed["components"]["crop_health"]["score"] == 95

    def test_end_to_end_http_sensitivity(self, client):
        """Verify the real HTTP endpoint POST /calculate-sustainability returns distinct JSON."""
        # Request A: Poor practices
        resp_a = client.post(
            "/calculate-sustainability",
            json={
                "soil_moisture": 20,
                "irrigation_method": "Flood",
                "water_availability": "Scarce",
                "nutrient_practice": "Intensive_Chemical",
                "soil_cover": "None",
                "crop_health_status": "Disease_Detected",
            },
        )
        assert resp_a.status_code == 200
        data_a = resp_a.get_json()

        # Request B: Strong practices
        resp_b = client.post(
            "/calculate-sustainability",
            json={
                "soil_moisture": 60,
                "irrigation_method": "Drip",
                "water_availability": "Abundant",
                "nutrient_practice": "Organic",
                "soil_cover": "Mulch",
                "crop_health_status": "Healthy",
            },
        )
        assert resp_b.status_code == 200
        data_b = resp_b.get_json()

        assert data_a["score"] != data_b["score"]
        assert data_a["components"]["water_efficiency"]["score"] != data_b["components"]["water_efficiency"]["score"]
        assert data_a["components"]["resource_use"]["score"] != data_b["components"]["resource_use"]["score"]
        assert data_a["components"]["crop_health"]["score"] != data_b["components"]["crop_health"]["score"]
        assert data_a["score"] < data_b["score"]

    def test_all_listed_crops_accepted_and_contextual_invariance(self):
        """Verify all 18 common crops are accepted and none alter the deterministic score."""
        service = SustainabilityService()
        baseline_res = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            crop="Tomato",
        )
        baseline_score = baseline_res["score"]
        baseline_water = baseline_res["components"]["water_efficiency"]["score"]
        baseline_resource = baseline_res["components"]["resource_use"]["score"]

        for crop_name in SustainabilityConfig.COMMON_CROPS:
            res = service.calculate_score(
                soil_moisture=50.0,
                irrigation_method="Drip",
                crop=crop_name,
            )
            assert res["status"] == "success"
            assert res["farm_inputs"]["crop"] == crop_name
            # Changing crop must NOT alter the sustainability score
            assert res["score"] == baseline_score
            assert res["components"]["water_efficiency"]["score"] == baseline_water
            assert res["components"]["resource_use"]["score"] == baseline_resource

    def test_other_not_listed_crop_accepted(self):
        """Verify 'Other / Not Listed' is accepted and purely contextual."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            crop="Other / Not Listed",
        )
        assert res["farm_inputs"]["crop"] == "Other / Not Listed"
        assert res["status"] == "success"

    def test_changing_crop_via_http_does_not_alter_score(self, client):
        """End-to-end HTTP verification that changing crop retains identical score."""
        base_payload = {
            "soil_moisture": 50,
            "irrigation_method": "Drip",
            "water_availability": "Moderate",
            "nutrient_practice": "Integrated",
            "soil_cover": "Bare_Soil",
            "crop_health_status": "Not_Assessed",
        }

        resp_tomato = client.post("/calculate-sustainability", json={**base_payload, "crop": "Tomato"})
        resp_other = client.post("/calculate-sustainability", json={**base_payload, "crop": "Other / Not Listed"})
        resp_soybean = client.post("/calculate-sustainability", json={**base_payload, "crop": "Soybean"})

        assert resp_tomato.status_code == 200
        assert resp_other.status_code == 200
        assert resp_soybean.status_code == 200

        data_t = resp_tomato.get_json()
        data_o = resp_other.get_json()
        data_s = resp_soybean.get_json()

        assert data_t["score"] == data_o["score"] == data_s["score"]
        assert data_t["farm_inputs"]["crop"] == "Tomato"
        assert data_o["farm_inputs"]["crop"] == "Other / Not Listed"
        assert data_s["farm_inputs"]["crop"] == "Soybean"


class TestPhase51DataDrivenCropContext:
    """Comprehensive test suite for Phase 5.1 data-driven crop requirements & growth stage sensitivity."""

    def test_crop_data_loads_from_csv(self):
        """1. Source-derived crop data loads correctly with complete schema and verified records."""
        from model.sustainability.crop_requirements import load_crop_requirements_dataset
        dataset = load_crop_requirements_dataset()
        assert len(dataset) >= 17
        assert "tomato" in dataset
        assert "rice" in dataset
        assert "wheat" in dataset
        assert "sugarcane" in dataset
        assert "chickpea" in dataset

        req = dataset["tomato"]
        assert req.display_name == "Tomato"
        assert req.scientific_name == "Solanum lycopersicum"
        assert req.seasonal_water_need_mm_min == 400.0
        assert req.seasonal_water_need_mm_max == 600.0
        assert req.agrismart_water_need_category == "Medium"
        assert req.fao_depletion_fraction_p == 0.40
        assert "Flowering" in req.critical_growth_stages
        assert "FAO" in req.source_reference

    def test_fao_p_values_are_depletion_fractions_not_moisture_percentages(self):
        """2. FAO p values are stored strictly as depletion fractions, NOT converted to moisture percentages."""
        from model.sustainability.crop_requirements import load_crop_requirements_dataset
        import csv
        from pathlib import Path
        from model.sustainability.crop_requirements import DATA_CSV_PATH

        # Verify CSV column headers do NOT have manufactured moisture percentages
        with open(DATA_CSV_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "optimal_moisture_min_pct" not in header
            assert "optimal_moisture_max_pct" not in header
            assert "fao_depletion_fraction_p" in header

        dataset = load_crop_requirements_dataset()
        for crop_key, req in dataset.items():
            # Depletion fraction p must be a dimensionless ratio (typically 0.20 - 0.65)
            assert 0.15 <= req.fao_depletion_fraction_p <= 0.70, f"{crop_key} p value out of range: {req.fao_depletion_fraction_p}"
            # Verify no attribute exists manufacturing moisture percentage from p
            assert not hasattr(req, "optimal_moisture_min_pct")
            assert not hasattr(req, "optimal_moisture_max_pct")

    def test_seasonal_mm_values_not_compared_with_instantaneous_moisture_percentages(self):
        """3. Seasonal mm values are contextual information and never compared/divided by moisture %."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=45.0,
            irrigation_method="Drip",
            crop="Sugarcane",
            growth_stage="Vegetative",
        )
        ctx = res["crop_sustainability_context"]
        assert ctx["seasonal_water_need_mm_min"] == 1500.0
        assert ctx["seasonal_water_need_mm_max"] == 2500.0

        # Verify the calculation string does NOT contain dimensional mismatch formulas
        calc_str = res["calculation"]
        assert "1500" not in calc_str
        assert "2500" not in calc_str
        assert "/" not in calc_str or "100%" in calc_str or "0.70" in calc_str

    def test_different_crops_produce_different_water_context_assessments(self):
        """4. Different crops produce different water-context assessments when source data supports it."""
        service = SustainabilityService()
        # High seasonal water need crop (Sugarcane) vs Drought-hardy pulse (Chickpea)
        # under identical Scarce water availability and Flood delivery
        sugarcane_res = service.calculate_score(
            soil_moisture=35.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            crop="Sugarcane",
            growth_stage="Vegetative",
        )
        chickpea_res = service.calculate_score(
            soil_moisture=35.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            crop="Chickpea",
            growth_stage="Vegetative",
        )

        sugarcane_water = sugarcane_res["components"]["water_efficiency"]["score"]
        chickpea_water = chickpea_res["components"]["water_efficiency"]["score"]

        # Sugarcane receives -10 penalty for Very_High water need in Scarce water
        # Chickpea receives +5 bonus for Low water need (drought resilience) in Scarce water
        assert sugarcane_water < chickpea_water
        assert chickpea_water - sugarcane_water == 15

        # Check breakdown labels
        sugarcane_breakdown = sugarcane_res["components"]["water_efficiency"]["breakdown"]
        chickpea_breakdown = chickpea_res["components"]["water_efficiency"]["breakdown"]

        assert any("AgriSmart project-defined alignment rule" in b["detail"] for b in sugarcane_breakdown)
        assert any("AgriSmart project-defined alignment rule" in b["detail"] for b in chickpea_breakdown)

    def test_growth_stage_sensitivity_affects_assessment_only_where_supported(self):
        """5. Growth stage affects assessment only where source data supports it."""
        service = SustainabilityService()
        # Tomato with deficient soil moisture (25%) during Flowering (critical stage) vs Maturity (dry-down)
        tomato_flowering = service.calculate_score(
            soil_moisture=25.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            crop="Tomato",
            growth_stage="Flowering",
        )
        tomato_maturity = service.calculate_score(
            soil_moisture=25.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            crop="Tomato",
            growth_stage="Maturity",
        )

        w_flowering = tomato_flowering["components"]["water_efficiency"]["score"]
        w_maturity = tomato_maturity["components"]["water_efficiency"]["score"]

        # Flowering has Ky=1.10 (critical), deficit penalizes 5 pts
        # Maturity is dry-down stage, 0 pts penalty
        assert w_flowering < w_maturity
        assert w_maturity - w_flowering == 5

        # Check indicator in context
        assert tomato_flowering["crop_sustainability_context"]["is_critical_stage_active"] is True
        assert tomato_maturity["crop_sustainability_context"]["is_critical_stage_active"] is False

    def test_unsupported_crop_produces_not_available(self):
        """6. Unsupported crops produce NOT_AVAILABLE with zero arbitrary score adjustments."""
        service = SustainabilityService()
        for unsupported_input in ["Other / Not Listed", "Dragonfruit", "Not Listed", ""]:
            res = service.calculate_score(
                soil_moisture=50.0,
                irrigation_method="Drip",
                water_availability="Moderate",
                crop=unsupported_input,
            )
            ctx = res["crop_sustainability_context"]
            assert ctx["crop_data_status"] == "NOT_AVAILABLE"
            assert ctx["is_available"] is False
            assert ctx["seasonal_water_need_mm"] == "N/A"
            assert ctx["fao_depletion_fraction_p"] is None

            # Calculation succeeds with standard scoring (no crashes, no defaults)
            assert res["status"] == "success"
            assert res["score"] > 0

    def test_no_arbitrary_crop_score_dictionary_exists(self):
        """7. Verify no arbitrary crop-to-score dictionary exists in code."""
        import inspect
        from model.sustainability import rules, config

        rules_source = inspect.getsource(rules)
        config_source = inspect.getsource(config)

        # Ensure no static score assignments per crop
        for forbidden in ['{"Tomato":', "crop_scores =", "CROP_SCORES", "CROP_WEIGHTS"]:
            assert forbidden not in rules_source
            assert forbidden not in config_source

    def test_identical_inputs_remain_strictly_deterministic(self):
        """8. Identical crop and stage inputs remain bit-for-bit deterministic."""
        service = SustainabilityService()
        res1 = service.calculate_score(
            soil_moisture=22.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            crop="Sugarcane",
            growth_stage="Grand_Growth",
        )
        res2 = service.calculate_score(
            soil_moisture=22.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            crop="Sugarcane",
            growth_stage="Grand_Growth",
        )
        assert res1 == res2
        assert res1["score"] == res2["score"]
        assert res1["components"]["water_efficiency"]["score"] == res2["components"]["water_efficiency"]["score"]

    def test_existing_sustainability_score_behavior_intact(self):
        """9. Existing core sustainability behavior (weights, renormalization, recommendations) remains intact."""
        service = SustainabilityService()
        # Missing crop health -> 70% weight renormalized
        res = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            crop_health_status="Not_Assessed",
        )
        assert res["weights"]["is_renormalized"] is True
        assert res["weights"]["total_evaluated_weight"] == 0.70
        assert res["components"]["crop_health"]["is_assessed"] is False

    def test_http_endpoint_returns_crop_sustainability_context(self, client):
        """10. End-to-end HTTP request returns structured crop_sustainability_context payload."""
        resp = client.post(
            "/calculate-sustainability",
            json={
                "soil_moisture": 25,
                "irrigation_method": "Drip",
                "water_availability": "Moderate",
                "nutrient_practice": "Integrated",
                "soil_cover": "Mulch",
                "crop": "Tomato",
                "growth_stage": "Flowering",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "crop_sustainability_context" in data
        ctx = data["crop_sustainability_context"]
        assert ctx["crop_name"] == "Tomato"
        assert ctx["growth_stage"] == "Flowering"
        assert ctx["crop_data_status"] == "AVAILABLE"
        assert ctx["is_available"] is True
        assert ctx["scientific_name"] == "Solanum lycopersicum"
        assert ctx["seasonal_water_need_mm"] == "400 – 600 mm"
        assert ctx["agrismart_water_need_category"] == "Medium"
        assert ctx["fao_depletion_fraction_p"] == 0.40
        assert ctx["is_critical_stage_active"] is True
        assert "FAO" in ctx["source_reference"]
        assert "AgriSmart applies transparent project-defined alignment rules" in ctx["scoring_basis_note"]


class TestPhase51CropLookupRegression:
    """
    Regression tests for the crop dataset lookup fix (Phase 5.1).
    Ensures generic, case-insensitive, CSV-driven crop lookup works for all
    supported crops without hardcoded special cases.
    """

    # ── Test 1: All supported crops resolve as AVAILABLE ──

    @pytest.mark.parametrize("crop_name", [
        "Rice (Paddy)", "Wheat", "Maize (Corn)", "Cotton", "Sugarcane",
        "Tomato", "Potato", "Onion", "Soybean", "Groundnut",
        "Chickpea", "Pigeon Pea", "Mustard", "Sorghum", "Pearl Millet",
        "Banana", "Mango",
    ])
    def test_supported_crop_lookup_available(self, crop_name):
        """Every supported crop resolves to is_available=True from CSV."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement(crop_name)
        assert req.is_available is True, f"{crop_name} should be available"
        assert req.crop_data_status == "AVAILABLE"
        assert req.display_name != ""
        assert req.scientific_name != "N/A"

    # ── Test 2: Case-insensitive lookup ──

    @pytest.mark.parametrize("crop_input,expected_key", [
        ("sugarcane", "sugarcane"),
        ("Sugarcane", "sugarcane"),
        ("SUGARCANE", "sugarcane"),
        ("rice", "rice"),
        ("Rice (Paddy)", "rice"),
        ("RICE (PADDY)", "rice"),
        ("wheat", "wheat"),
        ("WHEAT", "wheat"),
        ("chickpea", "chickpea"),
        ("Chickpea", "chickpea"),
    ])
    def test_case_insensitive_crop_lookup(self, crop_input, expected_key):
        """Crop lookup must be case-insensitive and resolve to the correct canonical key."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement(crop_input)
        assert req.is_available is True, f"{crop_input} should resolve to available crop"
        assert req.crop_key == expected_key

    # ── Test 3: Unknown crops return NOT_AVAILABLE ──

    @pytest.mark.parametrize("crop_name", [
        "Other / Not Listed", "Papaya", "Coconut", "Dragonfruit", "Unknown",
    ])
    def test_unsupported_crop_not_available(self, crop_name):
        """Unsupported/unknown crops must return NOT_AVAILABLE without fake defaults."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement(crop_name)
        assert req.is_available is False
        assert req.crop_data_status == "NOT_AVAILABLE"
        assert req.crop_key == "unsupported"

    # ── Test 4: Valid crop + unsupported growth stage ──

    def test_valid_crop_unsupported_stage_still_available(self):
        """A valid crop must remain AVAILABLE even with an unconventional growth stage."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement("Sugarcane", "Dormancy")
        assert req.is_available is True, "Crop must remain AVAILABLE regardless of stage"
        assert req.crop_data_status == "AVAILABLE"
        assert req.is_stage_critical("Dormancy") is False

    # ── Test 5: End-to-end API returns is_available=true for supported crops ──

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    @pytest.mark.parametrize("crop_name", [
        "Sugarcane", "Rice (Paddy)", "Chickpea", "Maize (Corn)",
    ])
    def test_http_supported_crop_returns_available(self, client, crop_name):
        """HTTP POST with supported crop must return is_available=True in response."""
        resp = client.post("/calculate-sustainability", json={
            "soil_moisture": 45,
            "irrigation_method": "Drip",
            "water_availability": "Moderate",
            "nutrient_practice": "Integrated",
            "soil_cover": "Mulch",
            "crop_health_status": "Healthy",
            "crop": crop_name,
            "growth_stage": "Vegetative",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        ctx = data["crop_sustainability_context"]
        assert ctx["is_available"] is True
        assert ctx["crop_data_status"] == "AVAILABLE"

    # ── Test 6: HTTP with unsupported crop returns NOT_AVAILABLE ──

    def test_http_unsupported_crop_returns_not_available(self, client):
        """HTTP POST with 'Other / Not Listed' must return is_available=False."""
        resp = client.post("/calculate-sustainability", json={
            "soil_moisture": 45,
            "irrigation_method": "Drip",
            "crop": "Other / Not Listed",
            "growth_stage": "Vegetative",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        ctx = data["crop_sustainability_context"]
        assert ctx["is_available"] is False
        assert ctx["crop_data_status"] == "NOT_AVAILABLE"

    # ── Test 7: Resource Use score stability ──

    def test_resource_use_score_unchanged(self):
        """Resource Use scoring must remain identical after UI fixes."""
        result = evaluate_resource_use(
            nutrient_practice="Integrated",
            soil_cover="Mulch",
        )
        assert result["score"] == 95  # Integrated=85 + Mulch=+10

    def test_resource_use_organic_cover_crops(self):
        """Resource Use: Organic + Cover Crops = 100."""
        result = evaluate_resource_use(
            nutrient_practice="Organic",
            soil_cover="Cover_Crops",
        )
        assert result["score"] == 100  # Organic=100 + Cover=+15, capped at 100

    # ── Test 8: Crop Health score stability ──

    def test_crop_health_score_healthy_unchanged(self):
        """Healthy foliage must still score 95."""
        result = evaluate_crop_health(crop_health_status="Healthy")
        assert result["score"] == 95
        assert result["is_assessed"] is True

    def test_crop_health_not_assessed_unchanged(self):
        """Not_Assessed must still exclude from scoring."""
        result = evaluate_crop_health(crop_health_status="Not_Assessed")
        assert result["is_assessed"] is False
        assert result["score"] is None

    # ── Test 9: Alias coverage ──

    def test_crop_alias_paddy(self):
        """'paddy' alias resolves to rice."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement("paddy")
        assert req.is_available is True
        assert req.crop_key == "rice"

    def test_crop_alias_corn(self):
        """'corn' alias resolves to maize."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement("corn")
        assert req.is_available is True
        assert req.crop_key == "maize"

    def test_crop_alias_bajra(self):
        """'bajra' alias resolves to pearl_millet."""
        from model.sustainability.crop_requirements import get_crop_requirement
        req = get_crop_requirement("bajra")
        assert req.is_available is True
        assert req.crop_key == "pearl_millet"


# ==========================================================================
# 11. Phase 5.2: Recommendation Sensitivity & Dynamic Matrix Tests
# ==========================================================================

class TestPhase52RecommendationSensitivity:
    """
    Tests ensuring recommendations are genuinely input-sensitive,
    deterministic, dynamic, and non-generic across all agronomic dimensions.
    """

    def test_soil_moisture_20_vs_100_percent(self):
        """20% moisture and 100% moisture MUST NOT produce the same water recommendation."""
        service = SustainabilityService()
        res_20 = service.calculate_score(
            soil_moisture=20.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        res_100 = service.calculate_score(
            soil_moisture=100.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )

        water_rec_20 = next(r for r in res_20["recommendations"] if r["category"] == "Water Stewardship")
        water_rec_100 = next(r for r in res_100["recommendations"] if r["category"] == "Water Stewardship")

        # Must be completely different
        assert water_rec_20["title"] != water_rec_100["title"]
        assert water_rec_20["action"] != water_rec_100["action"]

        # 100% must advise holding unnecessary irrigation
        assert "Hold" in water_rec_100["title"] or "hold" in water_rec_100["action"].lower()
        assert "100%" in water_rec_100["action"] or "100%" in water_rec_100["why"]
        assert "Hold unnecessary irrigation" in water_rec_100["action"] or "avoid additional irrigation" in water_rec_100["action"].lower()

        # 20% must advise managing deficit
        assert "deficit" in water_rec_20["title"].lower() or "deficit" in water_rec_20["action"].lower()
        assert "20%" in water_rec_20["action"] or "20%" in water_rec_20["why"]

    def test_moisture_bands_coverage(self):
        """All 5 moisture condition bands produce distinct recommendations."""
        service = SustainabilityService()
        bands = [15.0, 30.0, 50.0, 72.0, 95.0]
        titles = []
        for m in bands:
            res = service.calculate_score(
                soil_moisture=m,
                irrigation_method="Drip",
                water_availability="Moderate",
                nutrient_practice="Integrated",
                soil_cover="Mulch",
                crop_health_status="Healthy",
            )
            w_rec = next(r for r in res["recommendations"] if r["category"] == "Water Stewardship")
            titles.append(w_rec["title"])

        # All 5 bands must have distinct titles
        assert len(set(titles)) == 5

    def test_crop_health_healthy_vs_severe_damage(self):
        """Healthy crop must not receive disease management; Severe Damage must receive critical inspection."""
        service = SustainabilityService()
        res_healthy = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        res_severe = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Severe_Damage",
        )

        h_rec_healthy = next(r for r in res_healthy["recommendations"] if r["category"] == "Crop Health")
        h_rec_severe = next(r for r in res_severe["recommendations"] if r["category"] == "Crop Health")

        assert h_rec_healthy["title"] != h_rec_severe["title"]

        # Healthy: No disease management advice
        assert "disease" not in h_rec_healthy["title"].lower()
        assert "Continue Regular Crop-Health Monitoring" in h_rec_healthy["title"]
        assert h_rec_healthy["priority"] == "Low"

        # Severe: Critical priority field inspection
        assert h_rec_severe["priority"] == "Critical"
        assert "Urgent Field Inspection" in h_rec_severe["title"] or "Inspection" in h_rec_severe["title"]
        assert "plant pathologist" in h_rec_severe["action"] or "agronomic" in h_rec_severe["action"].lower()

    def test_crop_health_not_assessed_no_disease_advice(self):
        """Not Assessed crop health does NOT generate disease-management advice."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Not_Assessed",
        )
        h_rec = next((r for r in res["recommendations"] if r["category"] == "Crop Health"), None)
        assert h_rec is not None
        assert "disease" not in h_rec["title"].lower()
        assert "No crop-health observation was provided" in h_rec["action"]

    def test_organic_mulch_vs_chemical_bare(self):
        """Organic+Mulch maintains practices; Intensive Chemical+Bare recommends nutrient & soil conservation."""
        service = SustainabilityService()
        res_organic = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            nutrient_practice="Organic",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        res_chemical = service.calculate_score(
            soil_moisture=50.0,
            irrigation_method="Drip",
            nutrient_practice="Intensive_Chemical",
            soil_cover="Bare_Soil",
            crop_health_status="Healthy",
        )

        r_recs_organic = [r for r in res_organic["recommendations"] if r["category"] == "Resource Conservation"]
        r_recs_chemical = [r for r in res_chemical["recommendations"] if r["category"] == "Resource Conservation"]

        # Organic+Mulch must not say 'switch to organic'
        assert len(r_recs_organic) == 1
        assert "Maintain Organic" in r_recs_organic[0]["title"]
        assert "switch" not in r_recs_organic[0]["action"].lower()
        assert r_recs_organic[0]["priority"] == "Low"

        # Intensive Chemical+Bare must recommend nutrient management and soil cover
        titles_chem = [r["title"] for r in r_recs_chemical]
        assert any("Organic or Integrated" in t for t in titles_chem)
        assert any("Protective Soil Cover" in t for t in titles_chem)
        assert any(r["priority"] == "High" for r in r_recs_chemical)

    def test_drip_vs_flood_under_high_moisture(self):
        """Under high moisture (>80%), Flood and Drip receive delivery-appropriate advisories."""
        service = SustainabilityService()
        res_flood = service.calculate_score(
            soil_moisture=90.0,
            irrigation_method="Flood",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        res_drip = service.calculate_score(
            soil_moisture=90.0,
            irrigation_method="Drip",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )

        w_flood = next(r for r in res_flood["recommendations"] if r["category"] == "Water Stewardship")
        w_drip = next(r for r in res_drip["recommendations"] if r["category"] == "Water Stewardship")

        assert "Flood" in w_flood["title"] or "flood" in w_flood["action"].lower()
        assert "Micro-Irrigation" in w_drip["title"] or "drip" in w_drip["action"].lower()
        assert w_flood["title"] != w_drip["title"]

    def test_scarce_vs_abundant_water_availability(self):
        """Water availability affects water stewardship advice."""
        service = SustainabilityService()
        res_scarce = service.calculate_score(
            soil_moisture=30.0,
            irrigation_method="Drip",
            water_availability="Scarce",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        res_abundant = service.calculate_score(
            soil_moisture=30.0,
            irrigation_method="Drip",
            water_availability="Abundant",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )

        w_scarce = next(r for r in res_scarce["recommendations"] if r["category"] == "Water Stewardship")
        w_abundant = next(r for r in res_abundant["recommendations"] if r["category"] == "Water Stewardship")

        assert "scarce" in w_scarce["action"].lower() or "scarce" in w_scarce["why"].lower()
        assert "abundant" in w_abundant["action"].lower() or "abundant" in w_abundant["why"].lower()
        assert w_scarce["action"] != w_abundant["action"]

    def test_weather_significant_rain_vs_negligible_rain(self):
        """Distinguishes high rain probability with significant rain vs negligible rain."""
        from model.sustainability.analyzer import analyze_sustainability_score

        weather_sig = {
            "has_rain_forecast": True,
            "max_rain_prob": 75.0,
            "expected_rain_mm": 15.0,
            "is_mock": False,
            "location": "Pune",
        }
        weather_neg = {
            "has_rain_forecast": True,
            "max_rain_prob": 75.0,
            "expected_rain_mm": 0.3,
            "is_mock": False,
            "location": "Pune",
        }

        res_sig = analyze_sustainability_score(
            soil_moisture=20.0,
            irrigation_method="Flood",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
            weather_summary=weather_sig,
            weather_status="LIVE",
        )
        res_neg = analyze_sustainability_score(
            soil_moisture=20.0,
            irrigation_method="Flood",
            water_availability="Moderate",
            nutrient_practice="Integrated",
            soil_cover="Mulch",
            crop_health_status="Healthy",
            weather_summary=weather_neg,
            weather_status="LIVE",
        )

        w_sig = next(r for r in res_sig["recommendations"] if r["category"] == "Water Stewardship")
        w_neg = next(r for r in res_neg["recommendations"] if r["category"] == "Water Stewardship")

        # Significant rain warns to review timing against rainfall
        assert "15.0 mm" in w_sig["action"]
        assert "Review irrigation timing against expected rainfall" in w_sig["action"]

        # Negligible rain clarifies accumulation is negligible
        assert "0.3 mm" in w_neg["action"]
        assert "negligible" in w_neg["action"].lower()

    def test_crop_critical_growth_stage_sensitivity(self):
        """Moisture-sensitive growth stage adds explicit crop context note."""
        service = SustainabilityService()
        res_flowering = service.calculate_score(
            soil_moisture=28.0,
            irrigation_method="Drip",
            crop="Tomato",
            growth_stage="Flowering",
        )
        res_maturity = service.calculate_score(
            soil_moisture=28.0,
            irrigation_method="Drip",
            crop="Tomato",
            growth_stage="Maturity",
        )

        w_flowering = next(r for r in res_flowering["recommendations"] if r["category"] == "Water Stewardship")
        w_maturity = next(r for r in res_maturity["recommendations"] if r["category"] == "Water Stewardship")

        assert "moisture-sensitive in the available crop data" in w_flowering["action"]
        assert "dry-down" in w_maturity["action"]

    def test_other_not_listed_no_fake_crop_advice(self):
        """Unsupported crop does not produce hardcoded crop-specific advice."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=30.0,
            irrigation_method="Drip",
            crop="Other / Not Listed",
            growth_stage="Vegetative",
        )
        w_rec = next(r for r in res["recommendations"] if r["category"] == "Water Stewardship")
        assert "Tomato" not in w_rec["action"]
        assert "Rice" not in w_rec["action"]

    def test_deterministic_reproducibility(self):
        """Identical inputs produce 100% identical recommendations."""
        service = SustainabilityService()
        res1 = service.calculate_score(
            soil_moisture=25.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            nutrient_practice="Intensive_Chemical",
            soil_cover="Bare_Soil",
            crop_health_status="Disease_Detected",
        )
        res2 = service.calculate_score(
            soil_moisture=25.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            nutrient_practice="Intensive_Chemical",
            soil_cover="Bare_Soil",
            crop_health_status="Disease_Detected",
        )
        assert res1["recommendations"] == res2["recommendations"]


class TestPhase52RecommendationMatrix:
    """
    End-to-end scenario matrix tests verifying meaningfully different,
    agronomically coherent recommendation sets.
    """

    def test_scenario_a_critical_dry_field(self):
        """Scenario A: Critical Dry Field produces urgent, multi-dimensional interventions."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=20.0,
            irrigation_method="Flood",
            water_availability="Scarce",
            nutrient_practice="Intensive_Chemical",
            soil_cover="Bare_Soil",
            crop_health_status="Disease_Detected",
        )
        recs = res["recommendations"]
        titles = [r["title"] for r in recs]
        priorities = [r["priority"] for r in recs]

        # High or Critical severity across components
        assert any(p in ("Critical", "High") for p in priorities)
        assert any("Micro-Irrigation" in t for t in titles)
        assert any("Organic or Integrated" in t for t in titles)
        assert any("Protective Soil Cover" in t for t in titles)
        assert any("Disease Management" in t for t in titles)

    def test_scenario_b_very_wet_field(self):
        """Scenario B: Very Wet Field holds unnecessary irrigation, no disease/fertilizer switch."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=100.0,
            irrigation_method="Flood",
            water_availability="Abundant",
            nutrient_practice="Organic",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        recs = res["recommendations"]
        titles = [r["title"] for r in recs]

        # Water: Holds unnecessary irrigation
        w_rec = next(r for r in recs if r["category"] == "Water Stewardship")
        assert "Hold" in w_rec["title"]
        assert "100%" in w_rec["action"]

        # NO disease-management recommendation
        assert not any("disease" in t.lower() for t in titles)

        # NO fertilizer-switch recommendation
        assert not any("adopt" in t.lower() for t in titles)

        # Resource & Health are maintain
        assert any("Maintain Organic" in t for t in titles)
        assert any("Continue Regular Crop-Health Monitoring" in t for t in titles)

    def test_scenario_c_strong_balanced_field(self):
        """Scenario C: Strong Balanced Field produces purely maintain/monitor cards."""
        service = SustainabilityService()
        res = service.calculate_score(
            soil_moisture=55.0,
            irrigation_method="Drip",
            water_availability="Abundant",
            nutrient_practice="Organic",
            soil_cover="Mulch",
            crop_health_status="Healthy",
        )
        recs = res["recommendations"]
        titles = [r["title"] for r in recs]
        priorities = [r["priority"] for r in recs]

        # All priorities must be Low
        assert all(p == "Low" for p in priorities)
        assert any("Maintain Optimal Soil Moisture Balance" in t for t in titles)
        assert any("Maintain Organic" in t for t in titles)
        assert any("Continue Regular Crop-Health Monitoring" in t for t in titles)

    def test_scenarios_a_b_c_are_meaningfully_different(self):
        """Verify recommendation titles across Scenarios A, B, and C are mutually distinct."""
        service = SustainabilityService()
        rec_a = [r["title"] for r in service.calculate_score(20.0, "Flood", "Scarce", "Intensive_Chemical", "Bare_Soil", "Disease_Detected")["recommendations"]]
        rec_b = [r["title"] for r in service.calculate_score(100.0, "Flood", "Abundant", "Organic", "Mulch", "Healthy")["recommendations"]]
        rec_c = [r["title"] for r in service.calculate_score(55.0, "Drip", "Abundant", "Organic", "Mulch", "Healthy")["recommendations"]]

        assert rec_a != rec_b
        assert rec_b != rec_c
        assert rec_a != rec_c
