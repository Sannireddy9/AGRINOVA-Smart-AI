"""
AgriSmart AI — Crop Recommendation Test Suite
==============================================
Comprehensive unit, integration, and service isolation tests for the
Crop Recommendation module as specified in SIH 2026 Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.main import create_app
from app.services.crop_service import CropRecommendationService
from app.services.disease_service import DiseaseDetectionService
from model.crop_recommendation.config import CropRecommendationConfig
from model.crop_recommendation.dataset import (
    get_train_val_test_splits,
    load_dataset,
    validate_dataset,
)
from model.crop_recommendation.predict import (
    format_crop_display_name,
    load_crop_model,
    predict,
    validate_input_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INPUT VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """Tests for physical boundary checks and input parsing."""

    def test_valid_features_dict(self):
        valid = {
            "N": 50,
            "P": 50,
            "K": 50,
            "temperature": 28.5,
            "humidity": 70.0,
            "ph": 6.5,
            "rainfall": 120.0,
        }
        arr = validate_input_features(valid)
        assert arr.shape == (1, 7)
        assert arr[0, 0] == 50.0
        assert arr[0, 5] == 6.5

    def test_valid_features_aliases(self):
        aliased = {
            "nitrogen": 40,
            "phosphorus": 45,
            "potassium": 30,
            "temp": 26.0,
            "relative_humidity": 80.0,
            "soil_ph": 7.0,
            "rain": 150.0,
        }
        arr = validate_input_features(aliased)
        assert arr.shape == (1, 7)
        assert arr[0, 0] == 40.0
        assert arr[0, 3] == 26.0

    def test_valid_features_array(self):
        raw_list = [50.0, 50.0, 50.0, 25.0, 80.0, 6.5, 100.0]
        arr = validate_input_features(raw_list)
        assert arr.shape == (1, 7)

    def test_invalid_ph_negative(self):
        bad_data = {"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": 70, "ph": -1.0, "rainfall": 100}
        with pytest.raises(ValueError, match="outside permissible physical bounds"):
            validate_input_features(bad_data)

    def test_invalid_ph_too_high(self):
        bad_data = {"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": 70, "ph": 14.5, "rainfall": 100}
        with pytest.raises(ValueError, match="outside permissible physical bounds"):
            validate_input_features(bad_data)

    def test_invalid_humidity_low(self):
        bad_data = {"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": -5, "ph": 6.5, "rainfall": 100}
        with pytest.raises(ValueError, match="outside permissible physical bounds"):
            validate_input_features(bad_data)

    def test_invalid_humidity_high(self):
        bad_data = {"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": 105, "ph": 6.5, "rainfall": 100}
        with pytest.raises(ValueError, match="outside permissible physical bounds"):
            validate_input_features(bad_data)

    def test_invalid_rainfall_negative(self):
        bad_data = {"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": -20}
        with pytest.raises(ValueError, match="outside permissible physical bounds"):
            validate_input_features(bad_data)

    def test_missing_required_feature(self):
        missing = {"N": 50, "P": 50, "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 100}
        with pytest.raises(ValueError, match="Missing required model feature"):
            validate_input_features(missing)

    def test_non_numeric_feature(self):
        invalid_type = {"N": "high", "P": 50, "K": 50, "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 100}
        with pytest.raises(ValueError, match="must be a numeric value"):
            validate_input_features(invalid_type)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL AND DATASET PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCropModelAndDataset:
    """Tests for dataset splitting, model loading, and prediction ranking."""

    def test_dataset_loading_and_validation(self):
        df = load_dataset()
        info = validate_dataset(df)
        assert info["num_rows"] == 2200
        assert info["num_features"] == 7
        assert info["num_classes"] == 22

    def test_dataset_stratified_split_no_leakage(self):
        df = load_dataset()
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(df)

        # Confirm exact expected split proportions (70 / 15 / 15)
        assert len(X_train) == 1540
        assert len(X_val) == 330
        assert len(X_test) == 330

        # Confirm no index overlap across splits (zero leakage)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_model_loading(self):
        model, classes = load_crop_model()
        assert model is not None
        assert len(classes) == 22
        assert "rice" in classes

    def test_prediction_output_format(self):
        feat = {"N": 50, "P": 50, "K": 50, "temperature": 28, "humidity": 70, "ph": 6.5, "rainfall": 120}
        recs = predict(feat, top_k=3)

        assert isinstance(recs, list)
        assert len(recs) == 3

        for i, r in enumerate(recs, start=1):
            assert r["rank"] == i
            assert "crop" in r
            assert "raw_class" in r
            assert "suitability_percentage" in r
            assert "probability" in r

    def test_prediction_ranking_order_descending(self):
        feat = {"N": 90, "P": 40, "K": 40, "temperature": 22, "humidity": 82, "ph": 6.5, "rainfall": 200}
        recs = predict(feat, top_k=5)

        percentages = [r["suitability_percentage"] for r in recs]
        assert percentages == sorted(percentages, reverse=True)

    def test_prediction_probabilities_range(self):
        feat = {"N": 50, "P": 50, "K": 50, "temperature": 28, "humidity": 70, "ph": 6.5, "rainfall": 120}
        recs = predict(feat, top_k=5)

        for r in recs:
            assert 0.0 <= r["probability"] <= 1.0
            assert 0.0 <= r["suitability_percentage"] <= 100.0

    def test_dynamic_class_formatting(self):
        assert format_crop_display_name("rice") == "Rice"
        assert format_crop_display_name("pigeonpeas") == "Pigeon Peas"
        assert format_crop_display_name("kidneybeans") == "Kidney Beans"
        assert format_crop_display_name("blackgram") == "Black Gram"


# ─────────────────────────────────────────────────────────────────────────────
# 3. SERVICE BOUNDARY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCropRecommendationService:
    """Tests for application service layer: Live vs Demo mode, context segregation."""

    def test_service_live_mode(self):
        svc = CropRecommendationService()
        assert svc.is_model_available() is True

        payload = {
            "N": 50, "P": 50, "K": 50,
            "temperature": 28, "humidity": 70, "ph": 6.5, "rainfall": 120,
            "soil_type": "Loamy", "water_availability": "Medium", "season": "Kharif",
            "location": "Ahmedabad", "previous_crop": "Wheat",
        }
        res = svc.recommend(payload, top_k=3)

        assert res["status"] == "success"
        assert res["is_mock"] is False
        assert res["mode"] == "LIVE"
        assert len(res["recommendations"]) == 3
        assert "suitability_percentage" in res["recommendations"][0]

    def test_service_demo_mode_when_model_missing(self, tmp_path):
        fake_model_path = tmp_path / "non_existent_crop_model.joblib"
        svc = CropRecommendationService(model_path=fake_model_path)
        assert svc.is_model_available() is False

        payload = {
            "N": 50, "P": 50, "K": 50,
            "temperature": 28, "humidity": 70, "ph": 6.5, "rainfall": 120,
            "soil_type": "Loamy", "water_availability": "Medium", "season": "Kharif",
            "location": "Ahmedabad", "previous_crop": "Wheat",
        }
        res = svc.recommend(payload, top_k=3)

        assert res["status"] == "success"
        assert res["is_mock"] is True
        assert res["mode"] == "DEVELOPMENT_DEMO"
        assert "DEMO SIMULATION" in res["disclaimer"]
        assert all(r.get("is_simulated") is True for r in res["recommendations"])

    def test_service_model_vs_context_separation(self):
        svc = CropRecommendationService()
        payload = {
            "N": 60, "P": 55, "K": 44,
            "temperature": 23, "humidity": 82, "ph": 7.8, "rainfall": 260,
            "soil_type": "Clayey", "water_availability": "High", "season": "Kharif",
            "location": "Punjab", "previous_crop": "Legumes",
        }
        res = svc.recommend(payload)

        # Check model features contains exactly the 7 numeric variables
        model_feats = res["model_features_used"]
        assert set(model_feats.keys()) == set(CropRecommendationConfig.MODEL_FEATURES)

        # Check context fields contains the 5 contextual attributes
        context_feats = res["additional_context_collected"]
        for field in CropRecommendationConfig.CONTEXTUAL_FIELDS:
            assert field in context_feats

        assert context_feats["soil_type"] == "Clayey"
        assert context_feats["previous_crop"] == "Legumes"
        assert "_notice" in context_feats

    def test_service_agronomic_explanation(self):
        svc = CropRecommendationService()
        payload = {
            "N": 50, "P": 50, "K": 50,
            "temperature": 28, "humidity": 70, "ph": 6.5, "rainfall": 120,
            "soil_type": "Loamy", "water_availability": "Medium", "season": "Kharif",
            "location": "Ahmedabad", "previous_crop": "Wheat",
        }
        res = svc.recommend(payload)
        explanation = res["agronomic_explanation"]

        assert isinstance(explanation, str)
        assert len(explanation) > 20
        assert "pH 6.5" in explanation
        assert "28.0" in explanation or "28" in explanation

    def test_service_future_hooks_exist(self):
        svc = CropRecommendationService()
        # Should execute cleanly without errors as no-op extension points
        svc.attach_weather_context({"temp": 28.0})
        svc.attach_irrigation_context({"water_mm": 50.0})
        svc.attach_sustainability_context({"rotation_score": 85})
        svc.attach_assistant_context({"language": "hi"})
        svc.attach_advisor_context({"plan_horizon": 3})

    def test_service_isolation_from_disease_detection(self):
        """Crop recommendation must not affect or depend on disease detection logic."""
        crop_svc = CropRecommendationService()
        disease_svc = DiseaseDetectionService()

        # Both services exist independently
        assert hasattr(crop_svc, "recommend")
        assert hasattr(disease_svc, "analyze_crop_image")
        assert crop_svc.model_path != disease_svc.checkpoint_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEB ENDPOINT INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestCropRecommendationWebRoutes:
    """Tests for GET /crop-recommendation and POST /recommend-crops."""

    def test_get_crop_recommendation_page(self, client):
        response = client.get("/crop-recommendation")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Intelligent Crop Recommendation" in html
        assert "ML Features Used for Prediction" in html
        assert "Additional Farmer Context — Not Used by Current ML Model" in html
        assert "These details are collected for farm context and future advisory improvements" in html

    def test_post_recommend_crops_ajax_valid(self, client):
        payload = {
            "n": 50,
            "p": 50,
            "k": 50,
            "temperature": 28.0,
            "humidity": 70.0,
            "ph": 6.5,
            "rainfall": 120.0,
            "soil_type": "Loamy",
            "water_availability": "Medium",
            "season": "Kharif",
            "location": "Ahmedabad",
            "previous_crop": "Wheat",
        }
        response = client.post(
            "/recommend-crops",
            json=payload,
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert len(data["recommendations"]) >= 1
        assert "model_features_used" in data
        assert "additional_context_collected" in data

    def test_post_recommend_crops_invalid_ph(self, client):
        payload = {
            "n": 50, "p": 50, "k": 50,
            "temperature": 28.0, "humidity": 70.0,
            "ph": 15.0,  # Invalid pH (> 14)
            "rainfall": 120.0,
        }
        response = client.post(
            "/recommend-crops",
            json=payload,
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"

    def test_post_recommend_crops_missing_feature(self, client):
        payload = {
            "n": 50, "p": 50,
            "temperature": 28.0, "humidity": 70.0,
            # Missing K, ph, rainfall
        }
        response = client.post(
            "/recommend-crops",
            json=payload,
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
