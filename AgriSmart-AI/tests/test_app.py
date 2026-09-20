"""
Tests for AgriSmart AI Flask Web Application and Service Layer.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image

from app.main import create_app
from app.services.disease_service import DiseaseDetectionService
from app.services.weather_service import WeatherIntelligenceService
from app.services.irrigation_service import SmartIrrigationService
from app.services.crop_service import CropRecommendationService
from app.services.sustainability_service import SustainabilityService
from app.services.assistant_service import FarmerAssistantService
from app.services.iot_service import IoTTelemetryService
from app.services.advisor_service import AgenticAdvisorService


@pytest.fixture()
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture()
def synthetic_image_bytes() -> io.BytesIO:
    """Generate in-memory valid JPEG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────
# 1. Route Tests
# ──────────────────────────────────────────────

def test_landing_page(client):
    """The landing page should return 200 and include AgriSmart branding."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "AgriSmart AI" in html
    assert "Intelligent Agriculture for a Sustainable Future" in html
    assert "Upload Crop Leaf Image" in html


def test_health_endpoint(client):
    """The /health endpoint must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


def test_analyze_valid_image_mock_mode(client, synthetic_image_bytes):
    """Uploading a valid image when checkpoint is not trained returns mock diagnosis."""
    data = {
        "image": (synthetic_image_bytes, "leaf.jpg", "image/jpeg"),
    }
    response = client.post(
        "/analyze",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 200
    res = response.get_json()
    assert res["status"] == "success"
    # When best_model.pt is not trained yet, mock mode must be explicitly flagged
    assert res["is_mock"] is True
    assert "MOCK" in res["mock_disclaimer"].upper()
    assert "crop" in res
    assert "condition" in res
    assert "confidence" in res
    assert "precautionary_guidance" in res
    assert isinstance(res["precautionary_guidance"], list)
    assert len(res["precautionary_guidance"]) > 0


def test_analyze_missing_image(client):
    """Submitting with no image field should return 400 Bad Request."""
    response = client.post(
        "/analyze",
        data={},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    res = response.get_json()
    assert res["status"] == "error"
    assert "No image file provided" in res["message"]


def test_analyze_invalid_extension(client):
    """Submitting a non-image file should return 400 Bad Request."""
    text_file = io.BytesIO(b"Hello farm")
    data = {
        "image": (text_file, "notes.txt", "text/plain"),
    }
    response = client.post(
        "/analyze",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    res = response.get_json()
    assert res["status"] == "error"
    assert "Unsupported file type" in res["message"]


def test_analyze_corrupted_image(client):
    """Submitting an unreadable image payload should return 400 Bad Request."""
    bad_bytes = io.BytesIO(b"NOT_REALLY_AN_IMAGE_FILE")
    data = {
        "image": (bad_bytes, "corrupt.jpg", "image/jpeg"),
    }
    response = client.post(
        "/analyze",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    res = response.get_json()
    assert res["status"] == "error"
    assert "could not be read as a valid image" in res["message"]


# ──────────────────────────────────────────────
# 2. Disease Service & Dynamic Class Formatting
# ──────────────────────────────────────────────

class TestDiseaseDetectionService:
    """Test dynamic class formatting and advisory logic without hardcoded catalogs."""

    def test_dynamic_crop_and_disease_formatting(self):
        svc = DiseaseDetectionService()

        # Diseased class with triple underscore
        res1 = svc.format_class_name("Tomato___Early_blight")
        assert res1["crop"] == "Tomato"
        assert res1["condition"] == "Early Blight"
        assert res1["is_healthy"] is False
        assert res1["display_title"] == "Tomato — Early Blight"

        # Healthy class
        res2 = svc.format_class_name("Apple___healthy")
        assert res2["crop"] == "Apple"
        assert res2["condition"] == "Healthy"
        assert res2["is_healthy"] is True

        # Special casing with parentheses
        res3 = svc.format_class_name("Corn_(maize)___Common_rust_")
        assert "Corn" in res3["crop"]
        assert "Common Rust" in res3["condition"]
        assert res3["is_healthy"] is False

        # Completely unseen organizer class (dynamic test)
        res4 = svc.format_class_name("Soybean___Frogeye_leaf_spot")
        assert res4["crop"] == "Soybean"
        assert res4["condition"] == "Frogeye Leaf Spot"
        assert res4["is_healthy"] is False

    def test_precautionary_guidance_types(self):
        svc = DiseaseDetectionService()
        healthy_guidance = svc.get_precautionary_guidance(is_healthy=True, condition_name="Healthy")
        diseased_guidance = svc.get_precautionary_guidance(is_healthy=False, condition_name="Early Blight")

        assert isinstance(healthy_guidance, list) and len(healthy_guidance) >= 3
        assert isinstance(diseased_guidance, list) and len(diseased_guidance) >= 3
        # Confirm guidance is advisory rather than chemical prescription
        combined = " ".join(diseased_guidance).lower()
        assert "isolate" in combined or "prune" in combined
        assert "extension officer" in combined or "kvk" in combined


# ──────────────────────────────────────────────
# 3. Future Service Boundaries Tests
# ──────────────────────────────────────────────

class TestFutureServiceBoundaries:
    """Verify that all future module interfaces are cleanly defined and raise NotImplementedError."""

    def test_weather_service_stub(self):
        svc = WeatherIntelligenceService()
        with pytest.raises(NotImplementedError):
            svc.get_current_conditions(28.6139, 77.2090)

    def test_irrigation_service_stub(self):
        svc = SmartIrrigationService()
        with pytest.raises(NotImplementedError):
            svc.calculate_water_requirement("Tomato", "Vegetative", "Loam")

    def test_crop_service_stub(self):
        svc = CropRecommendationService()
        with pytest.raises(NotImplementedError):
            svc.recommend_crops(80, 40, 40, 6.5, 120)

    def test_sustainability_service_stub(self):
        svc = SustainabilityService()
        with pytest.raises(NotImplementedError):
            svc.compute_farm_score(50.0, 1000.0, True)

    def test_assistant_service_stub(self):
        svc = FarmerAssistantService()
        with pytest.raises(NotImplementedError):
            svc.answer_query("How do I control aphids?")

    def test_iot_service_stub(self):
        svc = IoTTelemetryService()
        with pytest.raises(NotImplementedError):
            svc.get_latest_telemetry("node-01")

    def test_advisor_service_stub(self):
        svc = AgenticAdvisorService()
        with pytest.raises(NotImplementedError):
            svc.generate_intervention_plan("farm-42", [])
