"""
AgriSmart AI — Main Route Handlers
===================================
Handles the farmer dashboard, crop image upload, and diagnosis rendering.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any
from PIL import Image

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
)
from werkzeug.utils import secure_filename

from app.config import AppConfig
from app.services.disease_service import DiseaseDetectionService
from app.services.crop_service import CropRecommendationService
from app.services.weather_service import WeatherIntelligenceService
from app.services.irrigation_service import SmartIrrigationService
from app.services.sustainability_service import SustainabilityService
from app.services.farmer_assistant_service import FarmerAssistantService

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)
disease_service = DiseaseDetectionService()
crop_service = CropRecommendationService()
weather_service = WeatherIntelligenceService()
irrigation_service = SmartIrrigationService(weather_service=weather_service)
sustainability_service = SustainabilityService(weather_service=weather_service)
farmer_assistant_service = FarmerAssistantService(
    weather_service=weather_service,
    irrigation_service=irrigation_service,
    sustainability_service=sustainability_service,
    crop_service=crop_service,
    disease_service=disease_service,
)


@main_bp.app_context_processor
def inject_module_statuses():
    """Inject module-specific operational statuses across all templates."""
    disease_live = disease_service.is_checkpoint_available()
    crop_live = crop_service.is_model_available()
    weather_live = weather_service.is_live_configured()
    irrigation_live = irrigation_service.is_live_configured()
    return {
        "is_disease_model_live": disease_live,
        "is_crop_model_live": crop_live,
        "is_weather_service_live": weather_live,
        "is_irrigation_live": irrigation_live,
        "is_sustainability_live": True,
        "is_assistant_live": True,
        "is_model_live": disease_live,
    }


@main_bp.app_template_filter("farmer_date")
def farmer_date_filter(date_str: str) -> str:
    """Format an ISO date string (YYYY-MM-DD) into farmer-friendly 'Thu, Sep 10'."""
    if not date_str:
        return ""
    try:
        clean_date = date_str.strip()[:10]
        dt = datetime.date.fromisoformat(clean_date)
        return dt.strftime("%a, %b %d")
    except Exception:
        return date_str


def _update_compact_farm_context(updates: dict[str, Any]) -> None:
    """Store strictly compact farmer inputs in session without heavy payloads."""
    ctx = dict(session.get("farm_context", {}))
    scalar_keys = (
        "crop",
        "growth_stage",
        "location",
        "soil_moisture",
        "irrigation_method",
        "water_availability",
        "nutrient_practice",
        "soil_cover",
        "crop_health_status",
        "disease_prediction",
        "recommended_crop",
        "sustainability_score",
        "language",
    )
    for key in scalar_keys:
        if key in updates and updates[key] is not None:
            val_str = str(updates[key]).strip()
            if not val_str:
                continue
            if key == "soil_moisture":
                try:
                    ctx[key] = float(updates[key])
                except (ValueError, TypeError):
                    pass
            elif key == "sustainability_score":
                try:
                    ctx[key] = round(float(updates[key]), 1) if "." in val_str else int(float(updates[key]))
                except (ValueError, TypeError):
                    pass
            else:
                ctx[key] = val_str
    session["farm_context"] = ctx


def _is_allowed_file(filename: str) -> bool:
    """Return True if filename has an allowed image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in AppConfig.ALLOWED_EXTENSIONS


@main_bp.route("/")
@main_bp.route("/disease")
def index():
    """Main dashboard and crop diagnosis landing page."""
    is_live = disease_service.is_checkpoint_available()
    return render_template(
        "index.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=is_live,
    )


@main_bp.route("/analyze", methods=["POST"])
def analyze_crop():
    """Process uploaded crop image and return diagnosis."""
    # 1. Check for file presence
    if "image" not in request.files:
        return (
            jsonify({"status": "error", "message": "No image file provided. Please choose a leaf image."}),
            400,
        )

    file = request.files["image"]
    if not file:
        return (
            jsonify({"status": "error", "message": "No image selected. Please choose a leaf image to analyze."}),
            400,
        )

    filename = file.filename
    if not filename:
        return (
            jsonify({"status": "error", "message": "No image selected. Please choose a leaf image to analyze."}),
            400,
        )

    # 2. Check extension
    if not _is_allowed_file(filename):
        allowed = ", ".join(sorted(AppConfig.ALLOWED_EXTENSIONS)).upper()
        return (
            jsonify({
                "status": "error",
                "message": f"Unsupported file type. Please upload an image ({allowed}).",
            }),
            400,
        )

    # 3. Secure filename and save to uploads folder
    orig_name = secure_filename(filename) or "upload.jpg"
    unique_name = f"{uuid.uuid4().hex[:10]}_{orig_name}"
    upload_path = AppConfig.UPLOAD_FOLDER / unique_name

    try:
        file.save(upload_path)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return (
            jsonify({"status": "error", "message": "Failed to store the uploaded image. Please try again."}),
            500,
        )

    # 4. Validate that file can be decoded as an image
    try:
        with Image.open(upload_path) as img:
            img.verify()
        with Image.open(upload_path) as img:
            img.load()
    except Exception as exc:
        logger.warning("Uploaded file is corrupted or not a valid image: %s", exc)
        # Clean up corrupted file
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass
        return (
            jsonify({
                "status": "error",
                "message": "The uploaded file could not be read as a valid image. It may be corrupted.",
            }),
            400,
        )

    # 5. Run diagnosis via disease service
    try:
        filepath = upload_path
        diagnosis = disease_service.diagnose(filepath)
        diagnosis["image_url"] = f"/uploads/{unique_name}"

        # Update compact farm context with detected disease and crop
        pred = diagnosis.get("predicted_class") or diagnosis.get("disease_name")
        crop_name = diagnosis.get("crop")
        _update_compact_farm_context({
            "disease_prediction": pred,
            "crop": crop_name if crop_name and str(crop_name).lower() != "unknown" else None,
        })
    except Exception as exc:
        logger.error("Error running disease analysis: %s", exc, exc_info=True)
        return (
            jsonify({
                "status": "error",
                "message": f"An error occurred while analyzing the image: {str(exc)}",
            }),
            500,
        )

    # 6. Return response (JSON for AJAX / SPA, or template render for form post)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json or "multipart/form-data" in request.content_type:
        return jsonify(diagnosis)

    return render_template(
        "index.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=disease_service.is_checkpoint_available(),
        result=diagnosis,
    )


@main_bp.route("/uploads/<filename>")
def uploaded_file(filename: str):
    """Serve uploaded image files for frontend display."""
    return send_from_directory(AppConfig.UPLOAD_FOLDER, filename)


@main_bp.route("/crop-recommendation")
def crop_recommendation_page():
    """Crop recommendation form and decision support page."""
    is_live = crop_service.is_model_available()
    return render_template(
        "crop_recommendation.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=disease_service.is_checkpoint_available(),
        is_crop_model_live=is_live,
    )


@main_bp.route("/recommend-crops", methods=["POST"])
def recommend_crops_endpoint():
    """Handle crop recommendation form submission or AJAX request."""
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    try:
        result = crop_service.recommend(data)
        if result and result.get("recommended_crop"):
            _update_compact_farm_context({
                "recommended_crop": result.get("recommended_crop"),
            })
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)
        return render_template(
            "crop_recommendation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            result=result,
            form_data=data,
        )
    except ValueError as exc:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return render_template(
            "crop_recommendation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            error=str(exc),
            form_data=data,
        ), 400
    except Exception as exc:
        logger.error("Crop recommendation error: %s", exc, exc_info=True)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": f"Server error: {str(exc)}"}), 500
        return render_template(
            "crop_recommendation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            error=f"An unexpected error occurred: {str(exc)}",
            form_data=data,
        ), 500


# ── Weather Intelligence Endpoints ────────────────────────────────────

@main_bp.route("/weather", methods=["GET"])
@main_bp.route("/weather-intelligence", methods=["GET"])
def weather_dashboard():
    """Render the Weather-Based Intelligence dashboard."""
    default_location = "Ahmedabad, Gujarat"
    default_context = {
        "crop": "Tomato",
        "growth_stage": "Vegetative",
        "soil_moisture": 45.0,
        "soil_type": "Loamy",
        "irrigation_method": "Drip",
    }
    force_refresh = request.args.get("force_refresh", "").lower() in ("true", "1", "yes")
    initial_analysis = weather_service.analyze(
        location=default_location,
        farm_context=default_context,
        force_refresh=force_refresh,
    )

    return render_template(
        "weather.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=disease_service.is_checkpoint_available(),
        is_crop_model_live=crop_service.is_model_available(),
        is_weather_service_live=weather_service.is_live_configured(),
        result=initial_analysis,
        form_data={"location": default_location, **default_context},
    )


@main_bp.route("/analyze-weather", methods=["POST"])
def analyze_weather_endpoint():
    """Execute meteorological evaluation for farmer parameters."""
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    location = str(data.get("location", "Ahmedabad, Gujarat")).strip() or "Ahmedabad, Gujarat"
    force_refresh = data.get("force_refresh", False) in (True, "true", "True", "1", 1)

    try:
        soil_moisture = float(data.get("soil_moisture", 50.0))
    except (ValueError, TypeError):
        soil_moisture = 50.0

    farm_context = {
        "crop": str(data.get("crop", "Tomato")).strip() or "Tomato",
        "growth_stage": str(data.get("growth_stage", "Vegetative")).strip() or "Vegetative",
        "soil_moisture": soil_moisture,
        "soil_type": str(data.get("soil_type", "Loamy")).strip() or "Loamy",
        "irrigation_method": str(data.get("irrigation_method", "Drip")).strip() or "Drip",
    }

    try:
        result = weather_service.analyze(
            location=location,
            farm_context=farm_context,
            force_refresh=force_refresh,
        )
        _update_compact_farm_context({
            "location": location,
            "crop": farm_context.get("crop"),
            "growth_stage": farm_context.get("growth_stage"),
            "soil_moisture": farm_context.get("soil_moisture"),
            "irrigation_method": farm_context.get("irrigation_method"),
        })
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)
        return render_template(
            "weather.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            result=result,
            form_data={"location": location, **farm_context},
        )
    except Exception as exc:
        logger.error("Weather intelligence analysis error: %s", exc, exc_info=True)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": f"Analysis error: {str(exc)}"}), 500
        return render_template(
            "weather.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            error=f"An error occurred during weather evaluation: {str(exc)}",
            form_data={"location": location, **farm_context},
        ), 500


# ── Smart Irrigation Advisor Routes ──────────────────────────────────────────

@main_bp.route("/irrigation")
@main_bp.route("/smart-irrigation")
def irrigation_page():
    """Render the Smart Irrigation Advisor dashboard."""
    raw_location = request.args.get("location")
    default_crop = request.args.get("crop", "Tomato").strip() or "Tomato"
    default_stage = request.args.get("growth_stage", "Vegetative").strip() or "Vegetative"
    try:
        default_moisture = float(request.args.get("soil_moisture", 35.0))
    except (ValueError, TypeError):
        default_moisture = 35.0
    default_soil_type = request.args.get("soil_type", "Loamy").strip() or "Loamy"
    default_method = request.args.get("irrigation_method", "Drip").strip() or "Drip"
    force_refresh = request.args.get("force_refresh", "").lower() in ("true", "1")

    initial_result = None
    error_msg = None

    if raw_location is not None:
        clean_location = raw_location.strip()
        if clean_location:
            try:
                initial_result = irrigation_service.get_advisory(
                    location=clean_location,
                    crop=default_crop,
                    growth_stage=default_stage,
                    soil_moisture=default_moisture,
                    soil_type=default_soil_type,
                    irrigation_method=default_method,
                    force_refresh=force_refresh,
                )
            except ValueError as val_err:
                error_msg = str(val_err)
            except Exception as exc:
                logger.warning("Could not pre-load irrigation advisory for '%s': %s", clean_location, exc)
                error_msg = "Unable to retrieve meteorological data for that location. Please retry."
        else:
            error_msg = "Please enter your farm location so we can retrieve the correct weather forecast."

    return render_template(
        "irrigation.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=disease_service.is_checkpoint_available(),
        is_crop_model_live=crop_service.is_model_available(),
        is_weather_service_live=weather_service.is_live_configured(),
        is_irrigation_live=irrigation_service.is_live_configured(),
        result=initial_result,
        error=error_msg,
        form_data={
            "location": raw_location.strip() if raw_location is not None else "",
            "crop": default_crop,
            "growth_stage": default_stage,
            "soil_moisture": default_moisture,
            "soil_type": default_soil_type,
            "irrigation_method": default_method,
        },
    )


@main_bp.route("/calculate-irrigation", methods=["POST"])
def calculate_irrigation_endpoint():
    """Calculate rule-based smart irrigation advisory based on manual moisture and weather forecast."""
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    raw_location = data.get("location")
    crop = str(data.get("crop", "Tomato")).strip() or "Tomato"
    growth_stage = str(data.get("growth_stage", "Vegetative")).strip() or "Vegetative"
    raw_moisture = data.get("soil_moisture", 35.0)
    soil_type = str(data.get("soil_type", "Loamy")).strip() or "Loamy"
    irrigation_method = str(data.get("irrigation_method", "Drip")).strip() or "Drip"
    force_refresh = data.get("force_refresh", False) in (True, "true", "True", "1", 1)

    # 1. Location validation: empty, whitespace-only, null, or missing
    if raw_location is None or not str(raw_location).strip():
        msg = "Please enter your farm location so we can retrieve the correct weather forecast."
        try:
            moisture_display_val = float(raw_moisture)
        except (ValueError, TypeError):
            moisture_display_val = 35.0

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": msg}), 400
        return render_template(
            "irrigation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            error=msg,
            form_data={
                "location": "",
                "crop": crop,
                "growth_stage": growth_stage,
                "soil_moisture": moisture_display_val,
                "soil_type": soil_type,
                "irrigation_method": irrigation_method,
            },
        ), 400

    clean_location = str(raw_location).strip()

    # 2. Validate soil moisture
    try:
        soil_moisture = float(raw_moisture)
        if soil_moisture < 0.0 or soil_moisture > 100.0:
            raise ValueError("Manual soil moisture must be between 0% and 100%.")
    except (ValueError, TypeError) as val_err:
        msg = f"Invalid soil moisture: {str(val_err)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": msg}), 400
        return render_template(
            "irrigation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            error=msg,
            form_data={
                "location": clean_location, "crop": crop, "growth_stage": growth_stage,
                "soil_moisture": 35.0, "soil_type": soil_type, "irrigation_method": irrigation_method
            },
        ), 400

    try:
        result = irrigation_service.get_advisory(
            location=clean_location,
            crop=crop,
            growth_stage=growth_stage,
            soil_moisture=soil_moisture,
            soil_type=soil_type,
            irrigation_method=irrigation_method,
            force_refresh=force_refresh,
        )
        _update_compact_farm_context({
            "location": clean_location,
            "crop": crop,
            "growth_stage": growth_stage,
            "soil_moisture": soil_moisture,
            "irrigation_method": irrigation_method,
        })

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)

        return render_template(
            "irrigation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            result=result,
            form_data={
                "location": clean_location,
                "crop": crop,
                "growth_stage": growth_stage,
                "soil_moisture": soil_moisture,
                "soil_type": soil_type,
                "irrigation_method": irrigation_method,
            },
        )
    except ValueError as val_err:
        logger.warning("Validation error in calculate_irrigation: %s", val_err)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": str(val_err)}), 400
        return render_template(
            "irrigation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            error=str(val_err),
            form_data={
                "location": clean_location, "crop": crop, "growth_stage": growth_stage,
                "soil_moisture": soil_moisture, "soil_type": soil_type, "irrigation_method": irrigation_method
            },
        ), 400
    except Exception as exc:
        logger.error("Smart irrigation advisory error: %s", exc, exc_info=True)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": "An error occurred during irrigation calculation. Please retry."}), 500
        return render_template(
            "irrigation.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            error="An error occurred during irrigation calculation. Please retry.",
            form_data={
                "location": clean_location, "crop": crop, "growth_stage": growth_stage,
                "soil_moisture": soil_moisture, "soil_type": soil_type, "irrigation_method": irrigation_method
            },
        ), 500


# ── Sustainability Score Routes ──────────────────────────────────────────────

@main_bp.route("/sustainability")
@main_bp.route("/sustainability-score")
def sustainability_page():
    """Render the Sustainability Score dashboard."""
    return render_template(
        "sustainability.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        is_model_live=disease_service.is_checkpoint_available(),
        is_crop_model_live=crop_service.is_model_available(),
        is_weather_service_live=weather_service.is_live_configured(),
        is_irrigation_live=irrigation_service.is_live_configured(),
        is_sustainability_live=True,
        result=None,
        form_data={
            "soil_moisture": 45,
            "irrigation_method": "Drip",
            "water_availability": "Moderate",
            "nutrient_practice": "Integrated",
            "soil_cover": "Mulched",
            "crop_health_status": "Healthy",
            "crop": "Tomato",
            "growth_stage": "Vegetative",
            "location": "",
        },
    )


@main_bp.route("/calculate-sustainability", methods=["POST"])
@main_bp.route("/calculate-sustainability-score", methods=["POST"])
def calculate_sustainability_endpoint():
    """Calculate rule-based sustainability score with transparent weight renormalization."""
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    raw_moisture = data.get("soil_moisture")
    irrigation_method = data.get("irrigation_method")
    water_availability = data.get("water_availability", "Moderate")
    nutrient_practice = data.get("nutrient_practice", "Integrated")
    soil_cover = data.get("soil_cover", "Bare_Soil")
    crop_health_status = data.get("crop_health_status", "Not_Assessed")
    crop = data.get("crop", "Tomato")
    growth_stage = data.get("growth_stage", "Vegetative")
    location = data.get("location")
    force_refresh = data.get("force_refresh", False) in (True, "true", "True", "1", 1)

    # 1. Validate required inputs
    if raw_moisture is None or str(raw_moisture).strip() == "":
        msg = "Please enter your manual soil moisture percentage (0–100%)."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": msg}), 400
        return render_template(
            "sustainability.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            is_sustainability_live=True,
            error=msg,
            form_data=data,
        ), 400

    if not irrigation_method:
        msg = "Please select an irrigation system."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": msg}), 400
        return render_template(
            "sustainability.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            is_sustainability_live=True,
            error=msg,
            form_data=data,
        ), 400

    try:
        result = sustainability_service.calculate_score(
            soil_moisture=raw_moisture,
            irrigation_method=irrigation_method,
            water_availability=water_availability,
            nutrient_practice=nutrient_practice,
            soil_cover=soil_cover,
            crop_health_status=crop_health_status,
            crop=crop,
            growth_stage=growth_stage,
            location=location,
            force_refresh=force_refresh,
        )
        _update_compact_farm_context({
            "soil_moisture": raw_moisture,
            "irrigation_method": irrigation_method,
            "water_availability": water_availability,
            "nutrient_practice": nutrient_practice,
            "soil_cover": soil_cover,
            "crop_health_status": crop_health_status,
            "crop": crop,
            "growth_stage": growth_stage,
            "location": location,
            "sustainability_score": result.get("score"),
        })

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)

        return render_template(
            "sustainability.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            is_sustainability_live=True,
            result=result,
            form_data=data,
        )
    except ValueError as val_err:
        logger.warning("Validation error in calculate_sustainability: %s", val_err)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": str(val_err)}), 400
        return render_template(
            "sustainability.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            is_sustainability_live=True,
            error=str(val_err),
            form_data=data,
        ), 400
    except Exception as exc:
        logger.error("Sustainability calculation error: %s", exc, exc_info=True)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"status": "error", "message": "An error occurred during sustainability score calculation."}), 500
        return render_template(
            "sustainability.html",
            app_name=AppConfig.APP_NAME,
            app_tagline=AppConfig.APP_TAGLINE,
            app_version=AppConfig.APP_VERSION,
            is_model_live=disease_service.is_checkpoint_available(),
            is_crop_model_live=crop_service.is_model_available(),
            is_weather_service_live=weather_service.is_live_configured(),
            is_irrigation_live=irrigation_service.is_live_configured(),
            is_sustainability_live=True,
            error="An error occurred during sustainability score calculation. Please retry.",
            form_data=data,
        ), 500


# ── Farmer Assistant (GenAI) Routes ──────────────────────────────────────────

@main_bp.route("/farmer-assistant")
@main_bp.route("/assistant")
def farmer_assistant_page():
    """Render the grounded conversational Farmer Assistant interface."""
    compact_ctx = session.get("farm_context", {})
    context_summary = farmer_assistant_service.get_context_summary(compact_ctx)
    return render_template(
        "farmer_assistant.html",
        app_name=AppConfig.APP_NAME,
        app_tagline=AppConfig.APP_TAGLINE,
        app_version=AppConfig.APP_VERSION,
        context_summary=context_summary,
        farm_context=compact_ctx,
    )


@main_bp.route("/api/farmer-assistant/context", methods=["GET"])
def api_farmer_assistant_context():
    """Return the authoritative current farm context summary."""
    compact_ctx = session.get("farm_context", {})
    context_summary = farmer_assistant_service.get_context_summary(compact_ctx)
    return jsonify({
        "status": "success",
        "ok": True,
        "context": context_summary,
        "context_summary": context_summary,
        "farm_context": compact_ctx,
    })


@main_bp.route("/api/farmer-assistant", methods=["POST"])
def api_farmer_assistant():
    """Grounded conversational API endpoint for the Farmer Assistant."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}

    raw_message = data.get("message")
    if raw_message is None:
        return jsonify({
            "status": "error",
            "ok": False,
            "message": "Message cannot be empty. Please ask a farming-related question.",
        }), 400

    message = str(raw_message).strip()
    if not message:
        return jsonify({
            "status": "error",
            "ok": False,
            "message": "Message cannot be empty. Please ask a farming-related question.",
        }), 400

    language = str(data.get("language") or "en").strip().lower()
    req_context = data.get("context")

    compact_ctx = session.get("farm_context", {})

    try:
        result = farmer_assistant_service.answer_farmer_query(
            query=message,
            language=language,
            session_context=compact_ctx,
            request_context=req_context,
        )

        # Update compact context if conversational update occurred (e.g. soil moisture, crop, etc.)
        if result.get("updated_farm_state"):
            _update_compact_farm_context(result["updated_farm_state"])

        return jsonify({
            "status": "success",
            "ok": True,
            "answer": result.get("answer", ""),
            "language": result.get("language", "en"),
            "intent": result.get("intent", "GENERAL"),
            "sources": result.get("sources", []),
            "data_status": result.get("data_status", "UNAVAILABLE"),
            "provider_mode": result.get("provider_mode", "LOCAL_RULE"),
            "provider": result.get("provider", "local"),
            "fallback_used": bool(result.get("fallback_used", False)),
            "fallback_reason": result.get("fallback_reason"),
            "mode_notice": result.get("mode_notice", "Assistant Demo · Local Rule Mode"),
            "disclaimer": result.get("disclaimer", ""),
            "context": result.get("context_summary", {}),
            "context_summary": result.get("context_summary", {}),
            "diagnostics": result.get("diagnostics", {}),
        })
    except ValueError as val_err:
        return jsonify({
            "status": "error",
            "ok": False,
            "message": str(val_err),
        }), 400
    except Exception as exc:
        logger.error("Farmer assistant processing error: %s", exc, exc_info=True)
        return jsonify({
            "status": "error",
            "ok": False,
            "message": "Assistant is temporarily unavailable. You can still use Smart Irrigation, Weather, and Sustainability modules directly.",
        }), 500
