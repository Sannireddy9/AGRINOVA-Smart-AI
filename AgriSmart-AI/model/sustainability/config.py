"""
AgriSmart AI — Sustainability Score Configuration
==================================================
Centralized configuration, heuristic weights, qualitative scoring tiers,
allowed input enumerations, and transparency notices for the Sustainability Score module.

NOTICE:
- All weights and scoring tiers are project-defined heuristics for this prototype.
- This module is strictly a deterministic advisory index, NOT an official
  agricultural standard or scientific carbon/LCA certification.
- No physical sensor dependency (Zero-IoT compliant).
- Missing components are dynamically excluded with remaining weights renormalized.
"""

from __future__ import annotations

from typing import Any


class SustainabilityConfig:
    """Configuration and heuristic constants for the Sustainability Score engine."""

    # ──────────────────────────────────────────────────────────────────────────
    # BASE COMPONENT WEIGHTS (sum = 1.00)
    # ──────────────────────────────────────────────────────────────────────────
    WEIGHT_WATER_EFFICIENCY: float = 0.40  # 40%
    WEIGHT_RESOURCE_USE: float = 0.30     # 30%
    WEIGHT_CROP_HEALTH: float = 0.30      # 30%

    # ──────────────────────────────────────────────────────────────────────────
    # SCORE CATEGORY BANDS (0 - 100)
    # ──────────────────────────────────────────────────────────────────────────
    CATEGORY_STRONG_THRESHOLD: float = 80.0
    CATEGORY_MODERATE_THRESHOLD: float = 60.0
    CATEGORY_NEEDS_IMPROVEMENT_THRESHOLD: float = 40.0

    @classmethod
    def get_category_meta(cls, score: float) -> dict[str, str]:
        """Map a 0-100 numerical score to a farmer-friendly qualitative category."""
        if score >= cls.CATEGORY_STRONG_THRESHOLD:
            return {
                "category": "Strong",
                "severity": "success",
                "badge_class": "badge-strong",
                "description": "Demonstrates balanced, resource-conserving agricultural practices.",
            }
        if score >= cls.CATEGORY_MODERATE_THRESHOLD:
            return {
                "category": "Moderate",
                "severity": "info",
                "badge_class": "badge-moderate",
                "description": "Satisfactory resource efficiency with practical opportunities for enhancement.",
            }
        if score >= cls.CATEGORY_NEEDS_IMPROVEMENT_THRESHOLD:
            return {
                "category": "Needs Improvement",
                "severity": "warning",
                "badge_class": "badge-warning",
                "description": "Key resource-management practices require active review to optimize field stewardship.",
            }
        return {
            "category": "Low",
            "severity": "danger",
            "badge_class": "badge-low",
            "description": "Elevated resource stress or significant inefficiencies detected; prioritize remedial field action.",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # ALLOWED INPUT DOMAINS (Strict Validation)
    # ──────────────────────────────────────────────────────────────────────────
    ALLOWED_IRRIGATION_METHODS: set[str] = {
        "Drip",
        "Sprinkler",
        "Flood",
        "Rainfed",
    }

    ALLOWED_WATER_AVAILABILITIES: set[str] = {
        "Abundant",
        "Moderate",
        "Scarce",
        "Rainfed",
    }

    ALLOWED_NUTRIENT_PRACTICES: set[str] = {
        "Organic",
        "Integrated",
        "Moderate_Chemical",
        "Intensive_Chemical",
        "Synthetic_Heavy",
    }

    ALLOWED_SOIL_COVERS: set[str] = {
        "Cover_Crops",
        "Mulch",
        "Mulched",
        "Minimum_Tillage",
        "Bare_Soil",
        "None",
        "Bare",
    }

    ALLOWED_CROP_HEALTH_STATUSES: set[str] = {
        "Healthy",
        "Minor_Stress",
        "Mild_Stress",
        "Disease_Detected",
        "Moderate_Disease",
        "Severe_Damage",
        "Severe_Disease",
        "Not_Assessed",
    }

    # Contextual crop options (does not alter deterministic sustainability score)
    COMMON_CROPS: list[str] = [
        "Rice (Paddy)",
        "Wheat",
        "Maize (Corn)",
        "Cotton",
        "Sugarcane",
        "Tomato",
        "Potato",
        "Onion",
        "Soybean",
        "Groundnut",
        "Chickpea",
        "Pigeon Pea",
        "Mustard",
        "Sorghum",
        "Pearl Millet",
        "Banana",
        "Mango",
        "Other / Not Listed",
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # BASE HEURISTIC COMPONENT SCORES
    # ──────────────────────────────────────────────────────────────────────────
    # Irrigation delivery method baseline efficiency
    IRRIGATION_BASE_SCORES: dict[str, int] = {
        "Drip": 90,       # High application uniformity, minimal evaporative loss
        "Sprinkler": 75,  # Moderate efficiency, evaporative/wind drift exposure
        "Flood": 50,      # High percolation and surface runoff losses
        "Rainfed": 85,    # Zero supplemental water extraction / pumping energy
    }

    # Nutrient management baseline scores
    NUTRIENT_BASE_SCORES: dict[str, int] = {
        "Organic": 95,             # Bio-fertilizers / compost; preserves soil biodiversity
        "Integrated": 85,          # INM: Balanced combination of organic & targeted mineral
        "Moderate_Chemical": 70,   # Standard synthetic fertilizer with split doses
        "Intensive_Chemical": 45,  # Heavy synthetic reliance; elevated runoff/salinity risks
        "Synthetic_Heavy": 45,     # Alias for intensive chemical
    }

    # Soil cover adjustments (+/- points to resource use)
    SOIL_COVER_SCORES: dict[str, int] = {
        "Cover_Crops": 15,
        "Mulch": 10,
        "Mulched": 10,
        "Minimum_Tillage": 5,
        "Bare_Soil": 0,
        "None": 0,
        "Bare": 0,
    }

    # Crop health baseline scores
    CROP_HEALTH_SCORES: dict[str, int | None] = {
        "Healthy": 95,              # Foliar tissue healthy, no active pathogen observed
        "Minor_Stress": 75,         # Slight chlorosis / physiological stress under monitoring
        "Mild_Stress": 75,          # Alias for minor stress
        "Disease_Detected": 45,     # Active pathogen or foliar lesions identified
        "Moderate_Disease": 45,     # Alias for disease detected
        "Severe_Damage": 25,        # Advanced disease progression / extensive damage
        "Severe_Disease": 25,       # Alias for severe damage
        "Not_Assessed": None,       # Excluded from scoring; weights dynamically renormalized
    }

    # ──────────────────────────────────────────────────────────────────────────
    # TRANSPARENCY DISCLAIMER
    # ──────────────────────────────────────────────────────────────────────────
    DISCLAIMER: str = (
        "Advisory Sustainability Index based on farmer-reported field parameters and rule-based "
        "agronomic heuristics (Water: 40%, Resource: 30%, Crop Health: 30%). This score is a "
        "decision-support heuristic for educational and hackathon evaluation; it does not constitute "
        "an official agricultural certification, carbon credit validation, or scientific environmental audit."
    )
