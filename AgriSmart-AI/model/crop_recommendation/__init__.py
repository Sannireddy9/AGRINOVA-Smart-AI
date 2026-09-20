"""
AgriSmart AI — Crop Recommendation Machine Learning Module
===========================================================
Dedicated tabular baseline for precision crop suitability ranking.
"""

from model.crop_recommendation.config import CropRecommendationConfig
from model.crop_recommendation.predict import predict

__all__ = ["CropRecommendationConfig", "predict"]
