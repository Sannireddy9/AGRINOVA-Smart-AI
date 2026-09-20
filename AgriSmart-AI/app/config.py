"""
AgriSmart AI — Web Application Configuration
=============================================
Central configuration settings for the Flask web application.
Environment variables override default settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Centralized authoritative .env loading for the application
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.is_file():
    load_dotenv(ENV_FILE, override=False)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MODEL_CHECKPOINT = PROJECT_ROOT / "model" / "checkpoints" / "best_model.pt"


class AppConfig:
    """Base application configuration."""

    # Security
    SECRET_KEY: str = os.environ.get(
        "SECRET_KEY", "agrismart-ai-dev-secret-key-change-in-production"
    )

    # File Uploads
    UPLOAD_FOLDER: Path = Path(os.environ.get("UPLOAD_FOLDER", str(UPLOAD_DIR)))
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB max image size
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "webp", "bmp"}

    # Model Integration
    CHECKPOINT_PATH: Path = Path(
        os.environ.get("CHECKPOINT_PATH", str(MODEL_CHECKPOINT))
    )

    # Server Settings
    HOST: str = os.environ.get("HOST", "127.0.0.1")
    PORT: int = int(os.environ.get("PORT", 5050))

    # Weather Intelligence
    WEATHER_API_KEY: str | None = os.environ.get("WEATHER_API_KEY", None)
    WEATHER_CACHE_TTL_MINUTES: int = int(os.environ.get("WEATHER_CACHE_TTL_MINUTES", 15))
    WEATHER_MODE: str = os.environ.get("WEATHER_MODE", "auto")

    # UI / App Info
    APP_NAME: str = "AgriSmart AI"
    APP_TAGLINE: str = "Intelligent Agriculture for a Sustainable Future"
    APP_VERSION: str = "0.1.0-alpha"

    @classmethod
    def ensure_upload_dir(cls) -> Path:
        """Ensure upload directory exists and return it."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        return cls.UPLOAD_FOLDER
