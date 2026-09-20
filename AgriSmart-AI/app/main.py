"""
AgriSmart AI — Application Entry Point
========================================
Creates and configures the Flask application instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from flask import Flask, jsonify

from app.config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app(config_class=AppConfig) -> Flask:
    """Application factory for AgriSmart AI.

    Args:
        config_class: Configuration class to load settings from.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # 1. Configuration
    app.config.from_object(config_class)
    config_class.ensure_upload_dir()

    # 2. Register Blueprints
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    # 3. Health check endpoint (for monitoring and smoke tests)
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "app": config_class.APP_NAME})

    # 4. Error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return (
            jsonify({
                "status": "error",
                "message": "File size exceeds the 16 MB limit. Please upload a smaller image.",
            }),
            413,
        )

    @app.errorhandler(404)
    def not_found_error(error):
        return (
            jsonify({"status": "error", "message": "The requested resource was not found."}),
            404,
        )

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error: %s", error)
        return (
            jsonify({
                "status": "error",
                "message": "An internal error occurred while processing your request.",
            }),
            500,
        )

    logger.info("AgriSmart AI application initialized.")
    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=True,
    )
