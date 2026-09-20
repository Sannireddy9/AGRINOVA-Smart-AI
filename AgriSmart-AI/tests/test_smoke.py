"""
Smoke tests — verify the project imports and app factory work.
"""

from app.main import create_app


def test_health_endpoint():
    """The /health endpoint should return status ok."""
    app = create_app()
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
