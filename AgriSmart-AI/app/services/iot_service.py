"""
AgriSmart AI — IoT / Sensor Service (Future Bonus Module)
==========================================================
Interface placeholder for ground sensors (NPK, moisture, canopy temp).
"""

from __future__ import annotations
from typing import Any


class IoTTelemetryService:
    """Service boundary for IoT device telemetry."""

    def ingest_reading(self, device_id: str, telemetry_payload: dict[str, Any]) -> bool:
        """Ingest sensor data from field node."""
        raise NotImplementedError("IoT telemetry module is planned for a future phase.")

    def get_latest_telemetry(self, field_id: str) -> dict[str, Any]:
        """Fetch latest soil and microclimate metrics."""
        raise NotImplementedError("IoT telemetry module is planned for a future phase.")
