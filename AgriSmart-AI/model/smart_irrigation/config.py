"""
AgriSmart AI — Smart Irrigation Advisor Configuration & Thresholds
==================================================================
Centralized, configurable heuristic thresholds and agronomic context dictionaries
for the Smart Irrigation Advisor.

NOTICE:
These thresholds and context notes are configurable decision-support heuristics.
Actual irrigation requirements vary by crop species, variety, root depth,
soil texture, organic matter, climate, evapotranspiration, irrigation infrastructure,
and local agricultural practices. This module does NOT claim scientifically calibrated
crop water coefficients, numerical multipliers, or precise water-volume predictions.
Zero-IoT Architecture: Soil moisture is strictly a farmer-provided manual observation.
"""

from __future__ import annotations
from typing import Any


class SmartIrrigationConfig:
    """Smart Irrigation specific configuration and heuristic thresholds."""

    # ── Soil Moisture Thresholds (Heuristic %) ─────────────────────────────
    # Configurable module-specific thresholds (independent of other modules)
    SOIL_MOISTURE_CRITICAL: float = 25.0    # At or below: Root zone at severe water deficit
    SOIL_MOISTURE_DEPLETED: float = 35.0    # At or below: Management allowed depletion range
    SOIL_MOISTURE_ADEQUATE: float = 65.0    # Above: Sufficient available soil moisture

    # ── Meteorological Heuristic Thresholds ────────────────────────────────
    IMMINENT_RAIN_PROB_THRESHOLD: float = 60.0      # Probability considered likely (>= 60%)
    IMMINENT_RAIN_AMOUNT_THRESHOLD: float = 8.0     # Meaningful rainfall accumulation (>= 8.0 mm)
    LIGHT_RAIN_AMOUNT_THRESHOLD: float = 2.0        # Negligible / light shower threshold (< 2.0 mm)
    TRANSITIONAL_RAIN_PROB_LOW: float = 30.0        # Lower bound of transitional rain window
    TRANSITIONAL_RAIN_PROB_HIGH: float = 59.0       # Upper bound of transitional rain window
    FUTURE_RAIN_EVENT_THRESHOLD: float = 15.0       # Significant upcoming rainfall in days 3–7 (mm)
    HEAT_STRESS_THRESHOLD: float = 35.0             # Ambient temperature accelerating evapotranspiration (°C)

    # ── Qualitative Crop Phenological Sensitivity Context ──────────────────
    # Pure qualitative agronomic guidance to inform farmer scouting.
    # No numerical irrigation multipliers, coefficients, or volume claims.
    GROWTH_STAGE_CONTEXT: dict[str, str] = {
        "Vegetative": (
            "Vegetative growth stage: Plants are developing root structure and foliage canopy. "
            "Maintain adequate, stable moisture to encourage root depth while avoiding waterlogging."
        ),
        "Flowering": (
            "Flowering / Reproductive stage: Highly sensitive to moisture fluctuations. "
            "Water deficit during pollination and flowering can reduce blossom set and fruit initiation. "
            "Monitor soil moisture closely."
        ),
        "Fruiting": (
            "Fruiting / Yield formation stage: Consistent moisture supports cell expansion and fruit filling. "
            "Avoid sharp wet-dry cycles that can induce physiological defects such as fruit splitting or blossom-end rot."
        ),
        "Maturity": (
            "Maturity / Ripening stage: Transpiration rates often moderate as foliage senesces. "
            "Excessive moisture near harvest can compromise quality, delay dry-down, or promote post-harvest pathogens."
        ),
    }

    CROP_SENSITIVITY_CONTEXT: dict[str, str] = {
        "Tomato": (
            "Solanaceous crop with moderate root depth. Sensitive to irregular watering during flowering and fruit set. "
            "Consistent moisture management helps prevent physiological stress."
        ),
        "Potato": (
            "Shallow-rooted tuber crop sensitive to moisture fluctuations. Tuber bulking requires uniform moisture; "
            "waterlogging promotes soft rot and tuber blight."
        ),
        "Rice": (
            "High water-demand crop commonly grown in bunded paddies or alternate wetting and drying (AWD). "
            "Adjust ponding depth or interval according to crop stage and local canal availability."
        ),
        "Wheat": (
            "Cereal crop with critical moisture windows at crown root initiation (CRI), tillering, flowering, and grain filling. "
            "Ensure adequate root-zone moisture during active reproductive phases."
        ),
        "Maize": (
            "Deep-rooting cereal highly sensitive to water stress at tasseling, silking, and grain fill. "
            "Moisture deficits during pollination can severely affect ear development."
        ),
        "Cotton": (
            "Deep-rooted commercial crop. High water requirement occurs between squaring and boll development; "
            "excess water late in the season can induce unwanted vegetative regrowth."
        ),
        "Sugarcane": (
            "Long-duration crop with substantial biomass. Peak water requirements correspond with formative and grand growth phases. "
            "Ensure regular irrigation cycles during peak vegetative elongation."
        ),
    }

    # ── Qualitative Irrigation Method Context ──────────────────────────────
    IRRIGATION_METHOD_CONTEXT: dict[str, str] = {
        "Drip": (
            "Drip Micro-Irrigation selected: Precise localized root-zone application minimizes surface evaporation. "
            "Adjust daily fertigation run times according to the rainfall forecast."
        ),
        "Sprinkler": (
            "Overhead Sprinkler selected: Application subject to wind drift and canopy interception. "
            "Avoid operating during periods of high wind or active precipitation to minimize evaporative loss."
        ),
        "Flood": (
            "Surface / Furrow / Flood irrigation selected: Higher volume application with surface runoff potential. "
            "Ensure fields have adequate surface drainage before forecast rainfall to avoid waterlogging."
        ),
        "Rainfed": (
            "Rainfed cultivation selected (no supplemental irrigation system): Operations depend entirely on precipitation. "
            "Focus field operations on soil moisture conservation, mulching, and rainwater harvesting."
        ),
    }

    # ── Transparency & Disclaimers ─────────────────────────────────────────
    METHODOLOGY_DISCLAIMER: str = (
        "The Smart Irrigation Advisor operates as a deterministic, rule-based decision-support tool. "
        "It combines manual farmer field estimates with meteorological forecasts from Open-Meteo. "
        "Configurable heuristic thresholds (Critical: 25%, Depleted: 35%, Adequate: 65%) are provided "
        "for advisory guidance. Actual irrigation schedules depend on crop variety, soil profile, "
        "rooting depth, evapotranspiration, and local agronomic practices. This tool does not control "
        "hardware or predict exact irrigation volumes."
    )

    ZERO_IOT_NOTICE: str = (
        "Zero-IoT Architecture: Soil moisture percentage is a manual estimate entered by the farmer. "
        "This platform does not require, connect to, or simulate physical microcontrollers, ESP32 modules, "
        "or IoT sensor hardware."
    )

    DEMO_WEATHER_WARNING: str = (
        "DEMO WEATHER — IRRIGATION ADVISORY IS BASED ON SIMULATED WEATHER. "
        "Live meteorological service was unavailable; displaying simulated forecast for demonstration. "
        "Verify with actual on-site field conditions before irrigating."
    )

    WATER_SAVING_STATEMENT: str = (
        "The advisor helps avoid unnecessary irrigation cycles when adequate soil moisture coincides "
        "with forecast precipitation. Always align watering decisions with local extension guidance."
    )

    @classmethod
    def get_crop_context(cls, crop_name: str) -> str:
        """Return qualitative agronomic context for the given crop."""
        clean = crop_name.strip().title()
        return cls.CROP_SENSITIVITY_CONTEXT.get(
            clean,
            f"General field crop context for {clean}: maintain balanced root-zone moisture and monitor growth stage."
        )

    @classmethod
    def get_stage_context(cls, stage_name: str) -> str:
        """Return qualitative agronomic context for the given growth stage."""
        clean = stage_name.strip().title()
        return cls.GROWTH_STAGE_CONTEXT.get(
            clean,
            f"Growth stage '{clean}': observe canopy development and adjust irrigation frequency to field drying cycles."
        )

    @classmethod
    def get_method_context(cls, method_name: str) -> str:
        """Return qualitative operational context for the irrigation delivery method."""
        clean = method_name.strip().title()
        return cls.IRRIGATION_METHOD_CONTEXT.get(
            clean,
            f"Irrigation delivery method '{clean}': review delivery efficiency in light of forecast precipitation."
        )
