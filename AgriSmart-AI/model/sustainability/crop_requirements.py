"""
Crop Requirements and Agronomic Context Module for Sustainability Score.

Loads and serves empirical, source-derived crop water-use parameters from
UN FAO (FAO 56, Table 22; FAO 33) and ICAR benchmarks.

IMPORTANT DESIGN INVARIANTS:
1. FAO provides the agronomic context (seasonal water requirements, depletion fraction p,
   critical growth stages).
2. Numerical score adjustments are AgriSmart project-defined alignment heuristics,
   NOT FAO points or thresholds.
3. FAO depletion fraction p represents the fraction of Total Available Soil Water (TAW)
   that can be depleted before stress. It is NEVER converted directly to soil moisture percentage
   (no optimal_moisture_min_pct = p * 100).
4. Seasonal water requirements in mm are NOT compared or divided by instantaneous soil moisture %.
5. Unsupported crops return crop_data_status = 'NOT_AVAILABLE' with zero arbitrary adjustments.
   No default, average, or nearest crops are ever assigned.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sustainability" / "crop_requirements.csv"


@dataclass(frozen=True)
class CropRequirement:
    """Agronomic crop requirement context container."""
    crop_key: str
    display_name: str
    scientific_name: str
    seasonal_water_need_mm_min: float
    seasonal_water_need_mm_max: float
    agrismart_water_need_category: str  # AgriSmart derived category: Very_High, High, Medium, Low
    fao_depletion_fraction_p: float      # FAO 56 Table 22 depletion fraction (0.20 - 0.65)
    critical_growth_stages: List[str]
    growth_stage_information: str
    source_reference: str
    crop_data_status: str = "AVAILABLE"  # 'AVAILABLE' or 'NOT_AVAILABLE'
    is_available: bool = True

    def is_stage_critical(self, stage: Optional[str]) -> bool:
        """Check if the specified growth stage is documented as moisture-stress critical."""
        if not stage or not self.is_available:
            return False
        stage_norm = stage.strip().lower().replace(" ", "_").replace("-", "_")
        for critical in self.critical_growth_stages:
            crit_norm = critical.strip().lower().replace(" ", "_").replace("-", "_")
            if stage_norm in crit_norm or crit_norm in stage_norm:
                return True
        return False


# In-memory cache for loaded requirements
_CROP_REQUIREMENTS_CACHE: Optional[Dict[str, CropRequirement]] = None

# Known aliases mapping user / dropdown inputs to canonical crop keys
CROP_ALIASES: Dict[str, str] = {
    "rice": "rice",
    "rice (paddy)": "rice",
    "paddy": "rice",
    "wheat": "wheat",
    "maize": "maize",
    "maize (corn)": "maize",
    "corn": "maize",
    "cotton": "cotton",
    "sugarcane": "sugarcane",
    "tomato": "tomato",
    "potato": "potato",
    "onion": "onion",
    "soybean": "soybean",
    "groundnut": "groundnut",
    "peanut": "groundnut",
    "chickpea": "chickpea",
    "gram": "chickpea",
    "chana": "chickpea",
    "pigeon pea": "pigeon_pea",
    "pigeon_pea": "pigeon_pea",
    "arhar": "pigeon_pea",
    "tur": "pigeon_pea",
    "mustard": "mustard",
    "rapeseed": "mustard",
    "sorghum": "sorghum",
    "jowar": "sorghum",
    "pearl millet": "pearl_millet",
    "pearl_millet": "pearl_millet",
    "bajra": "pearl_millet",
    "millet": "pearl_millet",
    "banana": "banana",
    "mango": "mango",
}


def load_crop_requirements_dataset(csv_path: Optional[Path] = None) -> Dict[str, CropRequirement]:
    """
    Load crop requirements dataset from CSV.
    Caches parsed records in memory.
    """
    global _CROP_REQUIREMENTS_CACHE
    if _CROP_REQUIREMENTS_CACHE is not None and csv_path is None:
        return _CROP_REQUIREMENTS_CACHE

    target_path = csv_path or DATA_CSV_PATH
    if not target_path.exists():
        logger.error(f"Crop requirements dataset not found at {target_path}")
        return {}

    requirements: Dict[str, CropRequirement] = {}

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_key = row["crop_key"].strip().lower()
            critical_stages = [
                s.strip()
                for s in row.get("critical_growth_stages", "").split(",")
                if s.strip()
            ]

            req = CropRequirement(
                crop_key=crop_key,
                display_name=row["display_name"].strip(),
                scientific_name=row["scientific_name"].strip(),
                seasonal_water_need_mm_min=float(row["seasonal_water_need_mm_min"]),
                seasonal_water_need_mm_max=float(row["seasonal_water_need_mm_max"]),
                agrismart_water_need_category=row["agrismart_water_need_category"].strip(),
                fao_depletion_fraction_p=float(row["fao_depletion_fraction_p"]),
                critical_growth_stages=critical_stages,
                growth_stage_information=row.get("growth_stage_information", "").strip(),
                source_reference=row.get("source_reference", "").strip(),
                crop_data_status="AVAILABLE",
                is_available=True,
            )
            requirements[crop_key] = req

    if csv_path is None:
        _CROP_REQUIREMENTS_CACHE = requirements

    return requirements


def get_crop_requirement(crop_name: Optional[str], growth_stage: Optional[str] = None) -> CropRequirement:
    """
    Retrieve source-derived crop requirement context for a given crop name.

    If crop is unknown, 'Other / Not Listed', or missing:
    Returns CropRequirement with crop_data_status='NOT_AVAILABLE' and is_available=False.
    DOES NOT assign a default crop, average crop, or nearest crop.
    """
    if not crop_name:
        return _create_unavailable_requirement("Not Specified")

    clean_name = crop_name.strip().lower()

    if clean_name in ("other", "other / not listed", "not listed", "unknown"):
        return _create_unavailable_requirement(crop_name.strip())

    dataset = load_crop_requirements_dataset()
    canonical_key = CROP_ALIASES.get(clean_name, clean_name.replace(" ", "_"))

    if canonical_key in dataset:
        return dataset[canonical_key]

    # Partial match fallback across canonical keys
    for key, req in dataset.items():
        if key in clean_name or clean_name in key:
            return req

    # Not found — strictly return NOT_AVAILABLE without fabricating defaults
    return _create_unavailable_requirement(crop_name.strip())


def _create_unavailable_requirement(crop_name: str) -> CropRequirement:
    """Creates a transparent NOT_AVAILABLE placeholder for unsupported crops."""
    return CropRequirement(
        crop_key="unsupported",
        display_name=crop_name,
        scientific_name="N/A",
        seasonal_water_need_mm_min=0.0,
        seasonal_water_need_mm_max=0.0,
        agrismart_water_need_category="Unknown",
        fao_depletion_fraction_p=0.0,
        critical_growth_stages=[],
        growth_stage_information="Crop-specific agronomic parameters are not available in the database for this crop.",
        source_reference="N/A",
        crop_data_status="NOT_AVAILABLE",
        is_available=False,
    )


def get_supported_crop_names() -> List[Dict[str, str]]:
    """Return list of supported crops with canonical keys and display names."""
    dataset = load_crop_requirements_dataset()
    items = [
        {"key": req.crop_key, "display_name": req.display_name}
        for req in dataset.values()
    ]
    items.append({"key": "other", "display_name": "Other / Not Listed"})
    return items
