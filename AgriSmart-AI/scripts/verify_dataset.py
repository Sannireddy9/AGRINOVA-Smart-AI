#!/usr/bin/env python
"""
AgriSmart AI — Dataset Verification Script
===========================================
Alias for scripts/check_dataset.py.
Verifies that the PlantVillage dataset is properly structured and ready.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --sample 50
    python scripts/verify_dataset.py --dataset-root "path/to/PlantVillage"
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.check_dataset import main

if __name__ == "__main__":
    main()
