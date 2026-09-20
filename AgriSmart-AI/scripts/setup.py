"""
AgriSmart AI — Setup Script
=============================
Utility script for project environment and dataset validation.

Usage:
    python scripts/setup.py
"""

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.check_dataset import check_dataset
from model import config


def main():
    print("🌾 AgriSmart AI — Setup & Dataset Verification")
    print("-" * 50)
    print(f"Checking dataset at: {config.DATASET_ROOT}")
    check_dataset(config.DATASET_ROOT)


if __name__ == "__main__":
    main()
