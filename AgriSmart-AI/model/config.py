"""
AgriSmart AI — Model Configuration
===================================
Central configuration for model hyperparameters, dataset paths,
training settings, and inference options.

All configurable values should be defined here so that train.py,
predict.py, evaluate.py, and dataset.py import from a single source
of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "report"
MODEL_SAVE_DIR = PROJECT_ROOT / "model" / "checkpoints"

# ──────────────────────────────────────────────
# Dataset Paths
# ──────────────────────────────────────────────
# PlantVillage dataset root — each sub-folder is one class.
# Example:  data/raw/PlantVillage/Tomato___Late_blight/
#           data/raw/PlantVillage/Potato___healthy/
DATASET_ROOT = RAW_DATA_DIR / "PlantVillage"

# Allowed image file extensions (case-insensitive check in code)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Class-to-index mapping produced by the dataset preparation step
CLASS_INDEX_FILE = PROCESSED_DATA_DIR / "class_index.json"

# ──────────────────────────────────────────────
# Dataset Split
# ──────────────────────────────────────────────
VALIDATION_SPLIT = 0.2       # 80 % train, 20 % validation
STRATIFIED_SPLIT = True      # Maintain class proportions in each split

# ──────────────────────────────────────────────
# Image Validation
# ──────────────────────────────────────────────
VERIFY_IMAGES = True         # Attempt to open every image during preparation
CORRUPTED_LOG_FILE = REPORT_DIR / "corrupted_images.log"

# ──────────────────────────────────────────────
# Training Hyperparameters (placeholder defaults)
# ──────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 4
DEVICE = "cuda"  # "cuda" or "cpu"
DROPOUT_RATE = 0.3

# ──────────────────────────────────────────────
# Early Stopping & LR Scheduler
# ──────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 7    # epochs without val-F1 improvement
SCHEDULER_FACTOR = 0.5         # LR reduction factor on plateau
SCHEDULER_PATIENCE = 3         # epochs before LR reduction

# ──────────────────────────────────────────────
# Normalisation (ImageNet pre-trained defaults)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Model Settings
# ──────────────────────────────────────────────
MODEL_NAME = "resnet50"  # Placeholder — will be finalised later
PRETRAINED = True

# ──────────────────────────────────────────────
# SIH Class-Count Expectation (advisory only)
# ──────────────────────────────────────────────
# The SIH 2026 problem statement specifies ~15–20 classes + healthy.
# This range is used ONLY for a non-blocking warning; it does NOT
# filter or reject classes.  The exact class list is provided by
# the organisers at kickoff.
EXPECTED_CLASS_RANGE = (10, 25)  # inclusive advisory bounds

# ──────────────────────────────────────────────
# Checkpoint Versioning
# ──────────────────────────────────────────────
CHECKPOINT_VERSION = "1.0"  # bumped when checkpoint format changes

# ──────────────────────────────────────────────
# Miscellaneous
# ──────────────────────────────────────────────
RANDOM_SEED = 42
