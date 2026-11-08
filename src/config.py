"""Shared project paths and settings."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

for directory in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRAINING_PATH = PROCESSED_DIR / "training_final.csv"
CLEAN_TRAINING_PATH = PROCESSED_DIR / "training_clean.csv"
CLEANING_LOG_PATH = REPORTS_DIR / "cleaning_log.csv"

SEED = 42
TARGET = "flooded"
NUMERIC_FEATURES = (
    "elevation_m",
    "slope_degrees",
    "distance_to_bayou_m",
    "paved_pct",
    "rainfall_inches",
)
CATEGORICAL_FEATURES = ("land_cover_type",)
DROP_COLS = ("year", "storm_name", "storm_date", "weight", "lat", "lon")
EXCLUDED_FEATURES = {"imperviousness": "unverified source, looks synthetic"}
