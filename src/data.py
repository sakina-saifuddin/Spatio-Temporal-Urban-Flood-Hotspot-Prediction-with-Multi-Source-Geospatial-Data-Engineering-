"""Functions for loading and cleaning FloodSense training data."""

from math import isfinite
from pathlib import Path

import pandas as pd

from src.config import (
    CLEANING_LOG_PATH,
    CLEAN_TRAINING_PATH,
    EXCLUDED_FEATURES,
    TRAINING_PATH,
)


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a project CSV and report its shape and per-column missing values."""
    df = pd.read_csv(path)
    print(f"{label} shape: {df.shape}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())
    return df


def load_training(path: Path | str | None = None) -> pd.DataFrame:
    """Load the training CSV and report its shape and per-column missing values."""
    if path is None:
        source_path = TRAINING_PATH
        duplicated_extension_path = TRAINING_PATH.with_suffix(
            f"{TRAINING_PATH.suffix}.csv"
        )
        if not source_path.exists() and duplicated_extension_path.exists():
            source_path = duplicated_extension_path
    else:
        source_path = Path(path)

    return _load_csv(source_path, "Training data")


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Load the cleaned training CSV and report its shape and missing values."""
    source_path = CLEAN_TRAINING_PATH if path is None else Path(path)
    return _load_csv(source_path, "Cleaned training data")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated location-and-storm rows and report how many were removed."""
    cleaned = df.drop_duplicates(subset=["lat", "lon", "storm_name"]).copy()
    print(f"Duplicate rows removed: {len(df) - len(cleaned)}")
    return cleaned


def keep_developed_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every flooded row and only developed-land non-flooded rows."""
    print("Class counts before filtering:")
    print(df["flooded"].value_counts(dropna=False).sort_index().to_string())

    developed = df["land_cover_type"].str.startswith("Developed", na=False)
    keep_rows = df["flooded"].eq(1) | (df["flooded"].eq(0) & developed)
    cleaned = df.loc[keep_rows].copy()

    print("Class counts after filtering:")
    print(cleaned["flooded"].value_counts(dropna=False).sort_index().to_string())
    return cleaned


def drop_nondeveloped_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove flooded rows outside developed land cover and report the count."""
    developed = df["land_cover_type"].str.startswith("Developed", na=False)
    remove_rows = df["flooded"].eq(1) & ~developed
    removed_count = int(remove_rows.sum())
    cleaned = df.loc[~remove_rows].copy()
    print(f"Non-developed flooded rows removed: {removed_count}")
    return cleaned


def add_grid_cell(df: pd.DataFrame, cell_deg: float = 0.01) -> pd.DataFrame:
    """Assign each coordinate to a rounded geographic grid cell."""
    if not isfinite(cell_deg) or cell_deg <= 0:
        raise ValueError("cell_deg must be a finite positive number.")

    gridded = df.copy()
    lat_cell = (gridded["lat"] / cell_deg).round().astype("Int64").astype("string")
    lon_cell = (gridded["lon"] / cell_deg).round().astype("Int64").astype("string")
    gridded["grid_id"] = lat_cell + "_" + lon_cell
    return gridded


def clean_training(df: pd.DataFrame) -> pd.DataFrame:
    """Clean rows, log filtering counts, add grid IDs, and save the result."""
    log_rows: list[dict[str, int | str]] = []
    cleaned = df.copy()

    for step, transform in (
        ("remove_duplicates", remove_duplicates),
        ("keep_developed_negatives", keep_developed_negatives),
        ("drop_nondeveloped_positives", drop_nondeveloped_positives),
    ):
        rows_before = len(cleaned)
        cleaned = transform(cleaned)
        log_rows.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_removed": rows_before - len(cleaned),
                "rows_after": len(cleaned),
            }
        )

    cleaned = add_grid_cell(cleaned)
    cleaned = cleaned.drop(columns=list(EXCLUDED_FEATURES))
    cleaned.to_csv(CLEAN_TRAINING_PATH, index=False)
    pd.DataFrame(log_rows).to_csv(CLEANING_LOG_PATH, index=False)
    print(f"Saved cleaned training data to {CLEAN_TRAINING_PATH}")
    print(f"Saved cleaning log to {CLEANING_LOG_PATH}")
    return cleaned
