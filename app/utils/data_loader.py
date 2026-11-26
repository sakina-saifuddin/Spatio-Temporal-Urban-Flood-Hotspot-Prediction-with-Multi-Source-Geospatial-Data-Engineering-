"""Cached loaders for the validated, precomputed FloodSense datasets.

Large datasets use ``st.cache_resource`` so every session shares one
in-memory copy (``st.cache_data`` would deserialize a fresh copy on every
call). Callers must treat these objects as read-only; all transformations in
the app create new frames (``assign``/``merge``/boolean filtering).

The application never fits models or computes SHAP values at runtime. Every
loader here reads files produced by ``scripts/prepare_app_data.py`` (or saved
research reports) and returns them unchanged, apart from read-only derived
helpers that are clearly named.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TUNING_DIR = REPORTS_DIR / "hyperparameter_tuning"
MASCOT_PATH = REPO_ROOT / "assets" / "images" / "mascot.png"
CSS_PATH = APP_DIR / "styles" / "custom.css"

DATA_FILES = {
    "locations": "location_predictions.parquet",
    "grid": "grid_summary.parquet",
    "boundaries": "grid_boundaries.geojson",
    "historical": "historical_predictions.parquet",
    "historical_grid": "historical_grid_summary.parquet",
    "storm_metrics": "storm_metrics.csv",
    "global_shap": "global_shap_importance.csv",
    "metadata": "model_metadata.json",
}

GROUPED_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "distance_to_bayou_m",
    "paved_pct",
    "land_cover_type",
]
SHAP_COLUMNS = [f"shap_{name}" for name in GROUPED_FEATURES]

STORM_ORDER = [
    "Memorial Day Flood",
    "Tax Day Flood",
    "Hurricane Harvey",
    "Tropical Storm Imelda",
]


class DataFileMissing(FileNotFoundError):
    """Raised when a required application data file is unavailable."""


def data_path(key: str) -> Path:
    """Return the absolute path for a named application dataset."""
    path = DATA_DIR / DATA_FILES[key]
    if not path.exists():
        raise DataFileMissing(
            f"Required FloodSense data file is missing: app/data/{DATA_FILES[key]}"
        )
    return path


def missing_data_files() -> list[str]:
    """List application data files that are not present."""
    return [name for name in DATA_FILES.values() if not (DATA_DIR / name).exists()]


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    with data_path("metadata").open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading sampled locations…")
def load_locations() -> pd.DataFrame:
    return pd.read_parquet(data_path("locations"))


@st.cache_resource(show_spinner="Loading geographic summaries…")
def load_grid() -> pd.DataFrame:
    return pd.read_parquet(data_path("grid"))


@st.cache_resource(show_spinner="Loading grid boundaries…")
def load_boundaries() -> dict[str, Any]:
    with data_path("boundaries").open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading historical observations…")
def load_historical() -> pd.DataFrame:
    return pd.read_parquet(data_path("historical"))


@st.cache_resource(show_spinner="Loading historical storm summaries…")
def load_historical_grid() -> pd.DataFrame:
    return pd.read_parquet(data_path("historical_grid"))


@st.cache_data(show_spinner=False)
def load_storm_metrics() -> pd.DataFrame:
    frame = pd.read_csv(data_path("storm_metrics"))
    order = {name: index for index, name in enumerate(STORM_ORDER)}
    return frame.sort_values("storm_name", key=lambda s: s.map(order)).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_global_shap() -> pd.DataFrame:
    return pd.read_csv(data_path("global_shap"))


@st.cache_data(show_spinner=False)
def load_report_csv(relative_path: str) -> pd.DataFrame | None:
    """Read a saved research report CSV (read-only). Returns None if absent."""
    path = REPORTS_DIR / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def grid_index() -> pd.DataFrame:
    """Development grid summary indexed by ``grid_id`` for fast lookup."""
    return load_grid().set_index("grid_id", drop=False)


def figure_path(name: str) -> Path | None:
    path = FIGURES_DIR / name
    return path if path.exists() else None
