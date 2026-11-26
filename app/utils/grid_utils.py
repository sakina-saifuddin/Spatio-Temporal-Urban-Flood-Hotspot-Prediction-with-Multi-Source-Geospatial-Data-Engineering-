"""Geographic grid matching and shared display vocabulary.

The research grid assigns each coordinate to ``round(lat / 0.01)`` and
``round(lon / 0.01)`` using pandas/NumPy rounding (round half to even), joined
with an underscore. Matching here reproduces that convention exactly and never
substitutes a neighbouring cell silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_CELL_DEGREES = 0.01
HALF_CELL = GRID_CELL_DEGREES / 2

# Study-area extent from model_metadata.json (sampled coordinates).
DEFAULT_CENTER = (29.80, -95.40)
DEFAULT_ZOOM = 10

# Human-readable names used across the interface; technical column names are
# still shown alongside them in research-oriented sections.
FEATURE_LABELS = {
    "elevation_m": "Elevation",
    "slope_degrees": "Terrain slope",
    "distance_to_bayou_m": "Distance to bayou",
    "paved_pct": "Highway proximity proxy",
    "land_cover_type": "Land cover type",
}
FEATURE_SHORT_LABELS = FEATURE_LABELS
FEATURE_UNITS = {
    "elevation_m": "m",
    "slope_degrees": "°",
    "distance_to_bayou_m": "m",
    "paved_pct": "share",
    "land_cover_type": "",
}

# Five equal-width score intervals used consistently on every score map.
SCORE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SCORE_COLORS = ["#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"]
SCORE_LABELS = ["0.00–0.20", "0.20–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]
SCORE_WORDS = ["Very low", "Low", "Moderate", "High", "Very high"]

COVERAGE_ORDER = ["Single sampled location", "Limited coverage", "Greater coverage"]
COVERAGE_COLORS = {
    "Single sampled location": "#C6DBEF",
    "Limited coverage": "#6BAED6",
    "Greater coverage": "#103D80",
}
NO_DATA_LABEL = "No data"

FACTOR_COLORS = {
    "elevation_m": "#2865DF",
    "paved_pct": "#8C3F08",
    "slope_degrees": "#46B8E9",
    "distance_to_bayou_m": "#7C3AED",
    "land_cover_type": "#F8D04B",
    "None": "#CBD5E1",
}

CLASS_COLORS = {"TP": "#15803D", "TN": "#64748B", "FP": "#D97706", "FN": "#DC2626"}
CLASS_LABELS = {
    "TP": "True positive",
    "TN": "True negative",
    "FP": "False positive",
    "FN": "False negative",
}
CLASS_EXPLANATIONS = {
    "TP": "Scored at or above 0.50 and associated with a recorded report.",
    "TN": "Scored below 0.50 with no associated report. This does not show the location stayed dry.",
    "FP": "Scored at or above 0.50 with no associated report. Flooding may still have gone unreported.",
    "FN": "Scored below 0.50 but associated with a recorded report — a missed report.",
}

REPORT_BINS = [0, 1, 3, 10, 50, math.inf]
REPORT_COLORS = ["#EFF6FF", "#BFDBFE", "#60A5FA", "#2563EB", "#103D80"]
REPORT_LABELS = ["0", "1–2", "3–9", "10–49", "50 or more"]


def compute_grid_id(lat: float, lon: float) -> str:
    """Return the research grid ID for a coordinate (ties round half to even)."""
    lat_cell = int(np.round(np.float64(lat) / GRID_CELL_DEGREES))
    lon_cell = int(np.round(np.float64(lon) / GRID_CELL_DEGREES))
    return f"{lat_cell}_{lon_cell}"


def grid_center(grid_id: str) -> tuple[float, float]:
    lat_cell, lon_cell = (int(part) for part in grid_id.split("_"))
    return round(lat_cell * GRID_CELL_DEGREES, 6), round(lon_cell * GRID_CELL_DEGREES, 6)


def grid_bounds(grid_id: str) -> dict[str, float]:
    lat, lon = grid_center(grid_id)
    return {
        "south": round(lat - HALF_CELL, 6),
        "north": round(lat + HALF_CELL, 6),
        "west": round(lon - HALF_CELL, 6),
        "east": round(lon + HALF_CELL, 6),
    }


def is_valid_grid_id(text: str) -> bool:
    parts = str(text).strip().split("_")
    if len(parts) != 2:
        return False
    try:
        int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return True


def validate_coordinates(lat: float | None, lon: float | None) -> str | None:
    """Return an error message for invalid coordinates, else None."""
    if lat is None or lon is None:
        return "Enter both a latitude and a longitude."
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return "Coordinates must be finite numbers."
    if not -90 <= lat <= 90:
        return "Latitude must be between -90 and 90."
    if not -180 <= lon <= 180:
        return "Longitude must be between -180 and 180."
    return None


def coverage_category(sample_count: int) -> str:
    if sample_count <= 0:
        return NO_DATA_LABEL
    if sample_count == 1:
        return "Single sampled location"
    if sample_count <= 4:
        return "Limited coverage"
    return "Greater coverage"


def score_bin_index(score: float | None) -> int | None:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    for index in range(len(SCORE_BINS) - 1):
        upper = SCORE_BINS[index + 1]
        if score < upper or index == len(SCORE_BINS) - 2:
            return index
    return None


def score_color(score: float | None) -> str | None:
    index = score_bin_index(score)
    return None if index is None else SCORE_COLORS[index]


def score_word(score: float) -> str:
    index = score_bin_index(score)
    return "No data" if index is None else SCORE_WORDS[index]


def report_color(count: int) -> str:
    for index in range(len(REPORT_BINS) - 1):
        if count < REPORT_BINS[index + 1]:
            return REPORT_COLORS[index]
    return REPORT_COLORS[-1]


def factor_label(name: str | None) -> str:
    if name is None or str(name) in {"None", "nan", ""}:
        return "None"
    return FEATURE_LABELS.get(str(name), str(name))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def nearby_sampled_cells(
    lat: float, lon: float, grid: pd.DataFrame, count: int = 5, max_km: float = 10.0
) -> pd.DataFrame:
    """Nearest sampled cells by centre distance — an explicit, separate action."""
    distances = haversine_km(lat, lon, grid["center_lat"].to_numpy(), grid["center_lon"].to_numpy())
    result = grid.assign(distance_km=distances)
    result = result[result["distance_km"] <= max_km].nsmallest(count, "distance_km")
    return result


@dataclass(frozen=True)
class CellMatch:
    """Outcome of strict grid matching for a coordinate."""

    lat: float
    lon: float
    grid_id: str
    sampled: bool


def match_coordinate(lat: float, lon: float, sampled_ids) -> CellMatch:
    grid_id = compute_grid_id(lat, lon)
    return CellMatch(lat=lat, lon=lon, grid_id=grid_id, sampled=grid_id in sampled_ids)


def format_score(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


def format_share(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def format_paved(value: float | None) -> str:
    """paved_pct is stored as a 0–1 fraction of the tract near major highways."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.0f}% of tract near major highways"


def format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"
