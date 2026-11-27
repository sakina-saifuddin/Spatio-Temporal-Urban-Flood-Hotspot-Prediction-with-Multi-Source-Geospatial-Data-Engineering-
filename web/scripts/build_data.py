"""Convert the precomputed app datasets (app/data/*) into small JSON files for the web map.

Run from the repository root:
    python web/scripts/build_data.py

Outputs (web/data/):
    cells.json     one record per ~1 km grid cell: score, land features, SHAP, 311 reports per storm
    storms.json    the four storms: name and model performance
    hotspots.json  top 10 highest-scoring well-sampled cells, with a nearby place name

Nothing here retrains the model or recomputes SHAP; it only reshapes the team's validated outputs.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "app" / "data"
OUT = ROOT / "web" / "data"
NAME_CACHE = Path(__file__).resolve().parent / "place_names_cache.json"

STORMS = {2015: "Memorial Day Flood", 2016: "Tax Day Flood", 2017: "Hurricane Harvey", 2019: "Tropical Storm Imelda"}
HOTSPOT_MIN_SAMPLES = 5  # ignore thinly sampled cells when ranking hotspots


def write(name: str, data) -> None:
    path = OUT / name
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:,.0f} KB")


def build_cells(grid: pd.DataFrame, hist: pd.DataFrame) -> list[dict]:
    # 311 report counts per cell per storm -> {"2015": 3, "2017": 12, ...} (only non-zero)
    reports = (
        hist[hist.report_positive_count > 0]
        .groupby("grid_id")
        .apply(lambda d: {str(y): int(c) for y, c in zip(d.year, d.report_positive_count)}, include_groups=False)
        .to_dict()
    )
    cells = []
    for r in grid.itertuples():
        cells.append({
            "id": r.grid_id,
            "lat": round(r.center_lat, 4),           # cell center; cells are 0.01° squares
            "lon": round(r.center_lon, 4),
            "score": round(r.mean_score, 3),         # mean susceptibility score (0–1)
            "n": int(r.sample_count),                # sampled locations in the cell
            "elev": round(r.mean_elevation_m, 1),
            "slope": round(r.mean_slope_degrees, 2),
            "bayou": round(r.mean_distance_to_bayou_m),
            "hwy": round(r.mean_paved_pct, 2),       # share of census tract near major highways (0–1)
            "land": r.dominant_land_cover.replace("Developed, ", ""),
            "reports": reports.get(r.grid_id, {}),
            # SHAP: how much each feature pushed the score up (+) or down (−), in log-odds
            "why": {
                "elev": round(r.shap_elevation_m, 2),
                "slope": round(r.shap_slope_degrees, 2),
                "bayou": round(r.shap_distance_to_bayou_m, 2),
                "hwy": round(r.shap_paved_pct, 2),
                "land": round(r.shap_land_cover_type, 2),
            },
        })
    return cells


def build_storms() -> list[dict]:
    metrics = pd.read_csv(SRC / "storm_metrics.csv").set_index("year")
    preds = pd.read_parquet(SRC / "historical_predictions.parquet", columns=["year", "lat", "lon", "y_true"])
    storms = []
    for year, name in STORMS.items():
        m = metrics.loc[year]
        reported = preds[(preds.year == year) & (preds.y_true == 1)]
        storms.append({
            "year": year,
            "name": name,
            "test": m.evaluation_protocol,           # "leave_one_storm_out" or "independent_test"
            "reports": int(m.report_positive_count),
            "caught": int(m.tp),                     # reported spots the model flagged (score >= 0.5)
            "recall": round(float(m.recall), 3),
            "accuracy": round(float(m.accuracy), 3),
            "precision": round(float(m.precision), 3),  # of the spots the model flagged, share that had a report
            "flagged": int(m.predicted_positive_count),
            # one dot per reported flood location in the cleaned research data: [lat, lon]
            "dots": [[round(a, 4), round(b, 4)] for a, b in zip(reported.lat, reported.lon)],
        })
    return storms


def place_name(lat: float, lon: float, cache: dict) -> str:
    """Reverse-geocode a cell center to a neighborhood name (OpenStreetMap Nominatim), cached on disk."""
    key = f"{lat:.4f},{lon:.4f}"
    if key in cache:
        return cache[key]
    query = urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 15})
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/reverse?{query}",
        headers={"User-Agent": "houston-flood-map-build/1.0"},
    )
    name = ""
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            addr = json.load(resp).get("address", {})
        parts = [addr.get(k) for k in ("neighbourhood", "suburb", "quarter", "road") if addr.get(k)]
        name = parts[0] if parts else ""
    except Exception as exc:  # offline or rate-limited: fall back to coordinates
        print(f"    reverse geocode failed for {key}: {exc}")
    time.sleep(1.1)  # Nominatim usage policy: max 1 request per second
    cache[key] = name
    return name


def build_hotspots(grid: pd.DataFrame) -> list[dict]:
    cache = json.loads(NAME_CACHE.read_text(encoding="utf-8")) if NAME_CACHE.exists() else {}
    top = grid[grid.sample_count >= HOTSPOT_MIN_SAMPLES].nlargest(10, "mean_score")
    hotspots = [
        {"id": r.grid_id, "lat": r.center_lat, "lon": r.center_lon, "score": round(r.mean_score, 3),
         "name": place_name(r.center_lat, r.center_lon, cache)}
        for r in top.itertuples()
    ]
    NAME_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return hotspots


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(SRC / "grid_summary.parquet")
    hist = pd.read_parquet(SRC / "historical_grid_summary.parquet")
    print("Writing:")
    write("cells.json", build_cells(grid, hist))
    write("storms.json", build_storms())
    write("hotspots.json", build_hotspots(grid))


if __name__ == "__main__":
    main()
