"""Folium map builders. Polygons come from the validated grid boundaries
(development) or from each storm's own cell bounds (historical)."""

from __future__ import annotations

import copy
from typing import Any

import folium
import pandas as pd
import streamlit as st
from folium.plugins import DualMap

from utils.grid_utils import (
    CLASS_COLORS,
    CLASS_LABELS,
    COVERAGE_COLORS,
    COVERAGE_ORDER,
    DEFAULT_CENTER,
    DEFAULT_ZOOM,
    FACTOR_COLORS,
    REPORT_COLORS,
    REPORT_LABELS,
    SCORE_COLORS,
    SCORE_LABELS,
    factor_label,
    report_color,
    score_color,
)

# CartoDB basemaps now require an API key; the standard OpenStreetMap tiles are
# used with their built-in attribution (light, interactive use only).
TILES = "OpenStreetMap"
EXPLORE_LAYERS = {
    "susceptibility": "Flood Susceptibility",
    "coverage": "Data Coverage",
    "factor": "Dominant Contributing Factor",
    "reports": "Historical Flood Reports (2015–2017 development data)",
}


TILE_PROVIDER = "OpenStreetMap standard tile servers (tile.openstreetmap.org)"
DEFAULT_OPACITY = 0.6
EXCLUDED_FILL = "#CBD5E1"


def base_map(center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM, scroll_zoom: bool = True) -> folium.Map:
    """``scroll_zoom=False`` stops accidental zooming while scrolling the page;
    the +/− controls, double-click, and touch pinch-zoom still work."""
    fmap = folium.Map(location=list(center), zoom_start=zoom, tiles=TILES, control_scale=True, prefer_canvas=True,
                      scrollWheelZoom=scroll_zoom)
    return fmap


def _fill_for(layer: str, props: dict) -> str:
    if layer == "susceptibility":
        return score_color(props["mean_score"]) or "#FFFFFF"
    if layer == "coverage":
        return COVERAGE_COLORS.get(props["coverage_category"], "#FFFFFF")
    if layer == "factor":
        return FACTOR_COLORS.get(str(props.get("dominant_positive_factor")), FACTOR_COLORS["None"])
    if layer == "reports":
        return report_color(int(props.get("development_report_positive_count", 0)))
    raise ValueError(layer)


def development_geojson(boundaries: dict, grid: pd.DataFrame, layer: str) -> dict[str, Any]:
    """Attach display properties for one layer to a copy of the boundaries."""
    lookup = grid.set_index("grid_id")[
        ["mean_score", "sample_count", "coverage_category", "dominant_positive_factor", "development_report_positive_count"]
    ].to_dict("index")
    features = []
    for feature in boundaries["features"]:
        gid = feature["properties"]["grid_id"]
        info = lookup.get(gid)
        if info is None:
            continue
        props = {
            "grid_id": gid,
            "mean_score": round(float(info["mean_score"]), 3),
            "sample_count": int(info["sample_count"]),
            "coverage_category": info["coverage_category"],
            "dominant_factor": factor_label(info["dominant_positive_factor"]),
            "reports": int(info["development_report_positive_count"]),
        }
        props["fill"] = _fill_for(layer, {**info, **props})
        features.append({"type": "Feature", "id": gid, "properties": props, "geometry": feature["geometry"]})
    return {"type": "FeatureCollection", "features": features}


def subset_geojson(geojson: dict, ids, fill: str | None = None) -> dict:
    """Keep only features whose id is in ``ids`` (optionally recolouring them)."""
    keep = set(ids)
    features = []
    for feature in geojson["features"]:
        if feature["id"] in keep:
            if fill:
                feature = {**feature, "properties": {**feature["properties"], "fill": fill}}
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def add_excluded_layer(fmap, geojson: dict) -> None:
    """Sampled cells hidden by planner filters: light grey, distinct from unsampled (unshaded) areas."""
    if not geojson["features"]:
        return
    folium.GeoJson(
        geojson,
        name="Sampled cells outside current filters",
        style_function=lambda f: {"fillColor": EXCLUDED_FILL, "color": "#94A3B8", "weight": 0.2, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["grid_id", "mean_score", "sample_count"],
                                      aliases=["Grid ID (outside filters)", "Mean score", "Sampled locations"]),
        smooth_factor=0,
    ).add_to(fmap)


@st.cache_resource(show_spinner="Preparing map layer…", max_entries=len(EXPLORE_LAYERS))
def cached_development_geojson(layer: str) -> dict[str, Any]:
    """One shared, read-only GeoJSON per layer for all pages and sessions."""
    from utils.data_loader import load_boundaries, load_grid

    return development_geojson(load_boundaries(), load_grid(), layer)


def add_grid_layer(fmap, geojson: dict, name: str, opacity: float = DEFAULT_OPACITY, extra_fields=None, extra_aliases=None) -> None:
    fields = ["grid_id", "mean_score", "sample_count"] + list(extra_fields or [])
    aliases = ["Grid ID", "Mean score", "Sampled locations"] + list(extra_aliases or [])
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda f: {
            "fillColor": f["properties"]["fill"],
            "color": "#475569",
            "weight": 0.25,
            "fillOpacity": opacity,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#103D80"},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=False),
        smooth_factor=0,
    ).add_to(fmap)


def add_selection(fmap, bounds: dict | None, marker: tuple[float, float] | None = None,
                  label: str = "Selected location", color: str = "#103D80") -> None:
    if bounds:
        folium.Rectangle(
            bounds=[[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]],
            color=color, weight=3, fill=False, dash_array="6 4", tooltip="Selected cell",
        ).add_to(fmap)
    if marker:
        folium.Marker(location=list(marker), tooltip=label, icon=folium.Icon(color="blue", icon="info-sign")).add_to(fmap)


def add_comparison(fmap, bounds: dict, label: str, color: str) -> None:
    folium.Rectangle(
        bounds=[[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]],
        color=color, weight=3, fill=False, tooltip=label,
    ).add_to(fmap)


def add_rank_labels(fmap, ranked: pd.DataFrame, limit: int = 100) -> None:
    for row in ranked.head(limit).itertuples():
        folium.Marker(
            location=[row.center_lat, row.center_lon],
            icon=folium.DivIcon(html=f'<div class="fs-rank-pin">{row.rank}</div>', icon_size=(22, 22), icon_anchor=(11, 11)),
            tooltip=f"Rank {row.rank}: {row.grid_id}",
        ).add_to(fmap)


# ---------------------------------------------------------------------------
# Historical layers built from each storm's own cell bounds.
# ---------------------------------------------------------------------------

HISTORICAL_LAYERS = {
    "predicted": "Predicted susceptibility (mean saved score)",
    "reported": "Recorded report-associated share",
    "classification": "Prediction vs. report (classification share)",
}


def historical_geojson(cells: pd.DataFrame, layer: str, focus_class: str = "FN") -> dict[str, Any]:
    features = []
    for row in cells.itertuples(index=False):
        if layer == "predicted":
            fill, opacity = score_color(row.mean_score), 0.75
        elif layer == "reported":
            fill, opacity = score_color(row.report_positive_share), 0.75
        else:
            share = float(getattr(row, f"{focus_class.lower()}_share"))
            fill, opacity = CLASS_COLORS[focus_class], (0.08 + 0.8 * share) if share > 0 else 0.03
        features.append(
            {
                "type": "Feature",
                "id": row.grid_id,
                "properties": {
                    "grid_id": row.grid_id,
                    "mean_score": round(float(row.mean_score), 3),
                    "sample_count": int(row.sample_count),
                    "report_share": f"{row.report_positive_share * 100:.0f}%",
                    "mix": f"TP {row.tp_count} · TN {row.tn_count} · FP {row.fp_count} · FN {row.fn_count}",
                    "fill": fill,
                    "opacity": round(opacity, 3),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [row.west, row.south], [row.east, row.south], [row.east, row.north],
                        [row.west, row.north], [row.west, row.south],
                    ]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _historical_layer(target, geojson: dict, name: str) -> None:
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda f: {
            "fillColor": f["properties"]["fill"],
            "color": "#475569",
            "weight": 0.25,
            "fillOpacity": f["properties"]["opacity"],
        },
        highlight_function=lambda f: {"weight": 2, "color": "#103D80"},
        tooltip=folium.GeoJsonTooltip(
            fields=["grid_id", "mean_score", "report_share", "sample_count", "mix"],
            aliases=["Grid ID", "Mean saved score", "Report-associated share", "Sampled observations", "Classifications"],
        ),
        smooth_factor=0,
    ).add_to(target)


def historical_map(geojson: dict, name: str, center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM) -> folium.Map:
    fmap = base_map(center, zoom)
    _historical_layer(fmap, geojson, name)
    return fmap


def historical_dual_map(left: dict, right: dict, left_name: str, right_name: str,
                        center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM) -> DualMap:
    dual = DualMap(location=list(center), zoom_start=zoom, tiles=TILES, layout="horizontal")
    _historical_layer(dual.m1, left, left_name)
    _historical_layer(dual.m2, copy.deepcopy(right), right_name)
    return dual


# ---------------------------------------------------------------------------
# Legends rendered as accessible HTML next to the map (not inside it), so they
# never obstruct the map and remain readable by screen readers.
# ---------------------------------------------------------------------------

def _swatches(items: list[tuple[str, str]], title: str, note: str = "") -> str:
    rows = "".join(
        f'<li><span class="fs-swatch" style="background:{color}" aria-hidden="true"></span>{label}</li>'
        for color, label in items
    )
    note_html = f'<p class="fs-legend-note">{note}</p>' if note else ""
    return f'<div class="fs-legend" role="group" aria-label="{title}"><p class="fs-legend-title">{title}</p><ul>{rows}</ul>{note_html}</div>'


NO_DATA_ITEM = ("transparent; border:1px dashed #94A3B8", "No data (unsampled — not shaded)")
EXCLUDED_ITEM = (EXCLUDED_FILL, "Sampled cell outside current filters (grey)")


def legend_html(layer: str, include_excluded: bool = False) -> str:
    html = _legend_html(layer)
    if include_excluded:
        swatch = (f'<li><span class="fs-swatch" style="background:{EXCLUDED_ITEM[0]}" aria-hidden="true"></span>'
                  f'{EXCLUDED_ITEM[1]}</li>')
        html = html.replace("</ul>", swatch + "</ul>", 1)
    return html


def legend_row(layer: str, include_excluded: bool = False) -> str:
    """Legend laid out horizontally above a full-width map (wraps on narrow screens)."""
    return f'<div class="fs-legend-row">{legend_html(layer, include_excluded)}</div>'


def _legend_html(layer: str) -> str:
    if layer in {"susceptibility", "predicted"}:
        items = list(zip(SCORE_COLORS, [f"{label}" for label in SCORE_LABELS])) + [NO_DATA_ITEM]
        return _swatches(items, "Mean susceptibility score", "Relative model score, not a probability of future flooding.")
    if layer == "reported":
        items = list(zip(SCORE_COLORS, [f"{label} share" for label in SCORE_LABELS])) + [NO_DATA_ITEM]
        return _swatches(items, "Share of sampled observations with a recorded report", "Not a complete inventory of flooding.")
    if layer == "coverage":
        items = [(COVERAGE_COLORS[c], c) for c in COVERAGE_ORDER] + [NO_DATA_ITEM]
        return _swatches(items, "Sampling coverage", "1 location · 2–4 limited · 5+ greater. Not prediction confidence.")
    if layer == "factor":
        items = [(FACTOR_COLORS[k], factor_label(k)) for k in FACTOR_COLORS if k != "None"]
        items += [(FACTOR_COLORS["None"], "No positive contribution"), NO_DATA_ITEM]
        return _swatches(items, "Strongest upward SHAP contribution (cell mean)", "Model attribution, not a cause of flooding.")
    if layer == "reports":
        items = list(zip(REPORT_COLORS, REPORT_LABELS)) + [NO_DATA_ITEM]
        return _swatches(items, "Report-associated development observations, 2015–2017",
                         "Cleaned, sampled research observations — not all flood reports.")
    if layer in CLASS_COLORS:
        return _swatches([(CLASS_COLORS[layer], f"{CLASS_LABELS[layer]} — darker = larger share of the cell"), NO_DATA_ITEM],
                         "Classification share within each cell")
    raise ValueError(layer)
