"""Page 2 — Explore Houston. Chapter 2: Discover Your Neighborhood."""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from components import charts, maps
from components.cards import html_table, stat_cards
from components.disclaimers import render_context_note, render_disclaimer_banner
from components.mascot import (
    explain_cell,
    explain_coverage,
    explain_no_data,
    explain_search,
    scout_says,
)
from components.navigation import chapter_header
from utils import geocoding
from utils.data_loader import GROUPED_FEATURES, grid_index, load_grid
from utils.grid_utils import (
    DEFAULT_CENTER,
    DEFAULT_ZOOM,
    FEATURE_LABELS,
    compute_grid_id,
    factor_label,
    format_int,
    format_paved,
    format_score,
    format_share,
    grid_bounds,
    grid_center,
    is_valid_grid_id,
    nearby_sampled_cells,
    score_word,
    validate_coordinates,
)

ss = st.session_state
ss.setdefault("explore_selected", None)  # dict: grid_id, lat, lon, label, source
ss.setdefault("explore_compare", {"A": None, "B": None})
ss.setdefault("explore_map_key", 0)
ss.setdefault("explore_view", {"center": DEFAULT_CENTER, "zoom": DEFAULT_ZOOM})
ss.setdefault("explore_last_click", None)
ss.setdefault("explore_places", [])
ss.setdefault("explore_show_nearby", False)
ss.setdefault("explore_layer", "susceptibility")

grid = load_grid()
gindex = grid_index()
sampled_ids = set(gindex.index)


layer_geojson = maps.cached_development_geojson


def _recenter(lat: float, lon: float, zoom: int = 13) -> None:
    """Move the map (used for searches/typed selections, not for map clicks)."""
    ss.explore_view = {"center": (lat, lon), "zoom": zoom}
    ss.explore_map_key += 1


def select_point(lat: float, lon: float, label: str, source: str, recenter: bool = True) -> None:
    gid = compute_grid_id(lat, lon)
    ss.explore_selected = {"grid_id": gid, "lat": lat, "lon": lon, "label": label, "source": source}
    ss.explore_show_nearby = False
    if recenter:
        _recenter(lat, lon)


def select_cell(gid: str, source: str) -> None:
    lat, lon = grid_center(gid)
    ss.explore_selected = {"grid_id": gid, "lat": None, "lon": None, "label": f"Cell {gid}", "source": source,
                           "center": (lat, lon)}
    ss.explore_show_nearby = False
    _recenter(lat, lon)


def reset_view() -> None:
    ss.explore_selected = None
    ss.explore_places = []
    ss.explore_last_click = None
    ss.explore_show_nearby = False
    ss.explore_view = {"center": DEFAULT_CENTER, "zoom": DEFAULT_ZOOM}
    ss.explore_map_key += 1


chapter_header(
    "explore",
    "Explore Houston",
    "Search a public place, enter coordinates, or click the map to inspect a geographic cell and see which "
    "environmental features drove the model's estimate.",
)
render_disclaimer_banner()

# ---------------------------------------------------------------------------
# Search controls
# ---------------------------------------------------------------------------
config = geocoding.provider_config()
tab_place, tab_coord, tab_cell = st.tabs(["🔎 Public place search", "📍 Coordinates", "🔢 Grid ID or list"])

with tab_place:
    st.caption(
        f"Place search sends the words you type to an external service ({config['provider']}). Use it only for "
        "public places such as parks, landmarks, or intersections. **For a private home, click the map or enter "
        "coordinates instead** — those are matched to a cell inside FloodSense and are never sent to the place-search "
        f"service. Like any web map, the background map images are loaded from the {maps.TILE_PROVIDER}, which "
        "receive standard requests for the map area on screen. FloodSense does not save your searches."
    )
    if not config["enabled"]:
        st.info("External place search is turned off for this deployment. Use map selection or coordinates.")
    with st.form("place_search", clear_on_submit=False, border=False):
        query = st.text_input("Public place name", placeholder="e.g. Buffalo Bayou Park, Houston TX", max_chars=200,
                              disabled=not config["enabled"])
        submitted = st.form_submit_button("Search place", type="primary", disabled=not config["enabled"])
    if submitted:
        with st.spinner("Searching…"):
            result = geocoding.geocode(query)
        if result.ok:
            ss.explore_places = [p.__dict__ for p in result.places]
        else:
            ss.explore_places = []
            st.warning(result.error)
    if ss.explore_places:
        options = list(range(len(ss.explore_places)))
        choice = st.radio(
            f"{len(options)} match{'es' if len(options) != 1 else ''} found — choose one:",
            options,
            format_func=lambda i: ss.explore_places[i]["name"],
        )
        if st.button("Show this place on the map", key="use_place"):
            place = ss.explore_places[choice]
            select_point(place["lat"], place["lon"], place["name"], "search")
            st.rerun()
    st.caption(config["attribution"])

with tab_coord:
    with st.form("coord_search", border=False):
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (decimal degrees)", value=None, min_value=-90.0, max_value=90.0,
                              format="%.6f", placeholder="29.760400")
        lon = c2.number_input("Longitude (decimal degrees)", value=None, min_value=-180.0, max_value=180.0,
                              format="%.6f", placeholder="-95.369800")
        go = st.form_submit_button("Find cell", type="primary")
    if go:
        error = validate_coordinates(lat, lon)
        if error:
            st.warning(error)
        else:
            select_point(float(lat), float(lon), f"{lat:.5f}, {lon:.5f}", "coordinates")
            st.rerun()

with tab_cell:
    st.caption("A keyboard-friendly alternative to clicking the map.")
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("grid_form", border=False):
            gid_text = st.text_input("Grid ID", placeholder="2976_-9537")
            go_gid = st.form_submit_button("Open cell")
        if go_gid:
            gid_text = gid_text.strip()
            if not is_valid_grid_id(gid_text):
                st.warning("Grid IDs look like `2976_-9537` (latitude×100 _ longitude×100).")
            else:
                select_cell(gid_text, "grid id")
                st.rerun()
    with c2:
        ordered = grid.sort_values("mean_score", ascending=False)["grid_id"].tolist()
        pick = st.selectbox(
            "Or browse sampled cells (sorted by mean score)", ordered, index=None,
            format_func=lambda g: f"{g} — score {gindex.at[g, 'mean_score']:.3f}, {gindex.at[g, 'sample_count']} sampled",
            placeholder="Choose a sampled cell…",
        )
        if pick and st.button("Open selected cell", key="open_browse"):
            select_cell(pick, "list")
            st.rerun()

# ---------------------------------------------------------------------------
# Map — the base map only re-renders when the view is deliberately moved
# (search, typed selection, reset). Layer changes and map clicks update the
# cell layer in place, so the user's pan/zoom and selection are kept.
# ---------------------------------------------------------------------------
layer_keys = list(maps.EXPLORE_LAYERS)
layer = st.radio("Map layer", layer_keys, format_func=lambda k: maps.EXPLORE_LAYERS[k], horizontal=True,
                 key="explore_layer")
st.markdown(maps.legend_row(layer), unsafe_allow_html=True)

selected = ss.explore_selected
fmap = maps.base_map(ss.explore_view["center"], ss.explore_view["zoom"], scroll_zoom=False)
cells_group = folium.FeatureGroup(name="FloodSense cells")
maps.add_grid_layer(
    cells_group, layer_geojson(layer), maps.EXPLORE_LAYERS[layer],
    extra_fields=["coverage_category", "dominant_factor", "reports"],
    extra_aliases=["Coverage", "Strongest upward factor", "Report-associated obs. (2015–17)"],
)
for slot, color in (("A", "#8C3F08"), ("B", "#7C3AED")):
    gid = ss.explore_compare[slot]
    if gid:
        maps.add_comparison(cells_group, grid_bounds(gid), f"Comparison area {slot}: {gid}", color)
if selected:
    marker = (selected["lat"], selected["lon"]) if selected.get("lat") is not None else None
    maps.add_selection(cells_group, grid_bounds(selected["grid_id"]), marker, selected["label"])
state = st_folium(
    fmap,
    key=f"explore_map_{ss.explore_map_key}",
    height=560,
    use_container_width=True,
    returned_objects=["last_clicked"],
    feature_group_to_add=cells_group,
)

ctrl_l, ctrl_r = st.columns([4, 1], vertical_alignment="center")
with ctrl_l:
    st.caption("Click a cell to inspect it (hover shows grid ID, mean score, and sample count). Use the +/− buttons, "
               "double-click, or pinch to zoom — scrolling the page won't zoom the map. Unsampled areas are not shaded.")
with ctrl_r:
    st.button("↺ Reset map", on_click=reset_view, width="stretch", help="Clear the selection and return to the full study area.")
render_context_note("grid")

if selected:
    gid = selected["grid_id"]
    if gid in sampled_ids:
        row = gindex.loc[gid]
        summary = (f"mean score <b>{row['mean_score']:.3f}</b> · {int(row['sample_count'])} sampled location"
                   f"{'s' if int(row['sample_count']) != 1 else ''} · {row['coverage_category']}")
    else:
        summary = "<b>no data</b> — no sampled locations in this cell"
    st.markdown(
        f'<div class="fs-selected" role="status" aria-live="polite">📍 <span><b>Selected: cell {gid}</b> — {summary}</span>'
        f'<a href="#cell-details">Details and SHAP explanation below ↓</a></div>',
        unsafe_allow_html=True,
    )

clicked = (state or {}).get("last_clicked")
if clicked and clicked != ss.explore_last_click:
    ss.explore_last_click = clicked
    select_point(float(clicked["lat"]), float(clicked["lng"]),
                 f"Map point {clicked['lat']:.5f}, {clicked['lng']:.5f}", "map", recenter=False)
    st.rerun()

# ---------------------------------------------------------------------------
# Selected area
# ---------------------------------------------------------------------------
st.divider()
if not selected:
    scout_says(
        "Pick a spot to get started: click a cell on the map, search a public place, or type coordinates. "
        "I'll explain the model's estimate for that cell and what's behind it.",
        title="Where should we look?",
    )
else:
    gid = selected["grid_id"]
    if gid not in sampled_ids:
        st.subheader(f"Cell {gid} — no data", anchor="cell-details")
        if selected["source"] == "search":
            scout_says(explain_search(selected["label"], gid, False))
        scout_says(explain_no_data(gid), title="Outside sampled coverage")
        if st.button("Explore nearby sampled cells (different areas)", key="nearby_btn"):
            ss.explore_show_nearby = True
        if ss.explore_show_nearby:
            lat0, lon0 = (selected["lat"], selected["lon"]) if selected.get("lat") is not None else selected["center"]
            nearby = nearby_sampled_cells(lat0, lon0, grid)
            if nearby.empty:
                st.info("No sampled cells lie within 10 km of this point. It is outside the research study area.")
            else:
                st.caption("These are **different cells** from the one you selected; their estimates do not apply to your point.")
                for row in nearby.itertuples():
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{row.grid_id}** — {row.distance_km:.1f} km away · mean score {row.mean_score:.3f} · "
                                f"{row.sample_count} sampled · {row.coverage_category}")
                    if c2.button("Open", key=f"near_{row.grid_id}"):
                        select_cell(row.grid_id, "nearby")
                        st.rerun()
    else:
        row = gindex.loc[gid]
        st.subheader(f"Cell {gid}", anchor="cell-details")
        if selected["source"] == "search":
            scout_says(explain_search(selected["label"], gid, True))
        st.markdown(
            f'<span class="fs-pill">{row["coverage_category"]}</span>'
            f'<span class="fs-pill">{score_word(row["mean_score"])} score</span>',
            unsafe_allow_html=True,
        )
        stat_cards([
            (format_score(row["mean_score"]), "Mean susceptibility score"),
            (f"{row['min_score']:.3f} – {row['max_score']:.3f}", "Sampled score range (min – max)"),
            (format_int(row["sample_count"]), "Sampled locations"),
            (format_share(row["share_above_cutoff"]), "Share of sampled locations at or above 0.50"),
            (format_int(row["development_report_positive_count"]), "Report-associated observations (2015–2017)",
             "Cleaned development observations linked to a 311 flood report — not a complete count of reports."),
            (format_int(row["development_observation_count"]), "Development observations (2015–2017)"),
            (factor_label(row["dominant_positive_factor"]), "Strongest upward factor"),
            (factor_label(row["dominant_negative_factor"]), "Strongest downward factor"),
        ], compact=True)

        scout_says(explain_cell(row) + "<br><br>" + explain_coverage(row), title="Scout's read on this cell")

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.shap_bar(row, "Why the model scored this cell this way"), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                f"Orange bars (▲) raise the model's raw output; blue bars (▼) lower it. Values are mean signed SHAP "
                f"contributions in raw log-odds across {int(row['sample_count'])} sampled location(s). Baseline "
                f"{row['mean_shap_base_value']:.3f} + contributions = mean raw score {row['mean_raw_model_score']:.3f}. "
                "They explain the mean raw output, not the mean probability directly, and are not causal effects."
            )
            with st.expander("SHAP values as a table"):
                table = charts.shap_frame(row).sort_values("contribution", ascending=False)
                html_table([(r.label, f"{r.contribution:+.3f}", r.direction) for r in table.itertuples()],
                           ["Feature", "Mean SHAP (log-odds)", "Direction"])
        with right:
            st.markdown("**Environmental features (cell averages)**")
            html_table(
                [
                    ("Elevation", f"{row['mean_elevation_m']:.1f} m"),
                    ("Terrain slope", f"{row['mean_slope_degrees']:.2f}°"),
                    ("Distance to bayou", f"{row['mean_distance_to_bayou_m']:,.0f} m"),
                    ("Highway proximity proxy", format_paved(row["mean_paved_pct"])),
                    ("Most common land cover type", row["dominant_land_cover"]),
                ],
                ["Feature", "Value"],
            )
            st.caption("The highway proximity proxy (`paved_pct`, stored as a 0–1 fraction) is the share of the census tract "
                       "near major highways — a research proxy, not a measure of all paved or impervious surfaces.")
            render_context_note("scores", "coverage", "in_sample")
            b1, b2 = st.columns(2)
            if b1.button("Set as Area A", key="cmpA", width="stretch"):
                ss.explore_compare["A"] = gid
                st.rerun()
            if b2.button("Set as Area B", key="cmpB", width="stretch"):
                ss.explore_compare["B"] = gid
                st.rerun()

# ---------------------------------------------------------------------------
# Area comparison
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Compare two areas")
cmp = ss.explore_compare
c1, c2 = st.columns(2)
ordered_ids = grid.sort_values("grid_id")["grid_id"].tolist()
for slot, col in (("A", c1), ("B", c2)):
    with col:
        current = cmp[slot]
        choice = st.selectbox(f"Area {slot}", ordered_ids, index=ordered_ids.index(current) if current in ordered_ids else None,
                              placeholder="Choose a sampled cell…", key=f"cmp_select_{slot}_{current}")
        if choice != current and choice is not None:
            cmp[slot] = choice
            st.rerun()
if st.button("Clear comparison"):
    ss.explore_compare = {"A": None, "B": None}
    st.rerun()

if cmp["A"] and cmp["B"]:
    a, b = gindex.loc[cmp["A"]], gindex.loc[cmp["B"]]
    stat_cards([
        (format_score(a["mean_score"]), f"Area A · {cmp['A']} — mean susceptibility score"),
        (format_score(b["mean_score"]), f"Area B · {cmp['B']} — mean susceptibility score"),
    ])
    rows = [
        ("Mean susceptibility score", format_score(a["mean_score"]), format_score(b["mean_score"])),
        ("Sampled locations", format_int(a["sample_count"]), format_int(b["sample_count"])),
        ("Coverage", a["coverage_category"], b["coverage_category"]),
        ("Share at or above 0.50", format_share(a["share_above_cutoff"]), format_share(b["share_above_cutoff"])),
        ("Mean elevation", f"{a['mean_elevation_m']:.1f} m", f"{b['mean_elevation_m']:.1f} m"),
        ("Mean terrain slope", f"{a['mean_slope_degrees']:.2f}°", f"{b['mean_slope_degrees']:.2f}°"),
        ("Mean distance to bayou", f"{a['mean_distance_to_bayou_m']:,.0f} m", f"{b['mean_distance_to_bayou_m']:,.0f} m"),
        ("Highway proximity proxy", format_paved(a["mean_paved_pct"]), format_paved(b["mean_paved_pct"])),
        ("Most common land cover type", a["dominant_land_cover"], b["dominant_land_cover"]),
        ("Strongest upward factor", factor_label(a["dominant_positive_factor"]), factor_label(b["dominant_positive_factor"])),
        ("Strongest downward factor", factor_label(a["dominant_negative_factor"]), factor_label(b["dominant_negative_factor"])),
    ]
    html_table(rows, ["Measure", f"Area A · {cmp['A']}", f"Area B · {cmp['B']}"])
    st.plotly_chart(charts.shap_compare(a, b, f"A · {cmp['A']}", f"B · {cmp['B']}"), width="stretch",
                    config={"displayModeBar": False})
    diff = a["mean_score"] - b["mean_score"]
    higher = "A" if diff > 0 else "B"
    gaps = {f: float(a[f"shap_{f}"] - b[f"shap_{f}"]) for f in GROUPED_FEATURES}
    key_feature = max(gaps, key=lambda f: abs(gaps[f]))
    scout_says(
        f"Area <b>{higher}</b> has the higher mean score (difference {abs(diff):.3f}). The largest gap in SHAP "
        f"contributions is <b>{FEATURE_LABELS[key_feature].lower()}</b> ({gaps[key_feature]:+.2f} log-odds, A minus B). "
        "Remember to weigh coverage: a cell with one sampled location is a thinner summary than one with many.",
        title="Comparing the two areas",
    )
elif cmp["A"] or cmp["B"]:
    st.info("Choose a second area to compare. Use the selectors above or the “Set as Area A/B” buttons on a selected cell.")
else:
    st.caption("Select two sampled cells to compare scores, coverage, environmental features, and SHAP contributions.")
