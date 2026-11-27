"""Page 3 — Priority Areas. Chapter 3: Build a More Resilient Houston."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from components import charts, maps
from components.disclaimers import render_context_note, render_disclaimer_banner
from components.mascot import explain_cell, explain_coverage, explain_planner, scout_says
from components.navigation import chapter_header
from utils.data_loader import grid_index, load_grid, load_locations, load_metadata
from utils.grid_utils import (
    COVERAGE_ORDER,
    compute_grid_id,
    factor_label,
    format_int,
    format_score,
    format_share,
    grid_bounds,
    grid_center,
)
from utils.metrics import FROZEN_CUTOFF, RANKING_METRICS, rank_grids, share_at_or_above, summary_statistics

ss = st.session_state
DEFAULTS = {
    "pr_top": "All",
    "pr_min_samples": 1,
    "pr_score": (0.0, 1.0),
    "pr_coverage": list(COVERAGE_ORDER),
    "pr_metric": "mean_score",
    "pr_cutoff": FROZEN_CUTOFF,
}
for key, value in DEFAULTS.items():
    ss.setdefault(key, value)
ss.setdefault("pr_selected", None)
ss.setdefault("pr_last_click", None)
ss.setdefault("pr_map_key", 0)


def reset_filters() -> None:
    for key, value in DEFAULTS.items():
        ss[key] = value
    ss.pr_selected = None
    ss.pr_last_click = None
    ss.pr_last_row = None
    ss.pr_map_key += 1


@st.cache_data(show_spinner="Recalculating shares at the selected cutoff…")
def shares_for(cutoff: float):
    return share_at_or_above(load_locations(), cutoff)


def full_geojson() -> dict:
    """Shared susceptibility layer (same cached object as Explore); subsets are cheap filters of it."""
    return maps.cached_development_geojson("susceptibility")


metadata = load_metadata()
ts = metadata["threshold_selection"]
grid = load_grid()
gindex = grid_index()

chapter_header(
    "priority",
    "Priority Areas",
    "A planner-oriented view for finding sampled areas that may warrant further investigation, using only the "
    "model's saved outputs. It is not an official intervention, investment, or emergency-response tool.",
)
render_disclaimer_banner()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Filters and ranking**")
    f1, f2 = st.columns(2)
    with f1:
        st.selectbox("Ranking metric", list(RANKING_METRICS), format_func=RANKING_METRICS.get, key="pr_metric")
        st.segmented_control("Show", ["10", "25", "50", "100", "All"], key="pr_top", required=True,
                             help="Number of top-ranked cells to display.")
        st.number_input("Minimum sampled locations per cell", min_value=1, max_value=int(grid["sample_count"].max()),
                        step=1, key="pr_min_samples")
    with f2:
        st.slider("Mean susceptibility score range", 0.0, 1.0, step=0.01, key="pr_score")
        st.slider("Exploratory classification cutoff", 0.05, 0.95, step=0.01, key="pr_cutoff",
                  help="Used to recalculate the share of each cell's sampled locations at or above this value. "
                       "The research froze 0.50; other values are exploratory and not validated.")
    st.multiselect("Coverage categories (sampling coverage, not prediction confidence)", COVERAGE_ORDER,
                   key="pr_coverage",
                   help="Single sampled location = 1 location · Limited coverage = 2–4 · Greater coverage = 5 or more.")
    st.button("↺ Reset filters", on_click=reset_filters)

cutoff = float(ss.pr_cutoff)
if abs(cutoff - FROZEN_CUTOFF) > 1e-9:
    st.warning(
        f"Exploratory cutoff **{cutoff:.2f}** in use. Only the frozen **0.50** cutoff was selected in the research "
        f"(observation-level geographic recall {ts['area_recall']:.2%}, precision {ts['area_precision']:.2%}; "
        f"cross-storm recall {ts['storm_recall']:.2%}, precision {ts['storm_precision']:.2%}). "
        "Other cutoffs have no validated performance, and none has been validated as a grid-level cutoff.",
        icon="⚠️",
    )
else:
    st.caption(
        f"Frozen observation-level cutoff 0.50: geographic recall {ts['area_recall']:.2%} / precision "
        f"{ts['area_precision']:.2%}; cross-storm recall {ts['storm_recall']:.2%} / precision {ts['storm_precision']:.2%}. "
        "These are observation-level results, not independently validated grid-level performance."
    )

top_n = None if ss.pr_top in (None, "All") else int(ss.pr_top)
ranked_all = rank_grids(
    grid, shares_for(cutoff), metric=ss.pr_metric, min_samples=int(ss.pr_min_samples),
    score_range=tuple(ss.pr_score), coverage=list(ss.pr_coverage),
)
ranked = ranked_all.head(top_n) if top_n else ranked_all
filters_active = any(ss[k] != v for k, v in DEFAULTS.items() if k not in ("pr_metric", "pr_top", "pr_cutoff"))
summary = summary_statistics(ranked)

scout_says(explain_planner(summary, RANKING_METRICS[ss.pr_metric], cutoff, filters_active), title="Planner summary")

if ranked.empty:
    st.info("No sampled cells match the current filters. Widen the score range, lower the minimum sample count, "
            "or include more coverage categories — or press **Reset filters**.")
    st.stop()

m = st.columns(4)
m[0].metric("Cells shown", f"{summary['cells']:,}", help=f"{len(ranked_all):,} cells match the filters before the top-N limit.")
m[1].metric("Sampled locations", f"{summary['locations']:,}")
m[2].metric("Average cell score", format_score(summary["mean_score"]),
            help="Unweighted average of the displayed cells' mean susceptibility scores — each cell counts once, "
                 "regardless of how many locations were sampled in it.")
m[3].metric(f"Locations at or above {cutoff:.2f}", format_share(summary["mean_share"]),
            help="Share of all sampled locations in the displayed cells whose individual score is at or above the cutoff.")
st.caption("Average cell score is an unweighted average across cells, not a location-weighted average.")

# ---------------------------------------------------------------------------
# Map + table (synchronized through ss.pr_selected)
# ---------------------------------------------------------------------------
display_cols = {
    "rank": "Rank", "grid_id": "Grid ID", "mean_score": "Mean score", "max_score": "Max score",
    "share_at_cutoff": f"Share ≥ {cutoff:.2f}", "sample_count": "Sampled locations",
    "coverage_category": "Coverage", "dominant_positive_factor": "Strongest upward factor",
    "development_report_positive_count": "Report-assoc. obs. (2015–17)",
}
table = ranked[list(display_cols)].copy()
table["dominant_positive_factor"] = table["dominant_positive_factor"].map(factor_label)

ranked_ids = set(ranked["grid_id"])
excluded_ids = set(grid["grid_id"]) - ranked_ids
show_all_labels = st.toggle("Show rank labels for all displayed cells (up to 100)", value=False, key="pr_all_labels",
                            help="By default only the top 10 ranks are labelled to keep the map readable.")
st.markdown(maps.legend_row("susceptibility", include_excluded=bool(excluded_ids)), unsafe_allow_html=True)

with st.container():
    selected = ss.pr_selected if ss.pr_selected in ranked_ids else None
    center = grid_center(selected) if selected else (29.80, -95.40)
    fmap = maps.base_map(center, 12 if selected else 10, scroll_zoom=False)
    full = full_geojson()
    maps.add_excluded_layer(fmap, maps.subset_geojson(full, excluded_ids))
    maps.add_grid_layer(fmap, maps.subset_geojson(full, ranked_ids), "Ranked cells", opacity=0.75)
    maps.add_rank_labels(fmap, ranked, limit=100 if show_all_labels else 10)
    if selected:
        maps.add_selection(fmap, grid_bounds(selected))
    state = st_folium(fmap, key=f"pr_map_{ss.pr_map_key}", height=500, use_container_width=True,
                      returned_objects=["last_clicked"])
    st.caption(
        "Coloured cells match the current filters. Grey cells were sampled but are outside the current filters or top-N "
        "limit — they are not 'no data'. Unshaded areas were never sampled. Numbered pins mark the top "
        f"{'100' if show_all_labels else '10'} ranks. Click a coloured cell or a table row to inspect it; use +/− or "
        "pinch to zoom."
    )

with st.container():
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=420, key=f"pr_table_{ss.pr_map_key}",
        on_select="rerun", selection_mode="single-row",
        column_config={
            "mean_score": st.column_config.NumberColumn("Mean score", format="%.3f"),
            "max_score": st.column_config.NumberColumn("Max score", format="%.3f"),
            "share_at_cutoff": st.column_config.ProgressColumn(display_cols["share_at_cutoff"], format="percent", min_value=0, max_value=1),
        },
    )
    st.download_button(
        "⬇ Download filtered results (CSV)",
        ranked.drop(columns=["south", "north", "west", "east"]).assign(cutoff_used=cutoff).to_csv(index=False).encode("utf-8"),
        file_name=f"floodsense_priority_areas_cutoff_{cutoff:.2f}.csv",
        mime="text/csv",
    )

rows = (event.selection.rows if event and hasattr(event, "selection") else []) or []
row_gid = table.iloc[rows[0]]["grid_id"] if rows and rows[0] < len(table) else None
if row_gid != ss.get("pr_last_row"):
    ss.pr_last_row = row_gid
    if row_gid and row_gid != ss.pr_selected:
        ss.pr_selected = row_gid
        st.rerun()

clicked = (state or {}).get("last_clicked")
if clicked and clicked != ss.pr_last_click:
    ss.pr_last_click = clicked
    gid = compute_grid_id(float(clicked["lat"]), float(clicked["lng"]))
    if gid in ranked_ids:
        ss.pr_selected = gid
        st.rerun()
    elif gid in excluded_ids:
        st.toast(f"Cell {gid} was sampled but is outside the current filters.")
    else:
        st.toast("That spot has no sampled data.")

st.plotly_chart(charts.score_histogram(ranked["mean_score"]), width="stretch", config={"displayModeBar": False})
st.caption("Distribution of the displayed cells' mean scores. No cutoff line is drawn: the 0.50 cutoff applies to "
           "individual observations and was not validated for cell means. Text alternative: the table above lists "
           "every coloured cell.")

# ---------------------------------------------------------------------------
# Selected cell detail
# ---------------------------------------------------------------------------
st.divider()
if ss.pr_selected and ss.pr_selected in gindex.index:
    row = gindex.loc[ss.pr_selected]
    rank_row = ranked_all[ranked_all["grid_id"] == ss.pr_selected]
    rank_text = f"rank {int(rank_row['rank'].iloc[0])} of {len(ranked_all):,}" if not rank_row.empty else "outside the current filters"
    st.subheader(f"Cell {ss.pr_selected} · {rank_text}")
    c = st.columns(4)
    c[0].metric("Mean score", format_score(row["mean_score"]))
    c[1].metric("Max sampled score", format_score(row["max_score"]))
    share_val = rank_row["share_at_cutoff"].iloc[0] if not rank_row.empty else None
    c[2].metric(f"Share ≥ {cutoff:.2f}", format_share(share_val))
    c[3].metric("Sampled locations", format_int(row["sample_count"]))
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.shap_bar(row, "Local SHAP explanation"), width="stretch", config={"displayModeBar": False})
        st.caption("Mean signed SHAP contributions (raw log-odds) across the cell's sampled locations; orange ▲ raises, blue ▼ lowers. Not causal.")
    with right:
        scout_says(explain_cell(row) + "<br><br>" + explain_coverage(row))
        render_context_note("planner", "coverage")
else:
    st.caption("Select a cell from the map or table to see its local explanation.")
