"""Page 4 — Historical Storm Explorer. Chapter 4: Journey Through Houston's Flood History."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from components import charts, maps
from components.disclaimers import render_context_note, render_disclaimer_banner
from components.mascot import explain_false_negatives, explain_storm, scout_says
from components.navigation import chapter_header, page_link
from utils.data_loader import STORM_ORDER, load_historical_grid, load_metadata, load_storm_metrics
from utils.grid_utils import CLASS_COLORS, CLASS_EXPLANATIONS, CLASS_LABELS, compute_grid_id, format_share, grid_bounds
from utils.metrics import cell_classification_breakdown

# Context limited to facts recorded in the project's own documentation.
STORM_CONTEXT = {
    "Memorial Day Flood": "A May 2015 storm. Its development observations were scored by a model trained on the Tax Day and Harvey data (leave-one-storm-out).",
    "Tax Day Flood": "An April 2016 storm. Its development observations were scored by a model trained on the Memorial Day and Harvey data (leave-one-storm-out).",
    "Hurricane Harvey": "An August 2017 hurricane. The project's rainfall diagnostic found Harvey's sampled rainfall reached 20.8 inches, beyond the range of the other training storms — one reason the final model excludes rainfall.",
    "Tropical Storm Imelda": "A September 2019 storm held back as the one-time independent test. The model and its 0.50 cutoff were frozen before Imelda was scored, and the result was not used to change anything.",
}

ss = st.session_state
ss.setdefault("hs_storm", STORM_ORDER[0])
ss.setdefault("hs_selected", None)
ss.setdefault("hs_last_click", None)
ss.setdefault("hs_map_key", 0)

metadata = load_metadata()
metrics = load_storm_metrics()


def step(delta: int) -> None:
    index = STORM_ORDER.index(ss.hs_storm)
    ss.hs_storm = STORM_ORDER[(index + delta) % len(STORM_ORDER)]
    ss.hs_selected = None
    ss.hs_last_click = None


def on_storm_change() -> None:
    ss.hs_selected = None
    ss.hs_last_click = None


@st.cache_data(show_spinner="Loading storm cells…")
def storm_cells(storm: str) -> pd.DataFrame:
    hg = load_historical_grid()
    return hg[hg["storm_name"] == storm].reset_index(drop=True)


@st.cache_resource(show_spinner=False, max_entries=24)
def storm_geojson(storm: str, layer: str, focus: str, classes: tuple[str, ...]) -> dict:
    return maps.historical_geojson(_filter(storm_cells(storm), classes), layer, focus)


def _filter(cells: pd.DataFrame, classes: tuple[str, ...]) -> pd.DataFrame:
    if not classes:
        return cells.iloc[0:0]
    if len(classes) == 4:
        return cells
    mask = pd.Series(False, index=cells.index)
    for key in classes:
        mask |= cells[f"{key.lower()}_count"] > 0
    return cells[mask]


chapter_header(
    "historical",
    "Historical Storm Explorer",
    "Compare the model's saved predictions with recorded flood-report associations for four Houston storms. "
    "Each storm has its own sample, coverage, and evaluation protocol.",
)
render_disclaimer_banner()

# ---------------------------------------------------------------------------
# Chronological navigation
# ---------------------------------------------------------------------------
nav_prev, nav_select, nav_next = st.columns([1, 4, 1], vertical_alignment="bottom")
nav_prev.button("◀ Previous storm", on_click=step, args=(-1,), width="stretch")
years = metadata["historical"]["storm_years"]
nav_select.segmented_control(
    "Storm", STORM_ORDER, key="hs_storm", required=True, on_change=on_storm_change,
    format_func=lambda s: f"{s} ({years[s]})", width="stretch",
)
nav_next.button("Next storm ▶", on_click=step, args=(1,), width="stretch")

storm = ss.hs_storm
mrow = metrics[metrics["storm_name"] == storm].iloc[0]
cells = storm_cells(storm)
is_test = mrow["evaluation_protocol"] == "independent_test"

st.markdown(
    f'<span class="fs-pill">{int(mrow["year"])}</span>'
    f'<span class="fs-pill">{"Independent test" if is_test else "Leave-one-storm-out validation"}</span>'
    f'<span class="fs-pill">{int(mrow["observation_count"]):,} observations</span>'
    f'<span class="fs-pill">{int(mrow["grid_count"]):,} sampled cells</span>',
    unsafe_allow_html=True,
)
scout_says(explain_storm(mrow, {}), title=f"Chapter stop: {storm}", details=STORM_CONTEXT[storm])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
cols = st.columns(6)
for col, (key, label) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                     ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC (AP)")]):
    col.metric(label, f"{mrow[key]:.3f}")
st.caption("Classification metrics use the frozen 0.50 cutoff. PR-AUC is average precision. Values are read from `storm_metrics.csv`; "
           "no evaluation was rerun." + (" Imelda values are the frozen one-time independent test results." if is_test else ""))

# ---------------------------------------------------------------------------
# Map controls
# ---------------------------------------------------------------------------
st.subheader("Map")
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    mode = st.radio("View", ["Single map", "Side-by-side comparison"], horizontal=True, key="hs_mode")
with c2:
    layer = st.selectbox("Layer (single map)", list(maps.HISTORICAL_LAYERS), format_func=maps.HISTORICAL_LAYERS.get,
                         key="hs_layer", disabled=mode != "Single map")
with c3:
    focus = st.selectbox("Classification to shade", list(CLASS_LABELS), index=3, format_func=lambda k: CLASS_LABELS[k],
                         key="hs_focus", disabled=not (mode == "Single map" and layer == "classification"),
                         help="Cells are shaded by the share of their observations in this category.")
classes = st.multiselect(
    "Only show cells containing these classifications", list(CLASS_LABELS), default=list(CLASS_LABELS),
    format_func=lambda k: CLASS_LABELS[k], key="hs_classes",
)
filtered = _filter(cells, tuple(classes))
st.caption(f"{len(filtered):,} of {len(cells):,} sampled cells shown for {storm}. Classifications are relative to recorded dataset labels.")

if filtered.empty:
    st.info("No cells contain the selected classifications. Choose at least one classification.")
else:
    map_col, legend_col = st.columns([4, 1])
    with map_col:
        if mode == "Single map":
            fmap = maps.historical_map(storm_geojson(storm, layer, focus, tuple(classes)), maps.HISTORICAL_LAYERS[layer])
            if ss.hs_selected:
                maps.add_selection(fmap, grid_bounds(ss.hs_selected))
            state = st_folium(fmap, key=f"hs_map_{ss.hs_map_key}_{storm}_{layer}", height=540,
                              use_container_width=True, returned_objects=["last_clicked"])
            clicked = (state or {}).get("last_clicked")
            if clicked and clicked != ss.hs_last_click:
                ss.hs_last_click = clicked
                gid = compute_grid_id(float(clicked["lat"]), float(clicked["lng"]))
                ss.hs_selected = gid
                st.rerun()
        else:
            dual = maps.historical_dual_map(
                storm_geojson(storm, "predicted", focus, tuple(classes)),
                storm_geojson(storm, "reported", focus, tuple(classes)),
                "Predicted susceptibility", "Recorded report-associated share",
            )
            st_folium(dual, key=f"hs_dual_{storm}", height=520, use_container_width=True, returned_objects=[])
            st.caption("Left: mean saved prediction score. Right: share of sampled observations associated with a recorded report. "
                       "The two maps pan and zoom together. Use the cell selector below to inspect a cell.")
    with legend_col:
        if mode == "Single map":
            st.markdown(maps.legend_html(focus if layer == "classification" else layer), unsafe_allow_html=True)
        else:
            st.markdown(maps.legend_html("predicted"), unsafe_allow_html=True)
            st.markdown(maps.legend_html("reported"), unsafe_allow_html=True)
        render_context_note("historical")

    st.download_button(
        f"⬇ Download {storm} cell summaries (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"floodsense_{storm.lower().replace(' ', '_')}_cells.csv",
        mime="text/csv",
        help="Aggregated cell-level summaries only; individual observation coordinates are not included.",
    )

# ---------------------------------------------------------------------------
# Cell inspection
# ---------------------------------------------------------------------------
st.subheader("Inspect a cell")
ids = cells.sort_values("grid_id")["grid_id"].tolist()
pick = st.selectbox("Choose a sampled cell for this storm (or click the single map)", ids,
                    index=ids.index(ss.hs_selected) if ss.hs_selected in ids else None,
                    placeholder="Choose a cell…", key=f"hs_pick_{storm}_{ss.hs_selected}")
if pick and pick != ss.hs_selected:
    ss.hs_selected = pick
    st.rerun()

if ss.hs_selected:
    match = cells[cells["grid_id"] == ss.hs_selected]
    if match.empty:
        st.info(f"Cell {ss.hs_selected} had no sampled observations for {storm}. That's no data — not a sign the area stayed dry.")
    else:
        row = match.iloc[0]
        k = st.columns(4)
        k[0].metric("Sampled observations", f"{int(row['sample_count']):,}")
        k[1].metric("Mean saved score", f"{row['mean_score']:.3f}")
        k[2].metric("Report-associated share", format_share(row["report_positive_share"]))
        k[3].metric("Correct at 0.50", format_share(row["accuracy"]))
        left, right = st.columns([3, 2])
        breakdown = cell_classification_breakdown(row)
        with left:
            st.plotly_chart(charts.classification_breakdown(breakdown), width="stretch", config={"displayModeBar": False})
        with right:
            st.dataframe(
                breakdown.assign(classification=breakdown["classification"].map(CLASS_LABELS),
                                 share=breakdown["share"].map(format_share)),
                hide_index=True, width="stretch",
                column_config={"classification": "Classification", "count": "Observations", "share": "Share of cell"},
            )
            scout_says(explain_false_negatives(row))
        st.caption("Event-specific SHAP explanations aren't available because the historical fold-specific models weren't saved.")
        page_link("explore", "See the general model explanation for this area on Explore Houston →")

with st.expander("What do the classifications mean?"):
    for key, text in CLASS_EXPLANATIONS.items():
        st.markdown(
            f'<span class="fs-swatch" style="background:{CLASS_COLORS[key]}"></span> **{CLASS_LABELS[key]} ({key})** — {text}',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Confusion matrix and cross-storm comparison
# ---------------------------------------------------------------------------
st.subheader("Performance")
left, right = st.columns([2, 3])
with left:
    st.plotly_chart(charts.confusion_matrix(mrow), width="stretch", config={"displayModeBar": False})
    st.caption(f"TP {int(mrow['tp']):,} · TN {int(mrow['tn']):,} · FP {int(mrow['fp']):,} · FN {int(mrow['fn']):,}. "
               "Hover over a cell for counts.")
with right:
    st.plotly_chart(charts.storm_comparison(metrics, storm), width="stretch", config={"displayModeBar": False})

table = metrics[["storm_name", "year", "evaluation_protocol", "observation_count", "grid_count", "accuracy", "precision",
                 "recall", "f1", "roc_auc", "pr_auc"]].copy()
table["evaluation_protocol"] = table["evaluation_protocol"].map(
    {"leave_one_storm_out": "Leave-one-storm-out", "independent_test": "Independent test"})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "storm_name": "Storm", "year": st.column_config.NumberColumn("Year", format="%d"), "evaluation_protocol": "Protocol",
    "observation_count": "Observations", "grid_count": "Cells",
    **{c: st.column_config.NumberColumn(c.replace("_", "-").upper() if "auc" in c else c.title(), format="%.3f")
       for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
})
st.caption(
    "Storms differ in their samples, coverage, and evaluation protocols, so compare them with care. The first three use "
    "leave-one-storm-out development predictions; Imelda uses the one-time independent test. This explorer compares "
    "saved snapshots — it is not a simulation of how floodwater moved."
)
render_context_note("labels")
