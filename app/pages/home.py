"""Page 1 — Home. Chapter 1: Welcome to Houston."""

from __future__ import annotations

import streamlit as st

from components.cards import stat_cards
from components.disclaimers import (
    CONTEXT_NOTES,
    render_disclaimer_banner,
    render_official_resources,
)
from components.mascot import MASCOT_ALT, mascot_data_uri, scout_says
from components.navigation import PAGES, page_link, switch_to
from utils.data_loader import load_metadata, load_storm_metrics
from utils.grid_utils import FEATURE_LABELS

metadata = load_metadata()
counts = metadata["dataset_counts"]
storms = metadata["historical"]["storm_years"]

uri = mascot_data_uri()
image = f'<img src="{uri}" alt="{MASCOT_ALT}">' if uri else ""
st.markdown(
    f'<section class="fs-hero" aria-label="FloodSense introduction">{image}<div>'
    '<p style="text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin:0;color:#F8D04B">Chapter 1 · Welcome to Houston</p>'
    "<h1>FloodSense</h1>"
    "<p>Explore where a machine learning model, trained on past Houston flood reports and the shape of the land, "
    "sees patterns of <b>historical flood susceptibility</b> — and see exactly why.</p>"
    '<span class="fs-tag">Research estimates · not a flood forecast</span>'
    "</div></section>",
    unsafe_allow_html=True,
)
render_disclaimer_banner()

scout_says(
    "Howdy, I'm <b>Scout</b>! I'll guide you through five chapters. You can jump to any of them, in any order. "
    "I'll explain what the model sees in each place, how much sampled data is available, and what the results "
    "<i>can't</i> tell you.",
    title="Meet your guide",
    size="large",
)

st.markdown("#### Start exploring")
with st.container(key="fs_cta"):
    cta_left, cta_right = st.columns([1, 1])
    with cta_left:
        if st.button("🗺️ Explore Houston — look up a place or coordinates", type="primary", width="stretch", key="cta_explore"):
            switch_to("explore")
    with cta_right:
        if st.button("🌧️ Historical Storm Explorer — revisit four storms", type="primary", width="stretch", key="cta_historical"):
            switch_to("historical")

st.markdown("### Why Houston?")
storm_list = ", ".join(f"{name} ({year})" for name, year in storms.items())
st.markdown(
    "Houston is flat, low-lying, and laced with bayous, and heavy rain has repeatedly flooded streets and homes. "
    f"FloodSense studies four storms from the research dataset: **{storm_list}**. "
    "During each one, residents filed City of Houston 311 flooding reports. The research team paired those reports, "
    "and nearby sampled places without reports, with terrain, drainage, land-cover, and highway-proximity data."
)

st.markdown("### What is flood susceptibility?")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "A **susceptibility score** (0 to 1) shows how closely a place's environmental features match locations that "
        "were associated with past flood reports. Higher scores mean the model saw a closer resemblance."
    )
    st.markdown(
        "- It **is** a relative, model-based view of historical patterns.\n"
        "- It **is not** a forecast, a probability that your home will flood, or an official flood map."
    )
with c2:
    st.markdown("**The five environmental features the model uses**")
    plain = {
        "elevation_m": "How high the ground is, in meters.",
        "slope_degrees": "How steep the terrain is, in degrees.",
        "distance_to_bayou_m": "How far the place is from a mapped bayou or stream, in meters.",
        "paved_pct": "The share of the surrounding census tract near major highways. It is not a measure of all paved surfaces.",
        "land_cover_type": "The kind of developed land (open space, low, medium, or high intensity).",
    }
    for name in metadata["feature_descriptions"]:
        st.markdown(f"- **{FEATURE_LABELS[name]}** — {plain.get(name, metadata['feature_descriptions'][name])}")
    st.caption(CONTEXT_NOTES["rainfall"])

st.markdown("### The research at a glance")
imelda = load_storm_metrics().query("evaluation_protocol == 'independent_test'").iloc[0]
stat_cards([
    (f"{counts['development_observations']:,}", "development observations (2015–2017)"),
    (f"{counts['unique_sampled_locations']:,}", "unique sampled locations with explanations"),
    (f"{counts['geographic_grid_cells']:,}", "geographic cells (~1.11 × 0.96 km)"),
    (f"{counts['historical_storms']}", "historical storms explored"),
    (f"{counts['historical_observations']:,}", "historical storm observations"),
    (f"{counts['independent_test_observations']:,}", "independent test observations (Imelda 2019)"),
    (f"{imelda['recall']:.1%}", "Imelda recall at the frozen 0.50 cutoff"),
    (f"{imelda['roc_auc']:.3f}", "Imelda ROC-AUC (one-time independent test)"),
])
st.caption(
    f"Model: {metadata['model']['algorithm']} without rainfall, using a frozen classification cutoff of "
    f"{metadata['model']['frozen_cutoff']:.2f}. See Behind the Predictions for sources and methods."
)

st.markdown("### Five chapters")
cols = st.columns(2)
for index, page in enumerate(PAGES):
    with cols[index % 2]:
        st.markdown(
            f'<div class="fs-card"><p style="color:#8C3F08;font-weight:600;font-size:.8rem;margin:0">{page["chapter"]}</p>'
            f'<h3>{page["icon"]} {page["title"]}</h3><p>{page["blurb"]}</p></div>',
            unsafe_allow_html=True,
        )
        page_link(page["key"], f"Open {page['title']}")

st.markdown("### Official flood information")
st.markdown("For current conditions, warnings, and official flood maps, use these resources:")
render_official_resources()
