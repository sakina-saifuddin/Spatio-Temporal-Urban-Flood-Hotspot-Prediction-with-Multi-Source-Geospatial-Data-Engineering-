"""FloodSense — explainable historical flood susceptibility explorer for Houston.

Run from the repository root:
    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from components.disclaimers import DISCLAIMER  # noqa: E402
from components.navigation import build_navigation  # noqa: E402
from utils.data_loader import CSS_PATH, MASCOT_PATH, missing_data_files  # noqa: E402

st.set_page_config(
    page_title="FloodSense · Houston flood susceptibility explorer",
    page_icon=str(MASCOT_PATH) if MASCOT_PATH.exists() else "💧",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "FloodSense — research-based historical flood susceptibility estimates for the Houston study area. " + DISCLAIMER},
)


@st.cache_data(show_spinner=False)
def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


st.html(f"<style>{_css()}</style>")

missing = missing_data_files()
if missing:
    st.error(
        "FloodSense cannot start because required data files are missing from `app/data/`: "
        + ", ".join(f"`{name}`" for name in missing)
    )
    st.stop()

navigation = build_navigation()

with st.sidebar:
    if MASCOT_PATH.exists():
        st.logo(str(MASCOT_PATH), size="large")
    st.markdown("### FloodSense")
    st.caption("Explainable historical flood susceptibility for the Houston study area.")
    st.info("Research estimates only — not a flood forecast, official warning, or property-level risk assessment.", icon="ℹ️")
    st.caption("In an emergency, follow official instructions and call 911.")

navigation.run()

st.markdown(
    '<div class="fs-footer">FloodSense · COSC 6368 research project, University of Houston. '
    "Model: LightGBM without rainfall (frozen). Basemap and place search © OpenStreetMap contributors."
    "Research estimates only — not an official flood warning.</div>",
    unsafe_allow_html=True,
)
