"""Research disclaimers, limitations, and verified official resources."""

from __future__ import annotations

import streamlit as st

DISCLAIMER = (
    "FloodSense provides research-based historical flood susceptibility estimates. "
    "It is not a real-time flood forecast, official warning system, or "
    "property-level flood risk assessment."
)

# URLs verified on 2026-11-25 by fetching each page.
OFFICIAL_RESOURCES = [
    {
        "name": "National Weather Service — Houston/Galveston",
        "url": "https://www.weather.gov/hgx/",
        "use": "Official forecasts, watches, and warnings.",
    },
    {
        "name": "FEMA Flood Map Service Center",
        "url": "https://msc.fema.gov/portal/home",
        "use": "Official flood hazard maps for the National Flood Insurance Program.",
    },
    {
        "name": "Harris County Flood Warning System",
        "url": "https://www.harriscountyfws.org/",
        "use": "Real-time rainfall and channel water-level gauges.",
    },
    {
        "name": "Harris County Flood Control District",
        "url": "https://www.hcfcd.org/",
        "use": "Flood risk reduction projects and county flood information.",
    },
    {
        "name": "AlertHouston (City of Houston Office of Emergency Management)",
        "url": "https://houstonoem.org/",
        "use": "Sign up for official City of Houston emergency alerts.",
    },
]

CONTEXT_NOTES = {
    "scores": "Scores are relative model outputs between 0 and 1, not calibrated probabilities that a place will flood in a future storm.",
    "rainfall": "The selected model does not use rainfall, so it cannot describe a specific upcoming storm.",
    "grid": "Each geographic cell is about 1.11 km north–south by 0.96 km east–west. Only sampled cells have estimates.",
    "coverage": "Coverage describes how many locations were sampled in a cell. It is not a measure of prediction confidence.",
    "labels": "Positive labels mean a sampled location was associated with a recorded flood report. No report does not prove a location stayed dry.",
    "shap": "SHAP contributions show how the model used each feature. They are not causal effects.",
    "in_sample": "General susceptibility scores come from the model fitted on all development data, so they are in-sample outputs.",
    "boundary": "The study area is a sampled grid and is not clipped to Houston city limits.",
    "planner": "Rankings support further investigation only. They are not official intervention, investment, or emergency-response decisions.",
    "historical": "Historical maps summarise sampled observations and saved predictions. They do not show actual inundation extent or flood movement.",
}


def render_disclaimer_banner() -> None:
    st.markdown(
        f'<div class="fs-disclaimer" role="note"><strong>Research estimates only.</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_context_note(*keys: str) -> None:
    notes = " ".join(CONTEXT_NOTES[key] for key in keys)
    st.caption(f"ℹ️ {notes}")


def render_official_resources(compact: bool = False) -> None:
    if compact:
        links = " · ".join(f"[{r['name']}]({r['url']})" for r in OFFICIAL_RESOURCES)
        st.markdown(f"**Official flood information:** {links}")
        return
    for resource in OFFICIAL_RESOURCES:
        st.markdown(f"- **[{resource['name']}]({resource['url']})** — {resource['use']}")
    st.caption(
        "FloodSense does not replace these official resources. In an emergency, follow "
        "instructions from local officials and call 911."
    )


def render_limitations(metadata: dict) -> None:
    for item in metadata.get("limitations", []):
        st.markdown(f"- {item}")
