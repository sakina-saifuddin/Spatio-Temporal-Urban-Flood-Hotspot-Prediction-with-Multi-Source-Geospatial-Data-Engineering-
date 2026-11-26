"""Scout, the FloodSense guide: a speech-bubble component and data-driven
explanation templates. Templates are deterministic — no external LLM."""

from __future__ import annotations

import base64
import html
from functools import lru_cache

import pandas as pd
import streamlit as st

from utils.data_loader import MASCOT_PATH
from utils.grid_utils import FEATURE_LABELS, factor_label, format_share, score_word

MASCOT_ALT = "Scout, the FloodSense guide: a friendly water droplet wearing a cowboy hat"


@lru_cache(maxsize=1)
def mascot_data_uri() -> str | None:
    if not MASCOT_PATH.exists():
        return None
    encoded = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scout_says(message: str, title: str | None = None, size: str = "small", details: str | None = None) -> None:
    """Render Scout with a speech bubble. ``message`` may contain basic Markdown-free HTML-escaped text."""
    uri = mascot_data_uri()
    image = (
        f'<img src="{uri}" alt="{MASCOT_ALT}" class="fs-scout-img fs-scout-{size}">'
        if uri
        else '<div class="fs-scout-fallback" aria-hidden="true">💧</div>'
    )
    heading = f'<div class="fs-scout-title">{html.escape(title, quote=False)}</div>' if title else ""
    st.markdown(
        f'<div class="fs-scout" role="note" aria-label="Scout says">{image}'
        f'<div class="fs-bubble">{heading}<div>{message}</div></div></div>',
        unsafe_allow_html=True,
    )
    if details:
        with st.expander("Scout explains more"):
            st.markdown(details)


def esc(value) -> str:
    return html.escape(str(value))


# --------------------------------------------------------------------------
# Explanation templates (all use the selected data).
# --------------------------------------------------------------------------

def _strongest(row: pd.Series) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    values = {name: float(row[f"shap_{name}"]) for name in FEATURE_LABELS}
    positive = max(values.items(), key=lambda item: item[1])
    negative = min(values.items(), key=lambda item: item[1])
    return (positive if positive[1] > 0 else None, negative if negative[1] < 0 else None)


def explain_cell(row: pd.Series) -> str:
    score = float(row["mean_score"])
    word = score_word(score).lower()
    positive, negative = _strongest(row)
    parts = [
        f"Cell <b>{esc(row['grid_id'])}</b> has a mean susceptibility score of <b>{score:.3f}</b> "
        f"(a <b>{word}</b> score on FloodSense's 0–1 scale) across "
        f"<b>{int(row['sample_count'])}</b> sampled location{'s' if int(row['sample_count']) != 1 else ''}."
    ]
    if score >= 0.6:
        parts.append("The model tended to score sampled spots here similarly to places linked with past flood reports.")
    elif score < 0.2:
        parts.append("The model tended to score sampled spots here lower than places linked with past flood reports — that is not the same as saying the area is safe.")
    else:
        parts.append("The sampled spots here fall in the middle of the model's range.")
    if positive:
        parts.append(
            f"The biggest push <b>up</b> came from <b>{esc(FEATURE_LABELS[positive[0]].lower())}</b> "
            f"({positive[1]:+.2f} log-odds on average)."
        )
    if negative:
        parts.append(
            f"The biggest push <b>down</b> came from <b>{esc(FEATURE_LABELS[negative[0]].lower())}</b> "
            f"({negative[1]:+.2f} log-odds)."
        )
    return " ".join(parts)


def explain_coverage(row: pd.Series) -> str:
    category = row["coverage_category"]
    n = int(row["sample_count"])
    if category == "Single sampled location":
        return (f"Heads up: only <b>one</b> location was sampled in this cell, so its summary rests on a single point. "
                "That's about how much data we have, not how sure the model is.")
    if category == "Limited coverage":
        return (f"This cell has <b>limited coverage</b> ({n} sampled locations). Treat the summary as a rough sketch.")
    return (f"This cell has <b>greater coverage</b> ({n} sampled locations). More samples give a fuller picture of the "
            "cell, though coverage still isn't a confidence score.")


def explain_no_data(grid_id: str) -> str:
    return (
        f"Hmm — cell <b>{esc(grid_id)}</b> had no sampled locations in the research data, so FloodSense has "
        "<b>no estimate</b> there. That doesn't mean low or high risk; we simply didn't look there. "
        "You can explore nearby sampled cells instead, clearly labelled as different areas."
    )


def explain_search(name: str, grid_id: str, sampled: bool) -> str:
    status = "falls inside a sampled cell" if sampled else "falls in a cell without sampled data"
    return f"Found it! <b>{esc(name)}</b> {status} (<b>{esc(grid_id)}</b>)."


def explain_storm(metrics_row: pd.Series, storm_info: dict) -> str:
    protocol = (
        "an <b>independent test</b> — the model never saw this storm before its single, frozen evaluation"
        if metrics_row["evaluation_protocol"] == "independent_test"
        else "<b>leave-one-storm-out</b> validation — the model was trained on the other development storms and scored this one"
    )
    return (
        f"Welcome to <b>{esc(metrics_row['storm_name'])}</b> ({int(metrics_row['year'])}). "
        f"Here we look at <b>{int(metrics_row['observation_count']):,}</b> sampled observations in "
        f"<b>{int(metrics_row['grid_count']):,}</b> cells, scored using {protocol}. "
        f"At the frozen 0.50 cutoff the model caught <b>{metrics_row['recall'] * 100:.1f}%</b> of "
        f"report-associated observations (recall) and <b>{metrics_row['precision'] * 100:.1f}%</b> of its "
        "positive calls matched a recorded report (precision)."
    )


def explain_false_negatives(row: pd.Series) -> str:
    fn = int(row["fn_count"])
    if fn == 0:
        return "No missed reports (false negatives) among this cell's sampled observations."
    return (
        f"In this cell, <b>{fn}</b> report-associated observation{'s were' if fn != 1 else ' was'} scored below 0.50 — "
        f"false negatives ({format_share(row['fn_share'])} of the cell's sample). These are spots where the "
        "model's environmental features didn't look like other reported locations; local drainage or storm-specific "
        "rain, which the model doesn't see, may matter there."
    )


def explain_planner(summary: dict, metric_label: str, cutoff: float, filters_active: bool) -> str:
    if summary["cells"] == 0:
        return "No cells match these filters. Try lowering the minimum sample count or widening the score range."
    text = (
        f"Showing <b>{summary['cells']:,}</b> sampled cells ranked by <b>{esc(metric_label.lower())}</b>, covering "
        f"<b>{summary['locations']:,}</b> sampled locations. Across them, "
        f"<b>{summary['mean_share'] * 100:.1f}%</b> of sampled locations score at or above "
        f"<b>{cutoff:.2f}</b>."
    )
    if abs(cutoff - 0.5) > 1e-9:
        text += " That cutoff is exploratory — only 0.50 was selected and tested in the research."
    if filters_active:
        text += " Filters are narrowing the list."
    return text + " Use these rankings as a starting point for further investigation, not as decisions."
