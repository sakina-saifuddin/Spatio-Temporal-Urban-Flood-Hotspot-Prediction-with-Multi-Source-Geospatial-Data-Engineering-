"""Streamlit AppTest functional smoke tests for all five pages."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
DISCLAIMER_FRAGMENT = "not a real-time flood forecast"
SAMPLED = "2986_-9529"


def _app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [s.value for s in at.subheader] + [w.value for w in at.warning]
    return "\n".join(str(p) for p in parts)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_home_page():
    at = _app()
    text = _text(at)
    assert "FloodSense" in text and "Scout" in text
    assert DISCLAIMER_FRAGMENT in text
    assert "62,138" in text and "4,492" in text and "66,514" in text
    for url in ("weather.gov/hgx", "msc.fema.gov", "harriscountyfws.org", "hcfcd.org", "houstonoem.org"):
        assert url in text
    assert "mascot" not in text.lower() or "alt=" in text  # mascot rendered with alt text
    assert len(at.get("page_link")) >= 5
    # Refinements: human-readable features, coverage wording, no file-path captions.
    for label in ("Elevation", "Terrain slope", "Distance to bayou", "Highway proximity proxy", "Land cover type"):
        assert f"**{label}**" in text, label
    assert "how much sampled data is available" in text and "how sure we can be" not in text
    assert "model_metadata.json" not in text and "storm_metrics.csv" not in text
    assert {"cta_explore", "cta_historical"} <= {b.key for b in at.button}


@pytest.mark.parametrize("key, page_title", [("cta_explore", "Explore Houston"), ("cta_historical", "Historical Storm Explorer")])
def test_home_primary_calls_to_action(key, page_title):
    at = _app()
    next(b for b in at.button if b.key == key).click().run()
    assert not at.exception
    assert page_title in _text(at)


def test_explore_grid_selection_shap_and_comparison():
    at = _app("pages/explore.py")
    assert DISCLAIMER_FRAGMENT in _text(at)
    # All four layers render.
    for layer in ("susceptibility", "coverage", "factor", "reports"):
        at.radio(key="explore_layer").set_value(layer).run()
        assert not at.exception
    next(t for t in at.text_input if t.label == "Grid ID").set_value(SAMPLED)
    _button(at, "Open cell").click().run()
    assert not at.exception
    text = _text(at)
    assert f"Cell {SAMPLED}" in text and "Scout's read on this cell" in text
    assert "Mean susceptibility score" in text and "Sampled score range (min – max)" in text
    assert f"Selected: cell {SAMPLED}" in text and "#cell-details" in text  # visible selection feedback
    assert "% of tract near major highways" in text  # paved_pct shown as a percentage
    assert len(at.get("plotly_chart")) >= 1
    _button(at, "Set as Area A").click().run()
    next(t for t in at.text_input if t.label == "Grid ID").set_value("2969_-9530")
    _button(at, "Open cell").click().run()
    _button(at, "Set as Area B").click().run()
    assert not at.exception
    text = _text(at)
    assert "Comparing the two areas" in text
    assert "Area A · 2986_-9529 — mean susceptibility score" in text and "Area B · 2969_-9530" in text
    _button(at, "Clear comparison").click().run()
    _button(at, "↺ Reset map").click().run()
    assert at.session_state["explore_selected"] is None


def test_explore_layer_change_keeps_selection_and_viewport():
    at = _app("pages/explore.py")
    next(t for t in at.text_input if t.label == "Grid ID").set_value(SAMPLED)
    _button(at, "Open cell").click().run()
    key_before, view_before = at.session_state["explore_map_key"], at.session_state["explore_view"]
    for layer in ("coverage", "factor", "reports", "susceptibility"):
        at.radio(key="explore_layer").set_value(layer).run()
        assert not at.exception
        assert at.session_state["explore_selected"]["grid_id"] == SAMPLED
        assert at.session_state["explore_map_key"] == key_before  # base map not re-created
        assert at.session_state["explore_view"] == view_before


def test_explore_privacy_wording_names_tile_provider():
    text = _text(_app("pages/explore.py"))
    assert "tile.openstreetmap.org" in text and "never sent to the place-search service" in text
    assert "nothing is sent externally" not in text


def test_explore_no_data_and_nearby():
    at = _app("pages/explore.py")
    lat, lon = (n for n in at.number_input)
    lat.set_value(29.0)
    lon.set_value(-95.0)
    _button(at, "Find cell").click().run()
    assert not at.exception
    assert "no data" in _text(at).lower()
    _button(at, "Explore nearby sampled cells (different areas)").click().run()
    assert not at.exception
    # Invalid grid ID handled gracefully.
    next(t for t in at.text_input if t.label == "Grid ID").set_value("not-a-cell")
    _button(at, "Open cell").click().run()
    assert any("Grid IDs look like" in w.value for w in at.warning)


def test_priority_filters_cutoff_and_empty_state():
    at = _app("pages/priority.py")
    assert "Planner summary" in _text(at)
    at.selectbox(key="pr_metric").set_value("max_score").run()
    at.session_state["pr_top"] = "25"
    at.run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Cells shown").value == "25"
    assert any(m.label == "Average cell score" for m in at.metric)
    assert "unweighted average" in _text(at)
    assert "outside current filters" in _text(at)  # filtered-out cells are not called "no data"
    at.toggle(key="pr_all_labels").set_value(True).run()
    assert not at.exception
    at.slider(key="pr_cutoff").set_value(0.3).run()
    assert any("Exploratory cutoff" in w.value for w in at.warning)
    at.number_input(key="pr_min_samples").set_value(200).run()
    assert not at.exception and at.session_state["pr_min_samples"] == 200
    at.multiselect(key="pr_coverage").set_value(["Single sampled location"]).run()
    assert any("No sampled cells match" in i.value for i in at.info)
    _button(at, "↺ Reset filters").click().run()
    assert not at.exception and at.session_state["pr_min_samples"] == 1
    assert len(at.get("download_button")) == 1


@pytest.mark.parametrize("storm", ["Memorial Day Flood", "Tax Day Flood", "Hurricane Harvey", "Tropical Storm Imelda"])
def test_historical_each_storm(storm):
    at = _app("pages/historical.py")
    at.session_state["hs_storm"] = storm
    at.run()
    assert not at.exception
    text = _text(at)
    assert storm in text
    assert ("Independent test" in text) == (storm == "Tropical Storm Imelda") or "independent" in text.lower()


def test_historical_navigation_modes_and_cell():
    at = _app("pages/historical.py")
    assert at.session_state["hs_storm"] == "Memorial Day Flood"
    _button(at, "Next storm ▶").click().run()
    assert at.session_state["hs_storm"] == "Tax Day Flood"
    _button(at, "◀ Previous storm").click().run()
    _button(at, "◀ Previous storm").click().run()
    assert at.session_state["hs_storm"] == "Tropical Storm Imelda"
    for layer in ("predicted", "reported", "classification"):
        at.selectbox(key="hs_layer").set_value(layer).run()
        assert not at.exception
    at.radio(key="hs_mode").set_value("Side-by-side comparison").run()
    assert not at.exception
    at.multiselect(key="hs_classes").set_value(["FN"]).run()
    assert not at.exception
    at.multiselect(key="hs_classes").set_value([]).run()
    assert any("No cells contain" in i.value for i in at.info)
    at.session_state["hs_selected"] = SAMPLED
    at.run()
    assert not at.exception
    assert any(m.label == "Sampled observations" for m in at.metric)
    assert len(at.get("download_button")) == 0  # no download in empty state


def test_model_insights_sections():
    at = _app("pages/model_insights.py")
    text = _text(at)
    for fragment in ("Selected model and preprocessing", "Hyperparameter tuning", "Why rainfall is excluded",
                     "Two separate SHAP analyses", "Research limitations", "Sakina Saifuddin", "nested cross-validation"):
        assert fragment in text, fragment
    assert len(at.get("imgs")) >= 8 or len(at.get("image")) >= 8
