"""Unit tests for geographic matching, planner ranking, and geocoding."""

from unittest import mock

import pandas as pd
import pytest
import requests

from utils import data_loader as dl
from utils import geocoding
from utils.grid_utils import (
    compute_grid_id,
    coverage_category,
    grid_bounds,
    is_valid_grid_id,
    match_coordinate,
    nearby_sampled_cells,
    score_bin_index,
    validate_coordinates,
)
from utils.metrics import rank_grids, share_at_or_above, summary_statistics


# --- geographic matching ---------------------------------------------------

def test_grid_id_basic():
    assert compute_grid_id(29.862192, -95.294884) == "2986_-9529"


def test_grid_id_ties_round_half_to_even():
    # 0.125/0.01 == 12.5 exactly in float -> ties to even gives 12, not 13.
    assert compute_grid_id(0.125, 0.135) == "12_14"
    assert compute_grid_id(-0.125, 0.0) == "-12_0"
    # Must agree with pandas Series.round(), the research convention.
    values = pd.Series([0.125, 0.135, 29.995, -95.005, 29.465])
    expected = (values / 0.01).round().astype(int).tolist()
    assert [int(compute_grid_id(v, 0).split("_")[0]) for v in values] == expected


def test_grid_bounds_and_validation():
    b = grid_bounds("2986_-9529")
    assert b == {"south": 29.855, "north": 29.865, "west": -95.295, "east": -95.285}
    assert is_valid_grid_id("2986_-9529") and not is_valid_grid_id("abc") and not is_valid_grid_id("1_2_3")
    assert validate_coordinates(None, 1) and validate_coordinates(91, 0) and validate_coordinates(0, 181)
    assert validate_coordinates(29.7, -95.3) is None


def test_strict_matching_no_silent_substitution():
    sampled = set(dl.load_grid()["grid_id"])
    hit = match_coordinate(29.862192, -95.294884, sampled)
    assert hit.sampled and hit.grid_id == "2986_-9529"
    miss = match_coordinate(40.7128, -74.0060, sampled)  # New York — outside study area
    assert not miss.sampled and miss.grid_id == "4071_-7401"


def test_nearby_is_separate_and_bounded():
    grid = dl.load_grid()
    near = nearby_sampled_cells(29.76, -95.37, grid, count=5)
    assert 0 < len(near) <= 5 and near["distance_km"].is_monotonic_increasing
    assert nearby_sampled_cells(40.71, -74.0, grid).empty


def test_coverage_and_bins():
    assert [coverage_category(n) for n in (0, 1, 2, 4, 5)] == [
        "No data", "Single sampled location", "Limited coverage", "Limited coverage", "Greater coverage"]
    assert [score_bin_index(s) for s in (0.0, 0.199, 0.2, 0.8, 1.0)] == [0, 0, 1, 4, 4]
    assert score_bin_index(None) is None


def test_paved_pct_is_fraction_displayed_as_percentage():
    from utils.grid_utils import format_paved

    loc = dl.load_locations()
    assert loc["paved_pct"].between(0, 1).all()
    assert format_paved(0.749071) == "75% of tract near major highways"
    assert format_paved(float("nan")) == "—"


# --- refinement regressions ----------------------------------------------------

def test_score_histogram_has_no_cutoff_line():
    from components.charts import score_histogram

    fig = score_histogram(dl.load_grid()["mean_score"])
    assert not fig.layout.shapes and not fig.layout.annotations


def test_shap_chart_axes_use_automargin():
    from components.charts import shap_bar

    fig = shap_bar(dl.load_grid().iloc[0], "t")
    assert fig.layout.yaxis.automargin and fig.layout.xaxis.automargin
    assert "Highway proximity proxy" in list(fig.data[0].y)


def test_scroll_zoom_disabled_on_request():
    from components.maps import base_map

    assert '"scrollWheelZoom": false' in base_map(scroll_zoom=False).get_root().render()
    assert '"scrollWheelZoom": true' in base_map().get_root().render()


def test_excluded_cells_distinct_from_no_data():
    from components import maps

    full = maps.development_geojson(dl.load_boundaries(), dl.load_grid(), "susceptibility")
    before = {f["id"]: f["properties"]["fill"] for f in full["features"]}
    sub = maps.subset_geojson(full, {"2986_-9529", "2969_-9530"}, fill=maps.EXCLUDED_FILL)
    assert len(sub["features"]) == 2 and all(f["properties"]["fill"] == maps.EXCLUDED_FILL for f in sub["features"])
    assert {f["id"]: f["properties"]["fill"] for f in full["features"]} == before  # shared source not mutated
    legend = maps.legend_html("susceptibility", include_excluded=True)
    assert "outside current filters" in legend and "No data (unsampled" in legend


def test_shared_layer_cache_returns_same_object():
    from components import maps

    assert maps.cached_development_geojson("susceptibility") is maps.cached_development_geojson("susceptibility")
    assert dl.load_locations() is dl.load_locations()


# --- planner -----------------------------------------------------------------

def test_share_at_frozen_cutoff_matches_saved_grid():
    grid = dl.load_grid()
    shares = share_at_or_above(dl.load_locations(), 0.5).set_index("grid_id")
    merged = grid.set_index("grid_id").join(shares)
    assert (merged["count_at_cutoff"] == merged["locations_above_cutoff"]).all()
    assert merged["share_at_cutoff"].sub(merged["share_above_cutoff"]).abs().max() < 1e-12


def test_exploratory_cutoff_uses_locations_not_grid_mean():
    locations = dl.load_locations()
    shares = share_at_or_above(locations, 0.3).set_index("grid_id")
    gid = "2986_-9529"
    subset = locations[locations["grid_id"] == gid]["susceptibility_score"]
    assert shares.at[gid, "share_at_cutoff"] == pytest.approx((subset >= 0.3).mean())
    assert shares["share_at_cutoff"].between(0, 1).all()
    low, high = share_at_or_above(locations, 0.2), share_at_or_above(locations, 0.8)
    assert (low.set_index("grid_id")["share_at_cutoff"] >= high.set_index("grid_id")["share_at_cutoff"]).all()


def test_rank_and_filters():
    grid = dl.load_grid()
    shares = share_at_or_above(dl.load_locations(), 0.5)
    ranked = rank_grids(grid, shares)
    assert len(ranked) == 4492 and ranked["rank"].tolist() == list(range(1, 4493))
    assert ranked["mean_score"].is_monotonic_decreasing
    top = rank_grids(grid, shares, metric="max_score", top_n=25)
    assert len(top) == 25 and top["max_score"].is_monotonic_decreasing
    filt = rank_grids(grid, shares, min_samples=5, score_range=(0.4, 0.9), coverage=["Greater coverage"])
    assert (filt["sample_count"] >= 5).all() and filt["mean_score"].between(0.4, 0.9).all()
    assert set(filt["coverage_category"]) == {"Greater coverage"}
    by_share = rank_grids(grid, shares, metric="share_at_cutoff", top_n=10)
    assert by_share["share_at_cutoff"].is_monotonic_decreasing
    empty = rank_grids(grid, shares, min_samples=10_000)
    assert empty.empty and summary_statistics(empty)["cells"] == 0
    with pytest.raises(ValueError):
        rank_grids(grid, shares, metric="composite_risk")


def test_summary_statistics():
    grid = dl.load_grid()
    ranked = rank_grids(grid, share_at_or_above(dl.load_locations(), 0.5))
    stats = summary_statistics(ranked)
    meta = dl.load_metadata()
    assert stats["locations"] == 62_138
    assert stats["mean_share"] == pytest.approx(meta["susceptibility"]["locations_at_or_above_cutoff"] / 62_138)


# --- geocoding (network mocked) ---------------------------------------------

class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("bad json")
        return self._payload


@pytest.fixture(autouse=True)
def _clear_geocoder(monkeypatch):
    geocoding.clear_cache()
    monkeypatch.setattr(geocoding, "MIN_INTERVAL_SECONDS", 0.0)
    yield
    geocoding.clear_cache()


def test_geocode_success_sends_identifying_user_agent():
    session = mock.Mock()
    session.get.return_value = FakeResponse(payload=[{"display_name": "Buffalo Bayou Park, Houston", "lat": "29.76", "lon": "-95.38", "type": "park"}])
    result = geocoding.geocode("Buffalo Bayou Park", session=session)
    assert result.ok and result.places[0].lat == pytest.approx(29.76)
    kwargs = session.get.call_args.kwargs
    assert "FloodSense" in kwargs["headers"]["User-Agent"]
    assert kwargs["timeout"] == geocoding.TIMEOUT_SECONDS
    assert kwargs["params"]["limit"] == 5
    # Identical repeat query is served from memory, not re-requested.
    geocoding.geocode("buffalo  bayou park", session=session)
    assert session.get.call_count == 1


@pytest.mark.parametrize(
    "response, message",
    [
        (FakeResponse(payload=[]), "No matching"),
        (FakeResponse(status=429), "rate limited"),
        (FakeResponse(status=500), "HTTP 500"),
        (FakeResponse(bad_json=True), "unreadable"),
    ],
)
def test_geocode_error_paths(response, message):
    session = mock.Mock()
    session.get.return_value = response
    result = geocoding.geocode("Some public place", session=session)
    assert not result.ok and message in result.error


def test_geocode_timeout_and_connection_errors():
    session = mock.Mock()
    session.get.side_effect = requests.Timeout()
    assert "did not respond" in geocoding.geocode("Place one", session=session).error
    session.get.side_effect = requests.ConnectionError()
    assert "could not be reached" in geocoding.geocode("Place two", session=session).error


def test_geocode_input_validation_and_disable(monkeypatch):
    session = mock.Mock()
    assert not geocoding.geocode("ab", session=session).ok
    assert not geocoding.geocode("x" * 300, session=session).ok
    monkeypatch.setenv("FLOODSENSE_GEOCODER_ENABLED", "false")
    assert "turned off" in geocoding.geocode("Discovery Green", session=session).error
    session.get.assert_not_called()


def test_geocoder_configurable(monkeypatch):
    monkeypatch.setenv("FLOODSENSE_GEOCODER_URL", "https://example.org/search")
    monkeypatch.setenv("FLOODSENSE_GEOCODER_USER_AGENT", "FloodSense-test/0")
    config = geocoding.provider_config()
    assert config["url"] == "https://example.org/search" and config["user_agent"] == "FloodSense-test/0"
