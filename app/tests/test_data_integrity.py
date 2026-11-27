"""Data integrity checks for the eight validated application datasets."""

import numpy as np
import pandas as pd
import pytest

from utils import data_loader as dl
from utils.grid_utils import compute_grid_id, coverage_category
from utils.metrics import classification_metrics, confusion_counts, shap_reconstruction_error


def test_all_eight_files_present_and_load():
    assert dl.missing_data_files() == []
    assert len(dl.DATA_FILES) == 8
    assert isinstance(dl.load_metadata(), dict)
    for loader in (dl.load_locations, dl.load_grid, dl.load_historical, dl.load_historical_grid,
                   dl.load_storm_metrics, dl.load_global_shap):
        assert not loader().empty
    assert dl.load_boundaries()["type"] == "FeatureCollection"


def test_expected_counts():
    meta = dl.load_metadata()["dataset_counts"]
    locations = dl.load_locations()
    assert len(locations) == 62_138 == meta["unique_sampled_locations"]
    assert locations[["lat", "lon"]].drop_duplicates().shape[0] == 62_138
    grid = dl.load_grid()
    assert len(grid) == 4_492 == meta["geographic_grid_cells"]
    assert len(dl.load_boundaries()["features"]) == 4_492
    assert len(dl.load_historical()) == 85_678
    assert len(dl.load_historical_grid()) == 15_873
    assert len(dl.load_storm_metrics()) == 4
    assert len(dl.load_global_shap()) == 8


def test_schemas_match_metadata():
    schema = dl.load_metadata()["data_schema"]
    assert list(dl.load_locations().columns) == schema["location_predictions.parquet"]
    assert list(dl.load_grid().columns) == schema["grid_summary.parquet"]
    assert list(dl.load_historical().columns) == schema["historical_predictions.parquet"]
    assert list(dl.load_historical_grid().columns) == schema["historical_grid_summary.parquet"]
    assert list(dl.load_storm_metrics().columns) == schema["storm_metrics.csv"]
    assert list(dl.load_global_shap().columns) == schema["global_shap_importance.csv"]


def test_scores_within_unit_interval():
    for frame, col in ((dl.load_locations(), "susceptibility_score"), (dl.load_grid(), "mean_score"),
                       (dl.load_historical(), "y_prob"), (dl.load_historical_grid(), "mean_score")):
        assert frame[col].between(0, 1).all(), col


def test_no_duplicate_ids():
    assert not dl.load_grid()["grid_id"].duplicated().any()
    assert not dl.load_historical()["historical_id"].duplicated().any()
    hg = dl.load_historical_grid()
    assert not hg.duplicated(["storm_name", "grid_id"]).any()


def test_coverage_categories():
    grid = dl.load_grid()
    counts = grid["coverage_category"].value_counts().to_dict()
    assert counts == {"Greater coverage": 3548, "Limited coverage": 720, "Single sampled location": 224}
    assert (grid["sample_count"].map(coverage_category) == grid["coverage_category"]).all()


def test_grid_ids_reproduce_research_rounding():
    sample = dl.load_locations().sample(5000, random_state=0)
    rebuilt = [compute_grid_id(a, b) for a, b in zip(sample["lat"], sample["lon"])]
    assert rebuilt == sample["grid_id"].tolist()


def test_grouped_shap_reconstructs_raw_score():
    assert shap_reconstruction_error(dl.load_locations()) < 1e-4


def test_frozen_cutoff_classification():
    loc = dl.load_locations()
    assert ((loc["susceptibility_score"] >= 0.5).astype(int) == loc["predicted_at_frozen_cutoff"]).all()
    assert int(loc["predicted_at_frozen_cutoff"].sum()) == dl.load_metadata()["susceptibility"]["locations_at_or_above_cutoff"]


def test_historical_confusion_counts_match_saved_metrics():
    hist = dl.load_historical()
    for row in dl.load_storm_metrics().itertuples():
        counts = confusion_counts(hist, row.storm_name)
        assert counts == {"TP": row.tp, "TN": row.tn, "FP": row.fp, "FN": row.fn}
        derived = classification_metrics(row.tp, row.tn, row.fp, row.fn)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert derived[key] == pytest.approx(getattr(row, key), abs=1e-9)


def test_historical_grid_counts_match_observations():
    hist = dl.load_historical()
    hg = dl.load_historical_grid()
    assert hg.groupby("storm_name")["sample_count"].sum().to_dict() == hist["storm_name"].value_counts().to_dict()
    assert hg.groupby("storm_name").size().to_dict() == dl.load_storm_metrics().set_index("storm_name")["grid_count"].to_dict()


def test_imelda_matches_frozen_results():
    imelda = dl.load_storm_metrics().query("storm_name == 'Tropical Storm Imelda'").iloc[0]
    assert imelda["evaluation_protocol"] == "independent_test"
    expected = {"accuracy": 0.819923, "precision": 0.797636, "recall": 0.865483, "f1": 0.830176,
                "roc_auc": 0.889570, "pr_auc": 0.860290}
    for key, value in expected.items():
        assert imelda[key] == pytest.approx(value, abs=5e-7)
    final = dl.load_report_csv("final_test_results.csv")
    saved = final.query("model == 'lightgbm_no_rain' and cutoff_type == 'frozen'").iloc[0]
    for key in expected:
        assert imelda[key] == pytest.approx(saved[key], abs=1e-12)


def test_storm_observation_counts():
    expected = {"Memorial Day Flood": 21780, "Tax Day Flood": 21850, "Hurricane Harvey": 22884, "Tropical Storm Imelda": 19164}
    assert dl.load_storm_metrics().set_index("storm_name")["observation_count"].to_dict() == expected


def test_grouped_importance_is_not_sum_of_encoded():
    meta = dl.load_metadata()["shap"]["grouped_mean_absolute_importance"]
    encoded = dl.load_global_shap()
    land_sum = encoded[encoded["feature"].str.startswith("land cover type")]["mean_abs_shap"].sum()
    assert set(meta) == set(dl.GROUPED_FEATURES)
    assert not np.isclose(meta["land_cover_type"], land_sum)
    loc = dl.load_locations()
    assert meta["elevation_m"] == pytest.approx(loc["shap_elevation_m"].abs().mean(), rel=1e-9)
