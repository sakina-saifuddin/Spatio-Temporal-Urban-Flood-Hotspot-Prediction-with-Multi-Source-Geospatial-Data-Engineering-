"""
Prepare precomputed datasets for the FloodSense application.

This script generates application-specific susceptibility predictions,
local SHAP explanations, geographic grid summaries, and historical
storm visualization datasets.

The original research results, model configuration, and independent
test lock must remain unchanged.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    SEED,
)

from src.features import (
    get_xy,
    build_preprocessor,
)

from src.models import get_lightgbm
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import shutil
import platform
from importlib.metadata import version
# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_DIR / "docs"
APP_DATA_DIR = PROJECT_DIR / "app" / "data"
ASSETS_DIR = PROJECT_DIR / "assets"

# ============================================================
# Source files
# ============================================================

TRAINING_PATH = PROCESSED_DIR / "training_clean.csv"
TEST_PATH = PROCESSED_DIR / "test_final_clean.csv"

AREA_OOF_PATH = (
    REPORTS_DIR / "oof_lightgbm_no_rain_area_group.csv"
)

STORM_OOF_PATH = (
    REPORTS_DIR / "oof_lightgbm_no_rain_leave_one_storm_out.csv"
)

FINAL_TEST_PREDICTIONS_PATH = (
    REPORTS_DIR / "final_test_predictions_lightgbm_no_rain.csv"
)

FINAL_SETUP_PATH = REPORTS_DIR / "final_setup.csv"
FINAL_TEST_RESULTS_PATH = REPORTS_DIR / "final_test_results.csv"
FINAL_TEST_LOCK_PATH = REPORTS_DIR / "final_test.lock"

GLOBAL_SHAP_PATH = REPORTS_DIR / "shap_feature_importance.csv"
SHAP_EXAMPLES_PATH = REPORTS_DIR / "shap_examples.csv"

SOURCE_HASHES_PATH = DOCS_DIR / "app_source_hashes.csv"

MASCOT_PATH = ASSETS_DIR / "images" / "mascot.png"

# ============================================================
# Application output files
# ============================================================

GRID_SUMMARY_PATH = APP_DATA_DIR / "grid_summary.parquet"

GRID_BOUNDARIES_PATH = APP_DATA_DIR / "grid_boundaries.geojson"

LOCATION_PREDICTIONS_PATH = (
    APP_DATA_DIR / "location_predictions.parquet"
)

HISTORICAL_PREDICTIONS_PATH = (
    APP_DATA_DIR / "historical_predictions.parquet"
)

HISTORICAL_GRID_SUMMARY_PATH = (
    APP_DATA_DIR / "historical_grid_summary.parquet"
)

STORM_METRICS_PATH = APP_DATA_DIR / "storm_metrics.csv"

APP_GLOBAL_SHAP_PATH = (
    APP_DATA_DIR / "global_shap_importance.csv"
)

MODEL_METADATA_PATH = APP_DATA_DIR / "model_metadata.json"

# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "lightgbm_no_rain"

DROP_FEATURES = ("rainfall_inches",)

NUMERIC_MODEL_FEATURES = (
    "elevation_m",
    "slope_degrees",
    "distance_to_bayou_m",
    "paved_pct",
)

CATEGORICAL_MODEL_FEATURES = (
    "land_cover_type",
)

MODEL_FEATURES = (
    *NUMERIC_MODEL_FEATURES,
    *CATEGORICAL_MODEL_FEATURES,
)

TARGET_COLUMN = "flooded"

EXPECTED_FROZEN_CUTOFF = 0.50

EXPECTED_RANDOM_SEED = 42

GRID_CELL_DEGREES = 0.01
# ============================================================
# Expected dataset sizes
# ============================================================

EXPECTED_TRAINING_ROWS = 66514
EXPECTED_UNIQUE_LOCATIONS = 62138
EXPECTED_GRID_CELLS = 4492

EXPECTED_TEST_ROWS = 19164
EXPECTED_TEST_GRID_CELLS = 3854

EXPECTED_HISTORICAL_ROWS = 85678
EXPECTED_PROTECTED_FILES = 28

EXPECTED_STORM_COUNTS = {
    "Memorial Day Flood": 21780,
    "Tax Day Flood": 21850,
    "Hurricane Harvey": 22884,
    "Tropical Storm Imelda": 19164,
}

STORM_YEARS = {
    "Memorial Day Flood": 2015,
    "Tax Day Flood": 2016,
    "Hurricane Harvey": 2017,
    "Tropical Storm Imelda": 2019,
}
# ============================================================
# Source integrity verification
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest().upper()
def verify_source_integrity() -> None:
    """Verify protected files against the saved SHA-256 baseline."""

    print("\nVerifying protected source files...")

    if not SOURCE_HASHES_PATH.is_file():
        raise FileNotFoundError(
            f"Source integrity manifest not found: {SOURCE_HASHES_PATH}"
        )

    manifest = pd.read_csv(
        SOURCE_HASHES_PATH,
        encoding="utf-8-sig",
        dtype=str
    )

    required_columns = {"File", "SHA256"}

    if not required_columns.issubset(manifest.columns):
        raise ValueError(
            "Source integrity manifest has missing required columns."
        )

    if len(manifest) != EXPECTED_PROTECTED_FILES:
        raise ValueError(
            f"Expected {EXPECTED_PROTECTED_FILES} protected files, "
            f"but found {len(manifest)}."
        )

    if manifest["File"].isna().any():
        raise ValueError("Missing file paths in integrity manifest.")

    if manifest["File"].duplicated().any():
        raise ValueError("Duplicate file paths in integrity manifest.")

    valid_hashes = manifest["SHA256"].str.fullmatch(
        r"[A-Fa-f0-9]{64}", na=False
    )

    if not valid_hashes.all():
        raise ValueError("Invalid SHA-256 hashes in integrity manifest.")

    failures = []
    verified_count = 0

    for _, row in manifest.iterrows():

        relative_path = row["File"]
        expected_hash = row["SHA256"].upper()

        file_path = (PROJECT_DIR / relative_path).resolve()

        if not file_path.is_relative_to(PROJECT_DIR):
            failures.append(
                f"Invalid source path: {relative_path}"
            )
            continue

        if not file_path.is_file():
            failures.append(
                f"Missing file: {relative_path}"
            )
            continue

        current_hash = calculate_file_hash(file_path)

        if current_hash != expected_hash:
            failures.append(
                f"Hash mismatch: {relative_path}"
            )
        else:
            verified_count += 1
            print(f"PASS: {relative_path}")

    print("\nSource integrity verification summary:")
    print("Expected protected files:", EXPECTED_PROTECTED_FILES)
    print("Successfully verified:", verified_count)
    print("Failed verification:", len(failures))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")

        raise RuntimeError(
            "Source integrity verification failed. "
            "Application preparation stopped."
        )

    print("All protected source files match the baseline.")
# ============================================================
# Research component verification
# ============================================================

def verify_research_components() -> None:
    """Verify compatibility with the original research configuration."""

    print("\nVerifying research model components...")

    numeric_features = tuple(
        feature for feature in NUMERIC_FEATURES
        if feature not in DROP_FEATURES
    )

    categorical_features = tuple(
        feature for feature in CATEGORICAL_FEATURES
        if feature not in DROP_FEATURES
    )

    if numeric_features != NUMERIC_MODEL_FEATURES:
        raise ValueError("Numerical model features do not match.")

    if categorical_features != CATEGORICAL_MODEL_FEATURES:
        raise ValueError("Categorical model features do not match.")

    if TARGET != TARGET_COLUMN:
        raise ValueError("Target variable does not match.")

    if SEED != EXPECTED_RANDOM_SEED:
        raise ValueError("Random seed does not match.")

    model = get_lightgbm()
    parameters = model.get_params()

    expected_parameters = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "random_state": EXPECTED_RANDOM_SEED,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
    }

    for parameter, expected_value in expected_parameters.items():

        actual_value = parameters.get(parameter)

        if actual_value != expected_value:
            raise ValueError(
                f"Unexpected model parameter: {parameter}. "
                f"Expected {expected_value}, found {actual_value}."
            )

        print(f"PASS: {parameter} = {actual_value}")

    print("\nNumerical features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Target:", TARGET)
    print("Random seed:", SEED)

    print("Research model components verified successfully.")



def main():
    """Run the complete FloodSense data preparation pipeline."""

    print("=" * 60)
    print("FloodSense Application Data Preparation")
    print("=" * 60)

    # Research protection
    verify_source_integrity()
    verify_research_components()

    # Batch 1: Model fitting and predictions
    frozen_configuration = load_frozen_configuration()

    cutoff = float(
        frozen_configuration["cutoff"]
    )

    training = load_development_data()

    pipeline = fit_application_model(training)

    locations = generate_location_predictions(
        training,
        pipeline,
        cutoff
    )

    print("\nBATCH 1 COMPLETED SUCCESSFULLY")

    # Batch 2: SHAP and geographic aggregation
    locations = calculate_local_shap(
        locations,
        pipeline
    )

    grid_summary = aggregate_grid_summaries(
        locations,
        training
    )

    grid_boundaries = generate_grid_boundaries(
        grid_summary
    )

    print("\nBATCH 2 COMPLETED SUCCESSFULLY")

    # Batch 3: Historical data
    historical = prepare_historical_predictions(
        training,
        cutoff
    )

    historical_grids = aggregate_historical_grids(
        historical
    )

    storm_metrics = calculate_storm_metrics(
        historical,
        historical_grids,
        cutoff
    )

    print("\nBATCH 3 COMPLETED SUCCESSFULLY")

    # Batch 4: Metadata and exports
    global_shap = prepare_global_shap()

    metadata = generate_model_metadata(
        frozen_configuration,
        pipeline,
        locations,
        grid_summary,
        historical,
        historical_grids,
        storm_metrics,
        global_shap
    )

    export_application_data(
        locations,
        grid_summary,
        grid_boundaries,
        historical,
        historical_grids,
        storm_metrics,
        global_shap,
        metadata
    )

    validate_application_exports(
        locations,
        grid_summary,
        grid_boundaries,
        historical,
        historical_grids,
        storm_metrics,
        global_shap,
        metadata
    )

    print("\nBATCH 4 COMPLETED SUCCESSFULLY")

    # Batch 5: Documentation and final research protection
    write_application_documentation(metadata)

    verify_source_integrity()

    print("\n" + "=" * 60)
    print("APPLICATION DATA PREPARATION COMPLETED")
    print("=" * 60)

    print("Model:", MODEL_NAME)
    print("Training observations:", len(training))
    print("Unique location predictions:", len(locations))
    print("Local SHAP explanations:", len(locations))
    print("Geographic grid summaries:", len(grid_summary))
    print("GeoJSON polygons:", len(grid_boundaries["features"]))
    print("Historical observations:", len(historical))
    print("Historical grid-storm summaries:", len(historical_grids))
    print("Historical storm metrics:", len(storm_metrics))
    print("Application data files: 8")
    print("Application documentation generated: Yes")
    print("Frozen cutoff:", cutoff)
    print("Independent test evaluation rerun: No")
    print("Protected research files modified: No")
    print("Application data ready for development: Yes")




# ============================================================
# Frozen model configuration
# ============================================================

def load_frozen_configuration() -> dict:
    """Load and validate the frozen LightGBM configuration."""

    print("\nLoading frozen model configuration...")

    setup = pd.read_csv(FINAL_SETUP_PATH)

    required_columns = {
        "model",
        "features_used",
        "cutoff",
        "area_recall_at_cutoff",
        "area_precision_at_cutoff",
        "storm_recall_at_cutoff",
        "storm_precision_at_cutoff",
        "role",
    }

    if not required_columns.issubset(setup.columns):
        raise ValueError("Missing frozen configuration columns.")

    selected = setup.loc[
        (setup["model"] == MODEL_NAME)
        & (setup["role"] == "final")
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one final LightGBM configuration."
        )

    configuration = selected.iloc[0].to_dict()

    feature_description = configuration["features_used"]

    expected_description = (
        ", ".join(MODEL_FEATURES) + "; no rainfall_inches"
    )

    if feature_description != expected_description:
        raise ValueError("Frozen model features do not match.")

    cutoff = float(configuration["cutoff"])

    if not np.isclose(cutoff, EXPECTED_FROZEN_CUTOFF):
        raise ValueError("Frozen classification cutoff does not match.")

    metric_columns = (
        "area_recall_at_cutoff",
        "area_precision_at_cutoff",
        "storm_recall_at_cutoff",
        "storm_precision_at_cutoff",
    )

    for column in metric_columns:
        value = float(configuration[column])

        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid frozen metric: {column}")

    print("Selected model:", configuration["model"])
    print("Model role:", configuration["role"])
    print("Frozen cutoff:", cutoff)

    print(
        "Geographic recall:",
        f"{configuration['area_recall_at_cutoff']:.2%}"
    )

    print(
        "Geographic precision:",
        f"{configuration['area_precision_at_cutoff']:.2%}"
    )

    print(
        "Cross-storm recall:",
        f"{configuration['storm_recall_at_cutoff']:.2%}"
    )

    print(
        "Cross-storm precision:",
        f"{configuration['storm_precision_at_cutoff']:.2%}"
    )

    print("Frozen model configuration validated successfully.")

    return configuration


# ============================================================
# Development dataset
# ============================================================

def load_development_data() -> pd.DataFrame:
    """Load and validate the complete development dataset."""

    print("\nLoading development dataset...")

    training = pd.read_csv(TRAINING_PATH)

    required_columns = [
        "lat",
        "lon",
        "grid_id",
        "storm_name",
        TARGET_COLUMN,
        *MODEL_FEATURES,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in training.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if len(training) != EXPECTED_TRAINING_ROWS:
        raise ValueError("Unexpected development dataset size.")

    if training[required_columns].isna().any().any():
        raise ValueError("Missing required development values.")

    numeric_columns = [
        "lat",
        "lon",
        *NUMERIC_MODEL_FEATURES,
    ]

    if not np.isfinite(
        training[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numerical values detected.")

    if set(training[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Unexpected target labels.")

    expected_development_storms = {
        storm: count
        for storm, count in EXPECTED_STORM_COUNTS.items()
        if storm != "Tropical Storm Imelda"
    }

    actual_storm_counts = (
        training["storm_name"].value_counts().to_dict()
    )

    if actual_storm_counts != expected_development_storms:
        raise ValueError("Unexpected development storm counts.")

    if training.duplicated(
        subset=["lat", "lon", "storm_name"]
    ).any():
        raise ValueError("Duplicate location-storm observations.")

    if training["grid_id"].nunique() != EXPECTED_GRID_CELLS:
        raise ValueError("Unexpected geographic grid count.")

    reconstructed_grid = (
        (training["lat"] / GRID_CELL_DEGREES)
        .round()
        .astype("Int64")
        .astype("string")
        + "_"
        + (training["lon"] / GRID_CELL_DEGREES)
        .round()
        .astype("Int64")
        .astype("string")
    )

    if not reconstructed_grid.eq(
        training["grid_id"].astype("string")
    ).all():
        raise ValueError("Geographic grid reconstruction mismatch.")

    print("Development observations:", len(training))
    print("Missing required values: 0")
    print("Duplicate location-storm observations: 0")
    print("Geographic grid cells:", training["grid_id"].nunique())

    print("\nTarget distribution:")
    print(training[TARGET_COLUMN].value_counts().sort_index())

    print("\nDevelopment storm distribution:")
    print(training["storm_name"].value_counts())

    print("\nDevelopment dataset validated successfully.")

    return training


# ============================================================
# Application model fitting
# ============================================================

def fit_application_model(training: pd.DataFrame) -> Pipeline:
    """Fit the frozen LightGBM configuration on development data."""

    print("\nPreparing model features and target...")

    X, y = get_xy(
        training,
        drop_features=DROP_FEATURES
    )

    if tuple(X.columns) != MODEL_FEATURES:
        raise ValueError("Extracted model features do not match.")

    if not X.index.equals(y.index):
        raise ValueError("Feature and target indices do not match.")

    if len(X) != EXPECTED_TRAINING_ROWS:
        raise ValueError("Unexpected model training size.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing model inputs.")

    if not y.equals(training[TARGET_COLUMN]):
        raise ValueError("Extracted target labels do not match.")

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Model features:", X.columns.tolist())

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor(drop_features=DROP_FEATURES)
        ),
        (
            "model",
            get_lightgbm()
        ),
    ])

    print("\nFitting frozen LightGBM configuration...")
    print("Training observations:", len(X))

    pipeline.fit(X, y)

    fitted_model = pipeline.named_steps["model"]

    if set(fitted_model.classes_) != {0, 1}:
        raise ValueError("Unexpected fitted model classes.")

    transformed = pipeline.named_steps[
        "preprocessor"
    ].transform(X.iloc[:10])

    if transformed.shape[0] != 10:
        raise ValueError("Unexpected transformed sample size.")

    print("Model classes:", fitted_model.classes_)
    print("Transformed feature count:", transformed.shape[1])
    print("Fitted estimators:", fitted_model.n_estimators_)

    print("Application model fitted successfully.")

    return pipeline


# ============================================================
# Unique-location susceptibility predictions
# ============================================================

def generate_location_predictions(
    training: pd.DataFrame,
    pipeline: Pipeline,
    cutoff: float
) -> pd.DataFrame:
    """Generate predictions for unique sampled locations."""

    print("\nPreparing unique sampled locations...")

    coordinate_columns = ["lat", "lon"]

    consistency_columns = [
        "grid_id",
        *MODEL_FEATURES,
    ]

    consistency = (
        training
        .groupby(coordinate_columns)[consistency_columns]
        .nunique(dropna=False)
    )

    if consistency.gt(1).any().any():
        raise ValueError(
            "Repeated coordinates contain inconsistent features."
        )

    location_columns = [
        "lat",
        "lon",
        "grid_id",
        *MODEL_FEATURES,
    ]

    locations = (
        training[location_columns]
        .drop_duplicates(subset=coordinate_columns)
        .reset_index(drop=True)
        .copy()
    )

    if len(locations) != EXPECTED_UNIQUE_LOCATIONS:
        raise ValueError("Unexpected unique location count.")

    if locations["grid_id"].nunique() != EXPECTED_GRID_CELLS:
        raise ValueError("Unexpected unique-location grid count.")

    if locations.duplicated(coordinate_columns).any():
        raise ValueError("Duplicate coordinates remain.")

    print("Original development rows:", len(training))
    print("Unique sampled locations:", len(locations))
    print(
        "Repeated coordinate records removed:",
        len(training) - len(locations)
    )

    print("\nGenerating susceptibility predictions...")

    X_locations = locations.loc[:, list(MODEL_FEATURES)]

    classes = pipeline.named_steps["model"].classes_

    positive_class_index = np.flatnonzero(classes == 1)

    if len(positive_class_index) != 1:
        raise ValueError("Unable to identify positive model class.")

    probabilities = pipeline.predict_proba(X_locations)

    scores = probabilities[:, positive_class_index[0]]

    if len(scores) != EXPECTED_UNIQUE_LOCATIONS:
        raise ValueError("Unexpected prediction count.")

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores detected.")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Prediction scores outside valid range.")

    locations["susceptibility_score"] = scores

    locations["predicted_at_frozen_cutoff"] = (
        locations["susceptibility_score"] >= cutoff
    ).astype(int)

    print("\nSusceptibility prediction summary:")

    print(
        locations["susceptibility_score"]
        .describe()
        .to_string()
    )

    print("\nClassification distribution at frozen cutoff:")
    print(
        locations["predicted_at_frozen_cutoff"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nLocations at or above frozen cutoff:",
        f"{locations['predicted_at_frozen_cutoff'].mean():.2%}"
    )

    print("Missing prediction scores:", np.isnan(scores).sum())
    print("Prediction scores within [0, 1]: True")
    print("Unique-location predictions generated successfully.")

    return locations

# ============================================================
# Local SHAP explanations
# ============================================================

def calculate_local_shap(
    locations: pd.DataFrame,
    pipeline: Pipeline
) -> pd.DataFrame:
    """Calculate and validate local SHAP explanations."""

    print("\nCalculating local SHAP explanations...")

    locations = locations.copy()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_locations = locations.loc[:, list(MODEL_FEATURES)]

    transformed = preprocessor.transform(X_locations)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()

    if transformed.shape[1] != len(feature_names):
        raise ValueError("Transformed feature count mismatch.")

    print("Transformed matrix shape:", transformed.shape)
    print("Encoded feature names:", feature_names.tolist())

    explainer = shap.TreeExplainer(model)

    explanation = explainer(transformed)

    shap_values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if shap_values.shape != transformed.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    if base_values.size == 1:
        base_values = np.full(
            len(locations),
            float(base_values.reshape(-1)[0])
        )
    else:
        base_values = base_values.reshape(-1)

    if len(base_values) != len(locations):
        raise ValueError("Unexpected SHAP base value count.")

    if not np.isfinite(shap_values).all():
        raise ValueError("Non-finite SHAP values detected.")

    if not np.isfinite(base_values).all():
        raise ValueError("Non-finite SHAP base values detected.")

    # Validate SHAP additivity against model raw scores.
    raw_scores = np.asarray(
        model.predict(transformed, raw_score=True)
    ).reshape(-1)

    reconstructed_scores = (
        base_values + shap_values.sum(axis=1)
    )

    additivity_error = np.abs(
        reconstructed_scores - raw_scores
    )

    if not np.allclose(
        reconstructed_scores,
        raw_scores,
        atol=1e-5,
        rtol=1e-5
    ):
        raise ValueError("SHAP additivity validation failed.")

    # Validate raw scores against predicted probabilities.
    reconstructed_probabilities = (
        1.0 / (1.0 + np.exp(-raw_scores))
    )

    if not np.allclose(
        reconstructed_probabilities,
        locations["susceptibility_score"].to_numpy(),
        atol=1e-7,
        rtol=1e-7
    ):
        raise ValueError("Raw score probability mismatch.")

    # Group encoded features into the original five features.
    clean_names = [
        str(name).split("__", 1)[-1]
        for name in feature_names
    ]

    used_indices = []

    for feature in NUMERIC_MODEL_FEATURES:

        indices = [
            i for i, name in enumerate(clean_names)
            if name == feature
        ]

        if len(indices) != 1:
            raise ValueError(
                f"Unable to identify SHAP feature: {feature}"
            )

        locations[f"shap_{feature}"] = (
            shap_values[:, indices[0]]
        )

        used_indices.extend(indices)

    land_cover_indices = [
        i for i, name in enumerate(clean_names)
        if name.startswith("land_cover_type_")
    ]

    if not land_cover_indices:
        raise ValueError(
            "Unable to identify encoded land cover features."
        )

    locations["shap_land_cover_type"] = (
        shap_values[:, land_cover_indices].sum(axis=1)
    )

    used_indices.extend(land_cover_indices)

    if sorted(used_indices) != list(
        range(transformed.shape[1])
    ):
        raise ValueError(
            "Not all encoded SHAP features were accounted for."
        )

    locations["shap_base_value"] = base_values
    locations["raw_model_score"] = raw_scores

    shap_columns = [
        f"shap_{feature}"
        for feature in MODEL_FEATURES
    ]

    grouped_reconstruction = (
        locations["shap_base_value"]
        + locations[shap_columns].sum(axis=1)
    )

    if not np.allclose(
        grouped_reconstruction,
        raw_scores,
        atol=1e-5,
        rtol=1e-5
    ):
        raise ValueError("Grouped SHAP additivity failed.")

    print("SHAP values shape:", shap_values.shape)
    print("Grouped SHAP features:", len(shap_columns))
    print("Locations explained:", len(locations))
    print("Maximum SHAP additivity error:", additivity_error.max())
    print("Raw score probability validation: PASS")
    print("Grouped SHAP additivity validation: PASS")

    print("\nMean absolute grouped SHAP values:")

    print(
        locations[shap_columns]
        .abs()
        .mean()
        .sort_values(ascending=False)
    )

    print("Local SHAP explanations calculated successfully.")

    return locations

# ============================================================
# Geographic grid summaries
# ============================================================

def aggregate_grid_summaries(
    locations: pd.DataFrame,
    training: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate predictions and explanations by geographic cell."""

    print("\nAggregating geographic grid summaries...")

    shap_columns = [
        f"shap_{feature}"
        for feature in MODEL_FEATURES
    ]

    grouped = locations.groupby("grid_id", sort=True)

    grid_summary = grouped.agg(
        sample_count=("susceptibility_score", "size"),

        mean_score=("susceptibility_score", "mean"),
        min_score=("susceptibility_score", "min"),
        max_score=("susceptibility_score", "max"),
        median_score=("susceptibility_score", "median"),
        std_score=("susceptibility_score", "std"),

        locations_above_cutoff=(
            "predicted_at_frozen_cutoff", "sum"
        ),

        share_above_cutoff=(
            "predicted_at_frozen_cutoff", "mean"
        ),

        mean_elevation_m=("elevation_m", "mean"),
        mean_slope_degrees=("slope_degrees", "mean"),
        mean_distance_to_bayou_m=(
            "distance_to_bayou_m", "mean"
        ),
        mean_paved_pct=("paved_pct", "mean"),

        dominant_land_cover=(
            "land_cover_type",
            lambda values: values.mode().iloc[0]
        ),

        mean_raw_model_score=("raw_model_score", "mean"),
        mean_shap_base_value=("shap_base_value", "mean"),
    )

    # A single sampled location has zero within-cell variation.
    grid_summary["std_score"] = (
        grid_summary["std_score"].fillna(0.0)
    )

    mean_shap = grouped[shap_columns].mean()

    grid_summary = grid_summary.join(mean_shap)

    # Historical development report-associated observations.
    historical_counts = training.groupby("grid_id").agg(
        development_observation_count=(
            TARGET_COLUMN, "size"
        ),
        development_report_positive_count=(
            TARGET_COLUMN, "sum"
        )
    )

    grid_summary = grid_summary.join(historical_counts)

    # Descriptive coverage categories, not confidence levels.
    grid_summary["coverage_category"] = np.select(
        [
            grid_summary["sample_count"] == 1,
            grid_summary["sample_count"].between(2, 4),
            grid_summary["sample_count"] >= 5,
        ],
        [
            "Single sampled location",
            "Limited coverage",
            "Greater coverage",
        ],
        default="No data"
    )

    # Identify the strongest positive and negative contributions.
    grid_summary["dominant_positive_factor"] = (
        grid_summary[shap_columns]
        .idxmax(axis=1)
        .str.removeprefix("shap_")
    )

    grid_summary.loc[
        grid_summary[shap_columns].max(axis=1) <= 0,
        "dominant_positive_factor"
    ] = "None"

    grid_summary["dominant_negative_factor"] = (
        grid_summary[shap_columns]
        .idxmin(axis=1)
        .str.removeprefix("shap_")
    )

    grid_summary.loc[
        grid_summary[shap_columns].min(axis=1) >= 0,
        "dominant_negative_factor"
    ] = "None"

    grid_summary = grid_summary.reset_index()

    # Reconstruct grid centers from the original grid identifiers.
    grid_coordinates = (
        grid_summary["grid_id"]
        .astype(str)
        .str.split("_", expand=True)
    )

    if grid_coordinates.shape[1] != 2:
        raise ValueError("Unexpected geographic grid ID format.")

    grid_summary["center_lat"] = (
        grid_coordinates[0].astype(int) * GRID_CELL_DEGREES
    ).round(6)

    grid_summary["center_lon"] = (
        grid_coordinates[1].astype(int) * GRID_CELL_DEGREES
    ).round(6)

    half_cell = GRID_CELL_DEGREES / 2

    grid_summary["south"] = (
        grid_summary["center_lat"] - half_cell
    ).round(6)

    grid_summary["north"] = (
        grid_summary["center_lat"] + half_cell
    ).round(6)

    grid_summary["west"] = (
        grid_summary["center_lon"] - half_cell
    ).round(6)

    grid_summary["east"] = (
        grid_summary["center_lon"] + half_cell
    ).round(6)

    # Validate aggregate counts and geographic coverage.
    if len(grid_summary) != EXPECTED_GRID_CELLS:
        raise ValueError("Unexpected grid summary count.")

    if grid_summary["grid_id"].duplicated().any():
        raise ValueError("Duplicate geographic grid IDs.")

    if grid_summary["sample_count"].sum() != EXPECTED_UNIQUE_LOCATIONS:
        raise ValueError("Aggregated sample count mismatch.")

    if (
        grid_summary["development_observation_count"].sum()
        != EXPECTED_TRAINING_ROWS
    ):
        raise ValueError("Aggregated development count mismatch.")

    if (
        grid_summary["development_report_positive_count"].sum()
        != training[TARGET_COLUMN].sum()
    ):
        raise ValueError("Aggregated positive report count mismatch.")

    if (
        grid_summary["locations_above_cutoff"].sum()
        != locations["predicted_at_frozen_cutoff"].sum()
    ):
        raise ValueError("Aggregated classification count mismatch.")

    score_columns = [
        "mean_score",
        "min_score",
        "max_score",
        "median_score",
        "share_above_cutoff",
    ]

    if not (
        grid_summary[score_columns].ge(0)
        & grid_summary[score_columns].le(1)
    ).all().all():
        raise ValueError("Invalid aggregated score range.")

    if not (
        grid_summary["min_score"]
        <= grid_summary["mean_score"]
    ).all():
        raise ValueError("Invalid minimum and mean scores.")

    if not (
        grid_summary["mean_score"]
        <= grid_summary["max_score"]
    ).all():
        raise ValueError("Invalid mean and maximum scores.")

    # Validate cell-level SHAP additivity.
    reconstructed_raw = (
        grid_summary["mean_shap_base_value"]
        + grid_summary[shap_columns].sum(axis=1)
    )

    if not np.allclose(
        reconstructed_raw,
        grid_summary["mean_raw_model_score"],
        atol=1e-5,
        rtol=1e-5
    ):
        raise ValueError("Grid-level SHAP additivity failed.")

    # Verify each sampled location falls within its assigned cell.
    bounds = locations[
        ["lat", "lon", "grid_id"]
    ].merge(
        grid_summary[
            ["grid_id", "south", "north", "west", "east"]
        ],
        on="grid_id",
        how="left",
        validate="many_to_one"
    )

    tolerance = 1e-8

    within_bounds = (
        (bounds["lat"] >= bounds["south"] - tolerance)
        & (bounds["lat"] <= bounds["north"] + tolerance)
        & (bounds["lon"] >= bounds["west"] - tolerance)
        & (bounds["lon"] <= bounds["east"] + tolerance)
    )

    if not within_bounds.all():
        raise ValueError("Sampled locations outside assigned cells.")

    if grid_summary.isna().any().any():
        raise ValueError("Missing geographic summary values.")

    print("Geographic grid summaries:", len(grid_summary))
    print("Total sampled locations:", grid_summary["sample_count"].sum())

    print("\nData coverage distribution:")
    print(grid_summary["coverage_category"].value_counts())

    print("\nMean grid susceptibility score distribution:")
    print(grid_summary["mean_score"].describe())

    print("\nDominant positive contributing factors:")
    print(
        grid_summary["dominant_positive_factor"].value_counts()
    )

    print("Grid-level SHAP additivity validation: PASS")
    print("Geographic location containment validation: PASS")
    print("Geographic grid summaries generated successfully.")

    return grid_summary

# ============================================================
# Geographic grid boundaries
# ============================================================

def generate_grid_boundaries(
    grid_summary: pd.DataFrame
) -> dict:
    """Generate GeoJSON polygons for sampled geographic cells."""

    print("\nGenerating geographic grid boundaries...")

    features = []

    for row in grid_summary.itertuples(index=False):

        west = float(row.west)
        east = float(row.east)
        south = float(row.south)
        north = float(row.north)

        if west >= east or south >= north:
            raise ValueError(
                f"Invalid grid boundaries: {row.grid_id}"
            )

        # GeoJSON uses longitude, latitude coordinate order.
        coordinates = [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]]

        feature = {
            "type": "Feature",
            "id": str(row.grid_id),

            "properties": {
                "grid_id": str(row.grid_id),
                "sample_count": int(row.sample_count),
                "mean_score": float(row.mean_score),
                "coverage_category": str(
                    row.coverage_category
                ),
                "dominant_positive_factor": str(
                    row.dominant_positive_factor
                ),
            },

            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
        }

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    if len(features) != EXPECTED_GRID_CELLS:
        raise ValueError("Unexpected GeoJSON feature count.")

    feature_ids = {
        feature["id"]
        for feature in features
    }

    if len(feature_ids) != EXPECTED_GRID_CELLS:
        raise ValueError("Duplicate GeoJSON feature identifiers.")

    if feature_ids != set(
        grid_summary["grid_id"].astype(str)
    ):
        raise ValueError("GeoJSON grid identifier mismatch.")

    # Confirm the object is valid JSON without NaN values.
    json.dumps(geojson, allow_nan=False)

    print("GeoJSON feature count:", len(features))
    print("Unique geographic identifiers:", len(feature_ids))
    print("Geometry type: Polygon")
    print("Grid dimensions: 0.01 x 0.01 degrees")
    print("GeoJSON serialization validation: PASS")
    print("Geographic grid boundaries generated successfully.")

    return geojson

# ============================================================
# Historical prediction alignment
# ============================================================

def align_historical_predictions(
    source: pd.DataFrame,
    prediction_path: Path,
    evaluation_protocol: str
) -> pd.DataFrame:
    """Align saved predictions with original source observations."""

    print("\nAligning:", prediction_path.name)

    predictions = pd.read_csv(prediction_path)

    required_prediction_columns = {
        "row_index",
        "y_true",
        "y_prob",
    }

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise ValueError("Missing historical prediction columns.")

    if len(predictions) != len(source):
        raise ValueError("Historical prediction count mismatch.")

    if predictions["row_index"].isna().any():
        raise ValueError("Missing historical row indices.")

    if not pd.api.types.is_integer_dtype(
        predictions["row_index"]
    ):
        raise ValueError("Historical row indices must be integers.")

    if predictions["row_index"].duplicated().any():
        raise ValueError("Duplicate historical row indices.")

    if not np.array_equal(
        np.sort(predictions["row_index"].to_numpy()),
        np.arange(len(source))
    ):
        raise ValueError("Incomplete historical row coverage.")

    if not predictions["y_true"].isin([0, 1]).all():
        raise ValueError("Invalid historical target labels.")

    scores = predictions["y_prob"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite historical predictions.")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid historical prediction range.")

    source_columns = [
        "year",
        "storm_name",
        "lat",
        "lon",
        "grid_id",
        TARGET_COLUMN,
        *MODEL_FEATURES,
    ]

    if not set(source_columns).issubset(source.columns):
        raise ValueError("Missing historical source columns.")

    source_rows = (
        source[source_columns]
        .reset_index(drop=True)
        .copy()
    )

    if source_rows.isna().any().any():
        raise ValueError("Missing historical source values.")

    source_rows["row_index"] = np.arange(len(source_rows))

    prediction_columns = [
        "row_index",
        "y_true",
        "y_prob",
    ]

    if "fold" in predictions.columns:
        prediction_columns.append("fold")

    aligned = source_rows.merge(
        predictions[prediction_columns],
        on="row_index",
        how="left",
        validate="one_to_one"
    )

    if not aligned[TARGET_COLUMN].eq(
        aligned["y_true"]
    ).all():
        raise ValueError("Historical target alignment mismatch.")

    if aligned["y_prob"].isna().any():
        raise ValueError("Missing aligned historical predictions.")

    if evaluation_protocol == "leave_one_storm_out":

        if "fold" not in aligned.columns:
            raise ValueError("Missing storm validation folds.")

        if aligned["fold"].isna().any():
            raise ValueError("Missing storm fold values.")

        storms_per_fold = (
            aligned.groupby("fold")["storm_name"].nunique()
        )

        folds_per_storm = (
            aligned.groupby("storm_name")["fold"].nunique()
        )

        if not storms_per_fold.eq(1).all():
            raise ValueError("Multiple storms in one validation fold.")

        if not folds_per_storm.eq(1).all():
            raise ValueError("Storm assigned to multiple folds.")

        if aligned["fold"].nunique() != 3:
            raise ValueError("Unexpected storm validation fold count.")

    aligned["evaluation_protocol"] = evaluation_protocol

    # Use a consistent text representation for fold identifiers.
    if "fold" in aligned.columns:
        aligned["fold"] = aligned["fold"].astype(str)
    else:
        aligned["fold"] = "independent_test"

    aligned = aligned.drop(columns=[TARGET_COLUMN])

    print("Aligned observations:", len(aligned))
    print("Label mismatches: 0")
    print("Missing prediction scores: 0")
    print("Evaluation protocol:", evaluation_protocol)

    return aligned

# ============================================================
# Historical storm dataset
# ============================================================

def prepare_historical_predictions(
    training: pd.DataFrame,
    cutoff: float
) -> pd.DataFrame:
    """Prepare saved predictions for all four historical storms."""

    print("\nPreparing historical storm predictions...")

    development_history = align_historical_predictions(
        training,
        STORM_OOF_PATH,
        "leave_one_storm_out"
    )

    test = pd.read_csv(TEST_PATH)

    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Unexpected independent test size.")

    if set(test["storm_name"].unique()) != {
        "Tropical Storm Imelda"
    }:
        raise ValueError("Unexpected independent test storm.")

    if test["grid_id"].nunique() != EXPECTED_TEST_GRID_CELLS:
        raise ValueError("Unexpected independent test grid count.")

    independent_history = align_historical_predictions(
        test,
        FINAL_TEST_PREDICTIONS_PATH,
        "independent_test"
    )

    historical = pd.concat(
        [development_history, independent_history],
        ignore_index=True
    )

    if len(historical) != EXPECTED_HISTORICAL_ROWS:
        raise ValueError("Unexpected total historical count.")

    actual_storm_counts = (
        historical["storm_name"].value_counts().to_dict()
    )

    if actual_storm_counts != EXPECTED_STORM_COUNTS:
        raise ValueError("Historical storm count mismatch.")

    expected_years = historical["storm_name"].map(STORM_YEARS)

    if not historical["year"].eq(expected_years).all():
        raise ValueError("Historical storm year mismatch.")

    if historical.duplicated(
        ["lat", "lon", "storm_name"]
    ).any():
        raise ValueError("Duplicate historical location-storm rows.")

    historical["historical_id"] = (
        historical["year"].astype(str)
        + "_"
        + historical["row_index"].astype(str)
    )

    if historical["historical_id"].duplicated().any():
        raise ValueError("Duplicate historical identifiers.")

    historical["y_pred"] = (
        historical["y_prob"] >= cutoff
    ).astype(int)

    historical["classification"] = np.select(
        [
            (historical["y_true"] == 1)
            & (historical["y_pred"] == 1),

            (historical["y_true"] == 0)
            & (historical["y_pred"] == 0),

            (historical["y_true"] == 0)
            & (historical["y_pred"] == 1),

            (historical["y_true"] == 1)
            & (historical["y_pred"] == 0),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="Invalid"
    )

    if historical["classification"].eq("Invalid").any():
        raise ValueError("Invalid historical classification.")

    historical["classification_correct"] = (
        historical["y_true"] == historical["y_pred"]
    ).astype(int)

    historical["frozen_cutoff"] = cutoff

    reconstructed_grid = (
        (historical["lat"] / GRID_CELL_DEGREES)
        .round()
        .astype("Int64")
        .astype("string")
        + "_"
        + (historical["lon"] / GRID_CELL_DEGREES)
        .round()
        .astype("Int64")
        .astype("string")
    )

    if not reconstructed_grid.eq(
        historical["grid_id"].astype("string")
    ).all():
        raise ValueError("Historical geographic grid mismatch.")

    historical = historical.sort_values(
        ["year", "row_index"]
    ).reset_index(drop=True)

    if historical.isna().any().any():
        raise ValueError("Missing historical dataset values.")

    print("\nHistorical observations:", len(historical))

    print("\nHistorical storm distribution:")
    print(historical["storm_name"].value_counts())

    print("\nHistorical classification distribution:")
    print(
        pd.crosstab(
            historical["storm_name"],
            historical["classification"]
        )
    )

    print("\nHistorical evaluation protocols:")
    print(historical["evaluation_protocol"].value_counts())

    print("Historical predictions prepared successfully.")

    return historical

# ============================================================
# Historical geographic summaries
# ============================================================

def aggregate_historical_grids(
    historical: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate historical predictions by storm and grid cell."""

    print("\nAggregating historical geographic summaries...")

    working = historical.copy()

    for category in ["TP", "TN", "FP", "FN"]:
        working[f"is_{category.lower()}"] = (
            working["classification"] == category
        ).astype(int)

    historical_grids = (
        working.groupby(
            ["year", "storm_name", "grid_id"],
            sort=True
        )
        .agg(
            evaluation_protocol=(
                "evaluation_protocol", "first"
            ),

            sample_count=("y_true", "size"),

            mean_score=("y_prob", "mean"),
            min_score=("y_prob", "min"),
            max_score=("y_prob", "max"),

            report_positive_count=("y_true", "sum"),
            report_positive_share=("y_true", "mean"),

            predicted_positive_count=("y_pred", "sum"),
            predicted_positive_share=("y_pred", "mean"),

            correct_count=("classification_correct", "sum"),
            accuracy=("classification_correct", "mean"),

            tp_count=("is_tp", "sum"),
            tn_count=("is_tn", "sum"),
            fp_count=("is_fp", "sum"),
            fn_count=("is_fn", "sum"),
        )
        .reset_index()
    )

    for category in ["tp", "tn", "fp", "fn"]:

        historical_grids[f"{category}_share"] = (
            historical_grids[f"{category}_count"]
            / historical_grids["sample_count"]
        )

    historical_grids["coverage_category"] = np.select(
        [
            historical_grids["sample_count"] == 1,
            historical_grids["sample_count"].between(2, 4),
            historical_grids["sample_count"] >= 5,
        ],
        [
            "Single sampled location",
            "Limited coverage",
            "Greater coverage",
        ],
        default="No data"
    )

    coordinates = (
        historical_grids["grid_id"]
        .astype(str)
        .str.split("_", expand=True)
    )

    if coordinates.shape[1] != 2:
        raise ValueError("Invalid historical grid ID format.")

    historical_grids["center_lat"] = (
        coordinates[0].astype(int) * GRID_CELL_DEGREES
    ).round(6)

    historical_grids["center_lon"] = (
        coordinates[1].astype(int) * GRID_CELL_DEGREES
    ).round(6)

    half_cell = GRID_CELL_DEGREES / 2

    historical_grids["south"] = (
        historical_grids["center_lat"] - half_cell
    ).round(6)

    historical_grids["north"] = (
        historical_grids["center_lat"] + half_cell
    ).round(6)

    historical_grids["west"] = (
        historical_grids["center_lon"] - half_cell
    ).round(6)

    historical_grids["east"] = (
        historical_grids["center_lon"] + half_cell
    ).round(6)

    if historical_grids.duplicated(
        ["storm_name", "grid_id"]
    ).any():
        raise ValueError("Duplicate historical grid summaries.")

    if (
        historical_grids["sample_count"].sum()
        != EXPECTED_HISTORICAL_ROWS
    ):
        raise ValueError("Historical aggregate count mismatch.")

    classification_columns = [
        "tp_count",
        "tn_count",
        "fp_count",
        "fn_count",
    ]

    if not historical_grids[
        classification_columns
    ].sum(axis=1).eq(
        historical_grids["sample_count"]
    ).all():
        raise ValueError("Historical confusion count mismatch.")

    if not (
        historical_grids["tp_count"]
        + historical_grids["fn_count"]
    ).eq(
        historical_grids["report_positive_count"]
    ).all():
        raise ValueError("Historical positive label count mismatch.")

    if not (
        historical_grids["tp_count"]
        + historical_grids["fp_count"]
    ).eq(
        historical_grids["predicted_positive_count"]
    ).all():
        raise ValueError("Historical predicted count mismatch.")

    share_columns = [
        "tp_share",
        "tn_share",
        "fp_share",
        "fn_share",
    ]

    if not np.allclose(
        historical_grids[share_columns].sum(axis=1),
        1.0
    ):
        raise ValueError("Historical classification share mismatch.")

    score_columns = [
        "mean_score",
        "min_score",
        "max_score",
        "report_positive_share",
        "predicted_positive_share",
        "accuracy",
        *share_columns,
    ]

    if not (
        historical_grids[score_columns].ge(0)
        & historical_grids[score_columns].le(1)
    ).all().all():
        raise ValueError("Invalid historical summary score range.")

    if historical_grids.isna().any().any():
        raise ValueError("Missing historical grid summary values.")

    print("Historical grid-storm summaries:", len(historical_grids))

    print("\nGeographic grid cells by storm:")
    print(
        historical_grids.groupby("storm_name")["grid_id"].nunique()
    )

    print(
        "\nTotal aggregated historical observations:",
        historical_grids["sample_count"].sum()
    )

    print("Historical confusion count validation: PASS")
    print("Historical classification share validation: PASS")
    print("Historical geographic summaries generated successfully.")

    return historical_grids

# ============================================================
# Historical performance metrics
# ============================================================

def calculate_storm_metrics(
    historical: pd.DataFrame,
    historical_grids: pd.DataFrame,
    cutoff: float
) -> pd.DataFrame:
    """Calculate metrics from existing saved storm predictions."""

    print("\nCalculating historical storm performance metrics...")

    results = []

    for storm, year in sorted(
        STORM_YEARS.items(),
        key=lambda item: item[1]
    ):

        storm_data = historical.loc[
            historical["storm_name"] == storm
        ]

        storm_grids = historical_grids.loc[
            historical_grids["storm_name"] == storm
        ]

        y_true = storm_data["y_true"].to_numpy()
        y_pred = storm_data["y_pred"].to_numpy()
        y_prob = storm_data["y_prob"].to_numpy()

        if set(np.unique(y_true)) != {0, 1}:
            raise ValueError(
                f"Both target classes required for {storm}."
            )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        if tp + tn + fp + fn != len(storm_data):
            raise ValueError(
                f"Confusion matrix count mismatch: {storm}"
            )

        for category, count in {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }.items():

            if storm_grids[f"{category}_count"].sum() != count:
                raise ValueError(
                    f"Grid confusion count mismatch: {storm}"
                )

        result = {
            "year": year,
            "storm_name": storm,

            "evaluation_protocol": (
                storm_data["evaluation_protocol"].iloc[0]
            ),

            "observation_count": len(storm_data),
            "grid_count": storm_data["grid_id"].nunique(),

            "frozen_cutoff": cutoff,

            "report_positive_count": int(y_true.sum()),
            "predicted_positive_count": int(y_pred.sum()),

            "accuracy": accuracy_score(y_true, y_pred),

            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),

            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),

            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),

            "roc_auc": roc_auc_score(y_true, y_prob),

            "pr_auc": average_precision_score(
                y_true, y_prob
            ),

            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        results.append(result)

    metrics = pd.DataFrame(results)

    if len(metrics) != 4:
        raise ValueError("Unexpected storm metric count.")

    if metrics["observation_count"].sum() != EXPECTED_HISTORICAL_ROWS:
        raise ValueError("Storm metric observation count mismatch.")

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    if not (
        metrics[metric_columns].ge(0)
        & metrics[metric_columns].le(1)
    ).all().all():
        raise ValueError("Invalid storm performance metrics.")

    if metrics.isna().any().any():
        raise ValueError("Missing storm performance metrics.")

    print("\nStorm performance metrics:")

    print(
        metrics[
            [
                "storm_name",
                "observation_count",
                *metric_columns,
            ]
        ].round(6).to_string(index=False)
    )

    print("\nStorm confusion matrices:")

    print(
        metrics[
            ["storm_name", "tp", "tn", "fp", "fn"]
        ].to_string(index=False)
    )

    print("\nHistorical performance metrics calculated successfully.")

    return metrics
# ============================================================
# Global SHAP resources
# ============================================================

def prepare_global_shap() -> pd.DataFrame:
    """Load the original research SHAP importance results."""

    print("\nPreparing global SHAP resources...")

    importance = pd.read_csv(GLOBAL_SHAP_PATH)

    required_columns = {
        "feature",
        "mean_abs_shap",
        "shap_rank",
        "lightgbm_split_importance",
        "lightgbm_split_rank",
    }

    if not required_columns.issubset(importance.columns):
        raise ValueError("Missing global SHAP columns.")

    if len(importance) != 8:
        raise ValueError("Unexpected global SHAP feature count.")

    if importance.isna().any().any():
        raise ValueError("Missing global SHAP values.")

    if importance["feature"].duplicated().any():
        raise ValueError("Duplicate global SHAP features.")

    if not np.isfinite(importance["mean_abs_shap"]).all():
        raise ValueError("Non-finite global SHAP importance.")

    if not importance["mean_abs_shap"].ge(0).all():
        raise ValueError("Negative mean absolute SHAP values.")

    if set(importance["shap_rank"]) != set(range(1, 9)):
        raise ValueError("Invalid global SHAP rankings.")

    importance = importance.sort_values(
        "shap_rank"
    ).reset_index(drop=True)

    print(importance.to_string(index=False))
    print("Global SHAP resources prepared successfully.")

    return importance

# ============================================================
# Model metadata
# ============================================================

def generate_model_metadata(
    configuration: dict,
    pipeline: Pipeline,
    locations: pd.DataFrame,
    grid_summary: pd.DataFrame,
    historical: pd.DataFrame,
    historical_grids: pd.DataFrame,
    storm_metrics: pd.DataFrame,
    global_shap: pd.DataFrame
) -> dict:
    """Generate application methodology and dataset metadata."""

    print("\nGenerating model metadata...")

    shap_columns = [
        f"shap_{feature}"
        for feature in MODEL_FEATURES
    ]

    grouped_importance = (
        locations[shap_columns]
        .abs()
        .mean()
        .sort_values(ascending=False)
    )

    metadata = {
        "application": {
            "name": "FloodSense",
            "mascot": "Scout",
            "purpose": (
                "Interactive exploration of historical "
                "flood susceptibility in the Houston study area."
            ),
            "research_only": True,
            "real_time_forecast": False,
        },

        "model": {
            "name": MODEL_NAME,
            "algorithm": "LightGBM",
            "role": "final",
            "random_seed": EXPECTED_RANDOM_SEED,
            "frozen_cutoff": float(configuration["cutoff"]),
            "features": list(MODEL_FEATURES),
            "excluded_features": list(DROP_FEATURES),
            "target": TARGET_COLUMN,
            "parameters": pipeline.named_steps[
                "model"
            ].get_params(),
            "training_observations": EXPECTED_TRAINING_ROWS,
            "transformed_feature_count": 8,
            "training_scope": (
                "Complete development dataset, 2015-2017"
            ),
        },

        "threshold_selection": {
            "area_recall": float(
                configuration["area_recall_at_cutoff"]
            ),
            "area_precision": float(
                configuration["area_precision_at_cutoff"]
            ),
            "storm_recall": float(
                configuration["storm_recall_at_cutoff"]
            ),
            "storm_precision": float(
                configuration["storm_precision_at_cutoff"]
            ),
            "interpretation": (
                "Frozen observation-level classification cutoff. "
                "Not independently validated as a grid-level cutoff."
            ),
        },

        "dataset_counts": {
            "development_observations": EXPECTED_TRAINING_ROWS,
            "unique_sampled_locations": len(locations),
            "geographic_grid_cells": len(grid_summary),
            "historical_observations": len(historical),
            "historical_grid_storm_summaries": len(
                historical_grids
            ),
            "independent_test_observations": EXPECTED_TEST_ROWS,
            "historical_storms": len(storm_metrics),
        },

        "geography": {
            "coordinate_reference_system": "EPSG:4326",
            "grid_cell_degrees": GRID_CELL_DEGREES,
            "approximate_cell_height_km": 1.11,
            "approximate_cell_width_km": 0.96,
            "grid_id_method": (
                "Round latitude and longitude divided by 0.01 "
                "using pandas rounding, then join with underscore."
            ),
            "extent": {
                "min_lat": float(locations["lat"].min()),
                "max_lat": float(locations["lat"].max()),
                "min_lon": float(locations["lon"].min()),
                "max_lon": float(locations["lon"].max()),
            },
            "municipal_boundary_clipped": False,
            "unsampled_cells": "No data",
        },

        "coverage": {
            "single_location": 1,
            "limited_coverage": [2, 4],
            "greater_coverage_minimum": 5,
            "counts": {
                str(key): int(value)
                for key, value in grid_summary[
                    "coverage_category"
                ].value_counts().items()
            },
            "interpretation": (
                "Sampling coverage, not prediction confidence."
            ),
        },

        "feature_descriptions": {
            "elevation_m": "Elevation in meters.",
            "slope_degrees": "Terrain slope in degrees.",
            "distance_to_bayou_m": (
                "Distance to mapped bayou or stream in meters."
            ),
            "paved_pct": (
                "Research proxy representing the share of a "
                "tract near major highways. Not a general "
                "impervious-surface measurement."
            ),
            "land_cover_type": "Developed land cover category.",
        },

        "susceptibility": {
            "score_range": [0, 1],
            "location_score": (
                "Positive-class model probability output."
            ),
            "grid_score": (
                "Mean score across unique sampled locations "
                "within a geographic cell."
            ),
            "interpretation": (
                "Relative model-based historical susceptibility, "
                "not a calibrated future flood probability."
            ),
            "locations_at_or_above_cutoff": int(
                locations[
                    "predicted_at_frozen_cutoff"
                ].sum()
            ),
        },

        "shap": {
            "units": "Raw model log-odds",
            "local_explanation_count": len(locations),
            "encoded_feature_count": 8,
            "grouped_feature_count": 5,
            "land_cover_grouping": (
                "Sum signed SHAP contributions from all "
                "one-hot encoded land cover features."
            ),
            "grid_aggregation": (
                "Mean signed SHAP contribution across "
                "unique sampled locations."
            ),
            "positive_contribution": (
                "Raises the raw model output relative "
                "to the SHAP baseline."
            ),
            "negative_contribution": (
                "Lowers the raw model output relative "
                "to the SHAP baseline."
            ),
            "grouped_mean_absolute_importance": {
                key.removeprefix("shap_"): float(value)
                for key, value in grouped_importance.items()
            },
            "original_global_importance_source": (
                "reports/shap_feature_importance.csv"
            ),
            "original_global_sample_size": 5000,
            "original_global_encoded_features": len(global_shap),
            "figures": sorted(
                str(path.relative_to(PROJECT_DIR)).replace(
                    "\\", "/"
                )
                for path in FIGURES_DIR.glob("shap_*.png")
            ),
        },

        "historical": {
            "storm_years": STORM_YEARS,
            "storm_observation_counts": EXPECTED_STORM_COUNTS,
            "development_protocol": "leave_one_storm_out",
            "independent_test_protocol": "independent_test",
            "development_prediction_source": (
                "reports/oof_lightgbm_no_rain_leave_one_storm_out.csv"
            ),
            "independent_prediction_source": (
                "reports/final_test_predictions_lightgbm_no_rain.csv"
            ),
            "positive_label": (
                "Observation associated with a recorded flood report."
            ),
            "negative_label": (
                "Sampled observation without an associated "
                "flood report. Not verified dry."
            ),
            "classification_cutoff": float(
                configuration["cutoff"]
            ),
            "pr_auc_method": "average_precision_score",
            "event_specific_shap_available": False,
        },

        "limitations": [
            "Research estimates, not official flood warnings.",
            "Not a real-time or future storm forecast.",
            "Rainfall is excluded from the selected model.",
            "No property-level flood risk assessment.",
            "Only sampled geographic cells have predictions.",
            "Sampling coverage is not prediction confidence.",
            "Negative labels do not establish absence of flooding.",
            "SHAP contributions are not causal explanations.",
            "Mean SHAP explains mean raw log-odds, not mean probability.",
            "General susceptibility predictions are in-sample outputs.",
            "Historical maps do not represent inundation extent.",
            "Historical storm samples and coverage differ.",
            "Development validation informed model selection.",
            "Study area is not clipped to municipal boundaries.",
        ],

        "research_protection": {
            "protected_source_files": EXPECTED_PROTECTED_FILES,
            "independent_test_evaluation_rerun": False,
            "independent_test_used_for_training": False,
            "original_model_configuration_modified": False,
        },

        "preparation_environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "lightgbm": version("lightgbm"),
            "shap": version("shap"),
            "pyarrow": version("pyarrow"),
        },

        "data_schema": {
            "location_predictions.parquet": (
                locations.columns.tolist()
            ),
            "grid_summary.parquet": (
                grid_summary.columns.tolist()
            ),
            "historical_predictions.parquet": (
                historical.columns.tolist()
            ),
            "historical_grid_summary.parquet": (
                historical_grids.columns.tolist()
            ),
            "storm_metrics.csv": (
                storm_metrics.columns.tolist()
            ),
            "global_shap_importance.csv": (
                global_shap.columns.tolist()
            ),
        },
    }

    # Confirm that all metadata can be serialized.
    json.dumps(metadata, allow_nan=False)

    print("Model metadata generated successfully.")

    return metadata

# ============================================================
# Application data exports
# ============================================================

def export_application_data(
    locations: pd.DataFrame,
    grid_summary: pd.DataFrame,
    grid_boundaries: dict,
    historical: pd.DataFrame,
    historical_grids: pd.DataFrame,
    storm_metrics: pd.DataFrame,
    global_shap: pd.DataFrame,
    metadata: dict
) -> None:
    """Export validated application-ready datasets."""

    print("\nExporting application datasets...")

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_exports = {
        LOCATION_PREDICTIONS_PATH: locations,
        GRID_SUMMARY_PATH: grid_summary,
        HISTORICAL_PREDICTIONS_PATH: historical,
        HISTORICAL_GRID_SUMMARY_PATH: historical_grids,
    }

    for path, dataframe in parquet_exports.items():

        dataframe.to_parquet(
            path,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        print(
            f"Saved: {path.name} "
            f"({len(dataframe):,} rows)"
        )

    csv_exports = {
        STORM_METRICS_PATH: storm_metrics,
        APP_GLOBAL_SHAP_PATH: global_shap,
    }

    for path, dataframe in csv_exports.items():

        dataframe.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path.name} "
            f"({len(dataframe):,} rows)"
        )

    with GRID_BOUNDARIES_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            grid_boundaries,
            file,
            allow_nan=False,
            separators=(",", ":")
        )

    print("Saved:", GRID_BOUNDARIES_PATH.name)

    with MODEL_METADATA_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            allow_nan=False
        )

    print("Saved:", MODEL_METADATA_PATH.name)

    print("Application data exported successfully.")

# ============================================================
# Export validation
# ============================================================

def validate_application_exports(
    locations: pd.DataFrame,
    grid_summary: pd.DataFrame,
    grid_boundaries: dict,
    historical: pd.DataFrame,
    historical_grids: pd.DataFrame,
    storm_metrics: pd.DataFrame,
    global_shap: pd.DataFrame,
    metadata: dict
) -> None:
    """Verify exported application files against validated data."""

    print("\nValidating exported application datasets...")

    expected_files = [
        LOCATION_PREDICTIONS_PATH,
        GRID_SUMMARY_PATH,
        GRID_BOUNDARIES_PATH,
        HISTORICAL_PREDICTIONS_PATH,
        HISTORICAL_GRID_SUMMARY_PATH,
        STORM_METRICS_PATH,
        APP_GLOBAL_SHAP_PATH,
        MODEL_METADATA_PATH,
    ]

    total_bytes = 0

    for path in expected_files:

        if not path.is_file():
            raise FileNotFoundError(path)

        size = path.stat().st_size

        if size == 0:
            raise ValueError(f"Empty output file: {path.name}")

        total_bytes += size

        print(
            f"PASS: {path.name} "
            f"({size / 1024**2:.3f} MB)"
        )

    parquet_exports = {
        LOCATION_PREDICTIONS_PATH: locations,
        GRID_SUMMARY_PATH: grid_summary,
        HISTORICAL_PREDICTIONS_PATH: historical,
        HISTORICAL_GRID_SUMMARY_PATH: historical_grids,
    }

    for path, expected in parquet_exports.items():

        actual = pd.read_parquet(path)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False
        )

        if actual.isna().any().any():
            raise ValueError(
                f"Missing exported values: {path.name}"
            )

        print("Parquet round-trip validation: PASS", path.name)

    csv_exports = {
        STORM_METRICS_PATH: storm_metrics,
        APP_GLOBAL_SHAP_PATH: global_shap,
    }

    for path, expected in csv_exports.items():

        actual = pd.read_csv(path)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-10
        )

        print("CSV round-trip validation: PASS", path.name)

    with GRID_BOUNDARIES_PATH.open(
        encoding="utf-8"
    ) as file:

        saved_boundaries = json.load(file)

    if saved_boundaries != grid_boundaries:
        raise ValueError("GeoJSON round-trip mismatch.")

    with MODEL_METADATA_PATH.open(
        encoding="utf-8"
    ) as file:

        saved_metadata = json.load(file)

    if saved_metadata != metadata:
        raise ValueError("Metadata round-trip mismatch.")

    print("GeoJSON round-trip validation: PASS")
    print("Metadata round-trip validation: PASS")

    # Compare Imelda metrics with the previously recorded
    # independent test results at their known precision.
    imelda = storm_metrics.loc[
        storm_metrics["storm_name"] == "Tropical Storm Imelda"
    ].iloc[0]

    expected_imelda = {
        "accuracy": 0.8199,
        "precision": 0.7976,
        "recall": 0.8655,
        "f1": 0.8302,
        "roc_auc": 0.8896,
        "pr_auc": 0.8603,
    }

    for metric, expected in expected_imelda.items():

        actual = float(imelda[metric])

        if round(actual, 4) != expected:
            raise ValueError(
                f"Independent test metric mismatch: {metric}"
            )

        print(
            f"PASS: Imelda {metric} = {actual:.6f}"
        )

    # Verify the established sampling coverage distribution.
    expected_coverage = {
        "Single sampled location": 224,
        "Limited coverage": 720,
        "Greater coverage": 3548,
    }

    actual_coverage = (
        grid_summary["coverage_category"]
        .value_counts()
        .to_dict()
    )

    if actual_coverage != expected_coverage:
        raise ValueError("Unexpected geographic coverage.")

    print("Geographic coverage validation: PASS")

    print("\nExport validation summary:")
    print("Expected application files:", len(expected_files))
    print("Successfully validated:", len(expected_files))
    print(
        "Total application data size:",
        f"{total_bytes / 1024**2:.2f} MB"
    )

    free_bytes = shutil.disk_usage(PROJECT_DIR).free

    print(
        "Remaining disk space:",
        f"{free_bytes / 1024**3:.2f} GB"
    )

    print("All application export validations passed.")

# ============================================================
# Application data documentation
# ============================================================

def write_application_documentation(
    metadata: dict
) -> None:
    """Document the application preparation methodology."""

    print("\nWriting application data documentation...")

    documentation_path = (
        DOCS_DIR / "APP_DATA_PREPARATION.md"
    )

    lines = [
        "# FloodSense Application Data Preparation",
        "",
        "## Purpose",
        "",
        "Precomputed datasets for the FloodSense research application.",
        "",
        "The application uses offline model predictions and SHAP",
        "explanations. No model training is required at runtime.",
        "",
        "## Reproduction",
        "",
        "From the repository root, using Python 3.11:",
        "",
        "```powershell",
        "python -m pip install -r scripts/requirements-app-prep.txt",
        "python -m scripts.prepare_app_data",
        "```",
        "",
        "The original research requirements remain unchanged.",
        "The application preparation environment uses SHAP 0.51.0",
        "for Python 3.11 compatibility.",
        "",
        "## Model methodology",
        "",
        "- Model: LightGBM without rainfall.",
        "- Training observations: 66,514.",
        "- Frozen classification cutoff: 0.50.",
        "- Predictions generated for 62,138 unique locations.",
        "- Training uses all development observations.",
        "- Deduplication applies only to geographic inference.",
        "- Independent test observations are not used for fitting.",
        "",
        "## Application datasets",
        "",
        "| File | Contents |",
        "|---|---|",
        "| location_predictions.parquet | 62,138 locations |",
        "| grid_summary.parquet | 4,492 geographic cells |",
        "| grid_boundaries.geojson | 4,492 grid polygons |",
        "| historical_predictions.parquet | 85,678 observations |",
        "| historical_grid_summary.parquet | 15,873 grid-storm summaries |",
        "| storm_metrics.csv | Four storm metric summaries |",
        "| global_shap_importance.csv | Eight encoded SHAP features |",
        "| model_metadata.json | Configuration and methodology |",
        "",
        "## Dataset schemas",
        "",
    ]

    for filename, columns in metadata["data_schema"].items():

        lines.extend([
            f"### {filename}",
            "",
        ])

        for column in columns:
            lines.append(f"- `{column}`")

        lines.append("")

    lines.extend([
        "## Important field interpretations",
        "",
        "- `susceptibility_score`: positive-class model output.",
        "- `mean_score`: average score across sampled locations.",
        "- `sample_count`: number of sampled locations or",
        "  observations, depending on the dataset.",
        "- `share_above_cutoff`: fraction at or above 0.50.",
        "- `development_report_positive_count`: cleaned",
        "  report-associated observations, not raw report volume.",
        "- `shap_*`: signed contributions in raw log-odds.",
        "- `dominant_positive_factor`: strongest positive mean SHAP.",
        "- `report_positive_share`: fraction of sampled observations",
        "  associated with a recorded report.",
        "- `y_true`: original historical dataset label.",
        "- `y_prob`: saved historical prediction probability.",
        "- `y_pred`: classification using the frozen cutoff.",
        "- `classification`: TP, TN, FP, or FN relative to labels.",
        "",
        "## Geographic methodology",
        "",
        "The original 0.01-degree research grid is reconstructed",
        "from grid identifiers. Grid dimensions are approximately",
        "1.11 km north-south and 0.96 km east-west in Houston.",
        "",
        "Unsampled cells must be displayed as no data.",
        "The grid is not clipped to Houston municipal boundaries.",
        "",
        "## SHAP methodology",
        "",
        "Local explanations use the full-development fitted model.",
        "One-hot land cover contributions are summed per location.",
        "Grid explanations average signed location contributions.",
        "These explain mean raw log-odds, not mean probability.",
        "",
        "Original global SHAP plots remain in reports/figures.",
        "The original encoded-feature importance and newly grouped",
        "importance must be identified as separate analyses.",
        "",
        "## Historical evaluation",
        "",
        "Memorial Day, Tax Day and Harvey use saved",
        "leave-one-storm-out development predictions.",
        "Imelda uses saved independent test predictions.",
        "",
        "No historical models were retrained for the application.",
        "No independent test evaluation was rerun.",
        "",
        "## Research limitations",
        "",
    ])

    for limitation in metadata["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Research protection",
        "",
        "The preparation script verifies 28 protected source files",
        "against docs/app_source_hashes.csv before and after processing.",
        "The independent test lock remains unchanged.",
        "",
        "## Application development",
        "",
        "Read model_metadata.json for the complete configuration,",
        "data schema, feature descriptions and limitations.",
        "",
        "Application runtime should load precomputed data rather",
        "than importing LightGBM, SHAP or PyTorch.",
        "",
    ])

    documentation_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("Saved:", documentation_path.name)

if __name__ == "__main__":
    main()