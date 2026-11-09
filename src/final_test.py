"""Run the single frozen final test and save its predictions and metrics."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.cnn import TorchCNNClassifier
from src.config import PROJECT_DIR, SEED
from src.evaluate import compute_metrics
from src.features import build_preprocessor, get_xy
from src.models import get_lightgbm, get_logistic_regression
from src.mlp import TorchMLPClassifier

REPORTS_DIR = PROJECT_DIR / "reports"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TILE_DIR = PROCESSED_DIR / "tiles"
SETUP_PATH = REPORTS_DIR / "final_setup.csv"
LOCK_PATH = REPORTS_DIR / "final_test.lock"
TRAINING_PATH = PROCESSED_DIR / "training_clean.csv"
TEST_PATH = PROCESSED_DIR / "test_final_clean.csv"
TRAIN_TILES_PATH = TILE_DIR / "train_tiles.npy"
TRAIN_TILE_INDEX_PATH = TILE_DIR / "train_tiles_row_index.npy"
TEST_TILES_PATH = TILE_DIR / "test_tiles.npy"
TEST_TILE_INDEX_PATH = TILE_DIR / "test_tiles_row_index.npy"
RESULTS_PATH = REPORTS_DIR / "final_test_results.csv"

FINAL_MODELS = (
    "logistic_regression_no_rain",
    "lightgbm_no_rain",
    "mlp_no_rain",
    "cnn_hybrid",
)
EXPERIMENT_MODELS = ("cnn_tile_only",)
DROP_FEATURES = ("rainfall_inches",)


def _acquire_lock() -> None:
    """Create the one-time lock atomically before accessing test labels."""
    try:
        with LOCK_PATH.open("x", encoding="utf-8") as lock_file:
            json.dump(
                {
                    "status": "started",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": SEED,
                    "note": "One-time final test; reruns are prohibited.",
                },
                lock_file,
                indent=2,
            )
            lock_file.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Final test is locked at {LOCK_PATH}; refusing to run again."
        ) from exc


def _finish_lock(model_names: list[str]) -> None:
    """Record successful completion while preserving the one-time lock."""
    LOCK_PATH.write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "models": model_names,
                "results_file": str(RESULTS_PATH.relative_to(PROJECT_DIR)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_frozen_setup() -> pd.DataFrame:
    setup = pd.read_csv(SETUP_PATH)
    required = {"model", "features_used", "cutoff", "role"}
    missing = required - set(setup.columns)
    if missing:
        raise ValueError(f"{SETUP_PATH} is missing columns: {sorted(missing)}")
    expected_order = [*FINAL_MODELS, *EXPERIMENT_MODELS]
    if setup["model"].tolist() != expected_order:
        raise ValueError(
            "Frozen setup model order or membership changed; "
            f"expected {expected_order}, got {setup['model'].tolist()}."
        )
    if not setup["features_used"].str.contains("no rainfall_inches", regex=False).all():
        raise ValueError("Every frozen CNN/tabular model must exclude rainfall_inches.")
    if setup.loc[setup["model"].isin(FINAL_MODELS), "role"].ne("final").any():
        raise ValueError("Frozen final model roles do not match the setup.")
    if setup.loc[setup["model"].isin(EXPERIMENT_MODELS), "role"].ne(
        "experiment only"
    ).any():
        raise ValueError("CNN tile-only must remain experiment-only.")
    if setup["cutoff"].isna().any() or not setup["cutoff"].between(0, 1).all():
        raise ValueError("Frozen cutoffs must be finite probabilities.")
    return setup


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Fitted estimator does not expose class 1.")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def _write_prediction_file(
    model_name: str, y_true: pd.Series, y_prob: np.ndarray
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true.to_numpy(),
            "y_prob": y_prob,
        }
    )
    path = REPORTS_DIR / f"final_test_predictions_{model_name}.csv"
    predictions.to_csv(path, index=False)
    return predictions


def _metric_rows(
    model_name: str,
    role: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    frozen_cutoff: float | None,
) -> list[dict[str, Any]]:
    rows = []
    thresholds = (
        [("frozen", frozen_cutoff), ("default_0.5", 0.5)]
        if frozen_cutoff is not None
        else [("reference_0.5", 0.5)]
    )
    for cutoff_type, threshold in thresholds:
        if threshold is None:
            raise RuntimeError("A frozen cutoff cannot be null.")
        rows.append(
            {
                "model": model_name,
                "role": role,
                "cutoff_type": cutoff_type,
                "cutoff": float(threshold),
                **compute_metrics(y_true, y_prob, float(threshold)),
            }
        )
    return rows


def _table_features(
    training: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train = get_xy(training, drop_features=DROP_FEATURES)
    X_test, y_test = get_xy(test, drop_features=DROP_FEATURES)
    return X_train, y_train, X_test, y_test


def _validate_tiles(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    train_tiles = np.load(TRAIN_TILES_PATH, mmap_mode="r")
    test_tiles = np.load(TEST_TILES_PATH, mmap_mode="r")
    train_indices = np.load(TRAIN_TILE_INDEX_PATH)
    test_indices = np.load(TEST_TILE_INDEX_PATH)
    expected_train = np.arange(len(train), dtype=np.int64)
    expected_test = np.arange(len(test), dtype=np.int64)
    if not np.array_equal(train_indices, expected_train):
        raise ValueError("Training tile row indices do not match training row order.")
    if not np.array_equal(test_indices, expected_test):
        raise ValueError("Test tile row indices do not match cleaned test row order.")
    if train_tiles.shape != (len(train), 32, 32):
        raise ValueError(f"Unexpected training tile shape: {train_tiles.shape}.")
    if test_tiles.shape != (len(test), 32, 32):
        raise ValueError(f"Unexpected test tile shape: {test_tiles.shape}.")
    if train_tiles.dtype != np.float16 or test_tiles.dtype != np.float16:
        raise ValueError("Training and test tiles must use the saved float16 format.")
    return train_tiles, test_tiles


def run_final_test() -> pd.DataFrame:
    """Fit frozen models once, predict once, then lock the final evaluation."""
    _acquire_lock()
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    setup = _read_frozen_setup()
    training = pd.read_csv(TRAINING_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train, X_test, y_test = _table_features(training, test)
    train_tiles, test_tiles = _validate_tiles(training, test)
    if len(setup) != len(FINAL_MODELS) + len(EXPERIMENT_MODELS):
        raise ValueError("Unexpected frozen model count.")

    metric_rows: list[dict[str, Any]] = []
    model_names = [*FINAL_MODELS, *EXPERIMENT_MODELS, "dummy"]
    factories = {
        "logistic_regression_no_rain": get_logistic_regression,
        "lightgbm_no_rain": get_lightgbm,
        "mlp_no_rain": TorchMLPClassifier,
    }

    for record in setup.itertuples(index=False):
        name = str(record.model)
        role = str(record.role)
        cutoff = float(record.cutoff)
        fit_started = time.perf_counter()

        if name in factories:
            estimator = Pipeline(
                [
                    ("preprocessor", build_preprocessor(drop_features=DROP_FEATURES)),
                    ("model", factories[name]()),
                ]
            )
            estimator.fit(X_train, y_train)
            y_prob = _positive_probability(estimator, X_test)
        elif name in {"cnn_hybrid", "cnn_tile_only"}:
            table_train = None
            table_test = None
            if name == "cnn_hybrid":
                preprocessor = build_preprocessor(drop_features=DROP_FEATURES)
                table_train = preprocessor.fit_transform(X_train)
                table_test = preprocessor.transform(X_test)
                if hasattr(table_train, "toarray"):
                    table_train = table_train.toarray()
                    table_test = table_test.toarray()
                table_train = np.asarray(table_train, dtype=np.float32)
                table_test = np.asarray(table_test, dtype=np.float32)
            cnn = TorchCNNClassifier(name, seed=SEED)
            cnn.fit(
                train_tiles,
                y_train,
                table_features=table_train,
                epochs=20,
                batch_size=256,
                learning_rate=0.001,
            )
            y_prob = cnn.predict_proba(test_tiles, table_features=table_test)[:, 1]
        else:
            raise ValueError(f"Unsupported frozen model: {name}")

        _write_prediction_file(name, y_test, y_prob)
        metric_rows.extend(_metric_rows(name, role, y_test, y_prob, cutoff))
        print(
            f"{name}: fitted and predicted once in "
            f"{time.perf_counter() - fit_started:.1f} seconds.",
            flush=True,
        )

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1), dtype=np.float32), y_train)
    dummy_prob = _positive_probability(
        dummy, np.zeros((len(y_test), 1), dtype=np.float32)
    )
    _write_prediction_file("dummy", y_test, dummy_prob)
    metric_rows.extend(_metric_rows("dummy", "experiment/reference", y_test, dummy_prob, None))

    results = pd.DataFrame(metric_rows)
    results.to_csv(RESULTS_PATH, index=False)
    _finish_lock(model_names)
    print(
        f"Final test completed in {time.perf_counter() - started:.1f} seconds. "
        f"Results: {RESULTS_PATH}",
        flush=True,
    )
    return results


if __name__ == "__main__":
    run_final_test()
