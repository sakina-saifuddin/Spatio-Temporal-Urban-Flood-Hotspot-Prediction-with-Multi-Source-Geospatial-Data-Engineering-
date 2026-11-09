"""SHAP explanations for the frozen no-rainfall LightGBM model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    CLEAN_TRAINING_PATH,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    SEED,
)
from src.features import build_preprocessor, get_xy
from src.models import get_lightgbm

MODEL_NAME = "lightgbm_no_rain"
DROP_FEATURES = ("rainfall_inches",)
SHAP_SAMPLE_SIZE = 5_000
FIGURES_DIR = REPORTS_DIR / "figures"
IMPORTANCE_PATH = REPORTS_DIR / "shap_feature_importance.csv"
EXAMPLES_PATH = REPORTS_DIR / "shap_examples.csv"


@dataclass(frozen=True)
class ShapRunResult:
    """Files and measured comparisons produced by the SHAP run."""

    importance: pd.DataFrame
    examples: pd.DataFrame
    rank_correlation: float
    top_three_overlap: int
    cutoff: float
    sample_size: int
    row_count: int


def _read_frozen_cutoff(setup_path: Path) -> float:
    """Read the frozen LightGBM cutoff without consulting final-test outputs."""
    with setup_path.open(encoding="utf-8", newline="") as setup_file:
        rows = list(csv.DictReader(setup_file))

    matches = [row for row in rows if row.get("model") == MODEL_NAME]
    if len(matches) != 1 or matches[0].get("role") != "final":
        raise ValueError(f"{MODEL_NAME} must appear once as a final model in {setup_path}.")

    row = matches[0]
    features = row.get("features_used", "")
    feature_text, separator, rainfall_note = features.partition(";")
    saved_features = tuple(
        feature.strip() for feature in feature_text.split(",") if feature.strip()
    )
    expected_features = tuple(
        name for name in NUMERIC_FEATURES if name not in DROP_FEATURES
    ) + CATEGORICAL_FEATURES
    if (
        not separator
        or rainfall_note.strip() != "no rainfall_inches"
        or "rainfall_inches" in saved_features
    ):
        raise ValueError("The frozen LightGBM setup must explicitly exclude rainfall_inches.")
    if saved_features != expected_features:
        raise ValueError("The frozen LightGBM feature list does not match configured features.")
    return float(row["cutoff"])


def _readable_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Name numeric and one-hot columns in a form suitable for plots."""
    categorical = preprocessor.named_transformers_["categorical"]
    land_cover_values = categorical.categories_[0]
    names = [name for name in NUMERIC_FEATURES if name not in DROP_FEATURES]
    names.extend(f"land cover type: {value}" for value in land_cover_values)
    if len(names) != len(preprocessor.get_feature_names_out()):
        raise ValueError("Readable feature names do not match the preprocessor output.")
    return names


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_global_plots(
    explanation: shap.Explanation,
    importance: pd.DataFrame,
    sampled_features: pd.DataFrame,
    names: list[str],
) -> list[Path]:
    """Save mean-absolute-SHAP, beeswarm, and numeric dependence plots."""
    paths: list[Path] = []

    ordered = importance.sort_values("mean_abs_shap", ascending=True)
    figure_height = max(5.5, 0.42 * len(ordered))
    figure, axis = plt.subplots(figsize=(11, figure_height))
    axis.barh(ordered["feature"], ordered["mean_abs_shap"], color="#3478a8")
    axis.set_xlabel("Mean absolute SHAP value (model log-odds)")
    axis.set_title("LightGBM feature importance by mean absolute SHAP")
    figure.tight_layout()
    path = FIGURES_DIR / "shap_lightgbm_mean_abs.png"
    _save_figure(figure, path)
    paths.append(path)

    shap.plots.beeswarm(
        explanation,
        max_display=len(names),
        show=False,
        plot_size=(12, figure_height),
    )
    figure = plt.gcf()
    figure.suptitle("LightGBM SHAP summary (5,000 training rows)", y=1.01)
    path = FIGURES_DIR / "shap_lightgbm_beeswarm.png"
    _save_figure(figure, path)
    paths.append(path)

    for feature in ("elevation_m", "distance_to_bayou_m", "slope_degrees"):
        column = names.index(feature)
        values = sampled_features[feature].to_numpy()
        shap_values = np.asarray(explanation.values)[:, column]
        figure, axis = plt.subplots(figsize=(8, 5.5))
        axis.scatter(values, shap_values, s=12, alpha=0.28, color="#3478a8")
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.65)
        axis.set_xlabel(feature)
        axis.set_ylabel("SHAP value (model log-odds)")
        axis.set_title(f"SHAP dependence: {feature}")
        figure.tight_layout()
        path = FIGURES_DIR / f"shap_dependence_{feature}.png"
        _save_figure(figure, path)
        paths.append(path)

    return paths


def _save_waterfalls(
    explanation: shap.Explanation,
    sampled_features: pd.DataFrame,
    sampled_labels: np.ndarray,
    sampled_probabilities: np.ndarray,
    sampled_row_indices: np.ndarray,
    cutoff: float,
    names: list[str],
) -> pd.DataFrame:
    """Choose and plot three deterministic example training rows."""
    predictions = (sampled_probabilities >= cutoff).astype(int)
    correct_flooded = np.flatnonzero((sampled_labels == 1) & (predictions == 1))
    correct_not_flooded = np.flatnonzero((sampled_labels == 0) & (predictions == 0))
    incorrect = np.flatnonzero(sampled_labels != predictions)
    if not len(correct_flooded) or not len(correct_not_flooded) or not len(incorrect):
        raise ValueError("The fixed sample lacks a row for one of the three requested examples.")

    wrong_confidence = np.where(
        sampled_labels[incorrect] == 1,
        sampled_probabilities[incorrect],
        1 - sampled_probabilities[incorrect],
    )
    chosen = {
        "flooded_correct": int(
            correct_flooded[np.argmax(sampled_probabilities[correct_flooded])]
        ),
        "not_flooded_correct": int(
            correct_not_flooded[np.argmin(sampled_probabilities[correct_not_flooded])]
        ),
        "wrong_prediction": int(incorrect[np.argmax(wrong_confidence)]),
    }

    rows: list[dict[str, object]] = []
    for name, sample_position in chosen.items():
        truth = int(sampled_labels[sample_position])
        predicted = int(predictions[sample_position])
        row_index = int(sampled_row_indices[sample_position])
        probability = float(sampled_probabilities[sample_position])
        rows.append(
            {
                "example": name,
                "row_index": row_index,
                "latitude": float(sampled_features.iloc[sample_position]["lat"]),
                "longitude": float(sampled_features.iloc[sample_position]["lon"]),
                "y_true": truth,
                "y_pred_at_frozen_cutoff": predicted,
                "y_prob": probability,
            }
        )

        base_values = np.asarray(explanation.base_values)
        base_value = base_values[sample_position]
        if np.ndim(base_value):
            base_value = np.asarray(base_value).reshape(-1)[-1]
        waterfall = shap.Explanation(
            values=np.asarray(explanation.values)[sample_position],
            base_values=float(base_value),
            data=np.asarray(explanation.data)[sample_position],
            feature_names=names,
        )
        shap.plots.waterfall(waterfall, max_display=len(names), show=False)
        figure = plt.gcf()
        figure.suptitle(
            f"{name.replace('_', ' ').title()} — true={truth}, predicted={predicted}, "
            f"p={probability:.3f}",
            y=1.01,
        )
        path = FIGURES_DIR / f"shap_waterfall_{name}.png"
        _save_figure(figure, path)

    return pd.DataFrame(rows)


def run_shap_analysis() -> ShapRunResult:
    """Fit the frozen LightGBM once and create training-only SHAP outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = _read_frozen_cutoff(REPORTS_DIR / "final_setup.csv")
    training = pd.read_csv(CLEAN_TRAINING_PATH)
    X, y = get_xy(training, drop_features=DROP_FEATURES)
    if len(training) < SHAP_SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {SHAP_SAMPLE_SIZE} training rows; found {len(training)}."
        )

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(drop_features=DROP_FEATURES)),
            ("model", get_lightgbm()),
        ]
    )
    print(f"Fitting {MODEL_NAME} once on {len(training):,} training rows.")
    pipeline.fit(X, y)

    sampled = training.sample(n=SHAP_SAMPLE_SIZE, random_state=SEED)
    sampled_row_indices = sampled.index.to_numpy(dtype=int)
    sampled_X = X.loc[sampled.index]
    sampled_y = y.loc[sampled.index].to_numpy(dtype=int)
    transformed = pipeline.named_steps["preprocessor"].transform(sampled_X)
    names = _readable_feature_names(pipeline.named_steps["preprocessor"])
    transformed_array = np.asarray(transformed)
    if transformed_array.ndim != 2 or transformed_array.shape[1] != len(names):
        raise ValueError("Transformed sample has an unexpected shape.")

    print(f"Computing TreeExplainer SHAP values for {SHAP_SAMPLE_SIZE:,} sampled rows.")
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    raw_explanation = explainer(transformed_array)
    explanation = shap.Explanation(
        values=raw_explanation.values,
        base_values=raw_explanation.base_values,
        data=transformed_array,
        feature_names=names,
    )
    if explanation.values.shape != transformed_array.shape:
        raise ValueError(
            f"Unexpected SHAP value shape {explanation.values.shape}; "
            f"expected {transformed_array.shape}."
        )

    mean_abs_shap = np.abs(np.asarray(explanation.values)).mean(axis=0)
    split_importance = pipeline.named_steps["model"].feature_importances_
    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": mean_abs_shap,
            "shap_rank": pd.Series(mean_abs_shap).rank(
                ascending=False, method="min"
            ).astype(int),
            "lightgbm_split_importance": split_importance,
            "lightgbm_split_rank": pd.Series(split_importance)
            .rank(ascending=False, method="min")
            .astype(int),
        }
    ).sort_values(["shap_rank", "feature"])
    rank_correlation = float(
        spearmanr(
            importance["mean_abs_shap"],
            importance["lightgbm_split_importance"],
        ).statistic
    )
    shap_top_three = set(
        importance.nsmallest(3, "shap_rank")["feature"].tolist()
    )
    split_top_three = set(
        importance.nsmallest(3, "lightgbm_split_rank")["feature"].tolist()
    )
    top_three_overlap = len(shap_top_three & split_top_three)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    sampled_feature_display = sampled.loc[
        :, [*X.columns, "lat", "lon"]
    ].reset_index(drop=True)
    probabilities = pipeline.predict_proba(sampled_X)[:, 1]
    examples = _save_waterfalls(
        explanation,
        sampled_feature_display,
        sampled_y,
        probabilities,
        sampled_row_indices,
        cutoff,
        names,
    )
    examples.to_csv(EXAMPLES_PATH, index=False)
    plot_paths = _save_global_plots(
        explanation,
        importance,
        sampled_feature_display,
        names,
    )
    print(f"Saved feature ranking to {IMPORTANCE_PATH}")
    print(f"Saved example rows to {EXAMPLES_PATH}")
    print(f"Saved {len(plot_paths) + len(examples)} figures under {FIGURES_DIR}")

    return ShapRunResult(
        importance=importance.reset_index(drop=True),
        examples=examples,
        rank_correlation=rank_correlation,
        top_three_overlap=top_three_overlap,
        cutoff=cutoff,
        sample_size=SHAP_SAMPLE_SIZE,
        row_count=len(training),
    )


if __name__ == "__main__":
    result = run_shap_analysis()
    print(
        f"Spearman correlation between SHAP and LightGBM split importance: "
        f"{result.rank_correlation:.3f}; top-three overlap: "
        f"{result.top_three_overlap}/3."
    )
