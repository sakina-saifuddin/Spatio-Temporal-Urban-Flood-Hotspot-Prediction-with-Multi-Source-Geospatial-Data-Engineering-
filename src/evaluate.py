"""Cross-validation metrics, out-of-fold predictions, and evaluation plots."""

from collections.abc import Callable, Iterable
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import REPORTS_DIR
from src.features import Split, StormSplit, build_preprocessor, get_xy

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
)
RESULTS_PATH = REPORTS_DIR / "results.csv"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CrossValidationSplit: TypeAlias = Split | StormSplit


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute binary flood metrics with flooded=1 as the positive class."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    true_values = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if true_values.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")
    if len(true_values) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same number of rows.")
    if len(true_values) == 0:
        raise ValueError("Cannot compute metrics on empty inputs.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("y_prob must contain finite probabilities between 0 and 1.")
    if not set(np.unique(true_values)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        true_values, predictions, labels=[0, 1]
    ).ravel()
    has_both_classes = len(np.unique(true_values)) == 2
    return {
        "accuracy": float(accuracy_score(true_values, predictions)),
        "precision": float(precision_score(true_values, predictions, zero_division=0)),
        "recall": float(recall_score(true_values, predictions, zero_division=0)),
        "f1": float(f1_score(true_values, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_values, probabilities))
        if has_both_classes
        else float("nan"),
        "pr_auc": float(average_precision_score(true_values, probabilities))
        if has_both_classes
        else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validate(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    splits: Iterable[CrossValidationSplit],
    name: str,
    split_name: str,
    drop_features: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fold-local pipelines and return fold metrics plus out-of-fold predictions."""
    drop_features = tuple(drop_features)
    X, y = get_xy(df, drop_features=drop_features)
    fold_metrics: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_number, split in enumerate(splits, start=1):
        if len(split) == 2:
            train_indices, validation_indices = split
        elif len(split) == 3:
            _, train_indices, validation_indices = split
        else:
            raise ValueError(
                "Each split must contain train/validation indices, optionally prefixed by a storm name."
            )

        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"Fold {fold_number} has empty training or validation indices.")

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(drop_features=drop_features)),
                ("model", model_factory()),
            ]
        )
        pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
        model = pipeline.named_steps["model"]
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("The fitted model does not expose class 1 for probability prediction.")
        positive_index = classes.index(1)
        probabilities = pipeline.predict_proba(X.iloc[validation_indices])[:, positive_index]
        true_values = y.iloc[validation_indices]
        metrics = compute_metrics(true_values, probabilities)
        fold_metrics.append(
            {
                "model": name,
                "split": split_name,
                "fold": fold_number,
                **metrics,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "row_index": df.index.to_numpy()[validation_indices],
                    "y_true": true_values.to_numpy(),
                    "y_prob": probabilities,
                    "fold": fold_number,
                }
            )
        )

    if not fold_metrics:
        raise ValueError("splits must contain at least one fold.")

    summary = pd.DataFrame(fold_metrics)
    metric_means = summary[list(METRIC_COLUMNS)].mean()
    metric_stds = summary[list(METRIC_COLUMNS)].std(ddof=1).fillna(0.0)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {"model": name, "split": split_name, "fold": "mean", **metric_means.to_dict()},
                    {"model": name, "split": split_name, "fold": "std", **metric_stds.to_dict()},
                ]
            ),
        ],
        ignore_index=True,
    )
    oof = pd.concat(oof_frames, ignore_index=True)
    return summary, oof


def save_results(
    summary_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    name: str,
    split_name: str,
) -> None:
    """Replace this model/split summary in results.csv and save its OOF predictions."""
    required_summary = {"model", "split", "fold", *METRIC_COLUMNS}
    missing = required_summary - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {sorted(missing)}")
    required_oof = {"row_index", "y_true", "y_prob", "fold"}
    missing_oof = required_oof - set(oof_df.columns)
    if missing_oof:
        raise ValueError(f"oof_df is missing required columns: {sorted(missing_oof)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mean_rows = summary_df.loc[summary_df["fold"].eq("mean")].copy()
    if len(mean_rows) != 1:
        raise ValueError("summary_df must contain exactly one mean row.")
    mean_rows.loc[:, "model"] = name
    mean_rows.loc[:, "split"] = split_name

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        if not required_summary.issubset(existing.columns):
            raise ValueError(f"{RESULTS_PATH} does not have the expected result columns.")
        existing = existing.loc[
            ~existing["model"].eq(name) | ~existing["split"].eq(split_name)
        ]
        results = pd.concat([existing, mean_rows], ignore_index=True)
    else:
        results = mean_rows
    results.to_csv(RESULTS_PATH, index=False)

    oof_path = REPORTS_DIR / f"oof_{_safe_name(name)}_{_safe_name(split_name)}.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved OOF predictions to {oof_path}")


def plot_confusion_matrix(
    oof_df: pd.DataFrame, threshold: float = 0.5
) -> Figure:
    """Plot and save the OOF confusion matrix at the requested threshold."""
    _require_oof_columns(oof_df)
    metrics = compute_metrics(oof_df["y_true"], oof_df["y_prob"], threshold)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:,}", ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Not flooded (0)", "Flooded (1)"])
    ax.set_yticks([0, 1], labels=["Not flooded (0)", "Flooded (1)"])
    ax.set(xlabel="Predicted class", ylabel="Actual class", title=f"Confusion matrix (threshold={threshold:.2f})")
    fig.colorbar(image, ax=ax, label="Number of rows")
    fig.tight_layout()
    return _save_figure(fig, "confusion_matrix.png")


def plot_roc_pr(oof_df: pd.DataFrame) -> Figure:
    """Plot and save ROC and precision-recall curves from OOF predictions."""
    _require_oof_columns(oof_df)
    y_true = oof_df["y_true"].to_numpy()
    y_prob = oof_df["y_prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC and PR curves require both classes in y_true.")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"Average precision = {pr_auc:.3f}")
    axes[1].axhline(y_true.mean(), linestyle="--", color="gray", label="Positive prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    fig.tight_layout()
    return _save_figure(fig, "roc_pr_curves.png")


def _require_oof_columns(oof_df: pd.DataFrame) -> None:
    """Validate fields needed to plot out-of-fold predictions."""
    missing = {"y_true", "y_prob"} - set(oof_df.columns)
    if missing:
        raise ValueError(f"oof_df is missing required columns: {sorted(missing)}")


def _save_figure(fig: Figure, filename: str) -> Figure:
    """Save a figure under reports/figures and return it."""
    output_path = FIGURES_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return fig


def _safe_name(value: str) -> str:
    """Convert a model or split label to a filesystem-safe filename component."""
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
