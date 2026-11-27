"""Threshold selection helpers for saved out-of-fold predictions."""

from collections.abc import Iterable

import pandas as pd

from src.evaluate import compute_metrics


def recall_table(
    oof_df: pd.DataFrame, thresholds: Iterable[float]
) -> pd.DataFrame:
    """Return classification metrics for each probability threshold."""
    required = {"y_true", "y_prob"}
    missing = required - set(oof_df.columns)
    if missing:
        raise ValueError(f"oof_df is missing required columns: {sorted(missing)}")

    rows = []
    for threshold in thresholds:
        metrics = compute_metrics(oof_df["y_true"], oof_df["y_prob"], threshold)
        rows.append(
            {
                "threshold": threshold,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "threshold",
            "recall",
            "precision",
            "f1",
            "accuracy",
            "fp",
            "fn",
        ),
    )


def pick_threshold(
    oof_df: pd.DataFrame, target_recall: float = 0.85
) -> float | None:
    """Return the highest 0.05–0.95 threshold meeting recall, or None."""
    thresholds = (step / 100 for step in range(5, 96))
    table = recall_table(oof_df, thresholds)
    eligible = table.loc[table["recall"].ge(target_recall), "threshold"]
    if eligible.empty:
        return None
    return float(eligible.max())
