"""Planner ranking, exploratory cutoffs, and historical metric helpers.

All functions are pure pandas/NumPy transformations of the precomputed data.
No model is fitted and no independent evaluation is rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FROZEN_CUTOFF = 0.5

RANKING_METRICS = {
    "mean_score": "Mean susceptibility score",
    "max_score": "Maximum sampled score",
    "share_at_cutoff": "Share of sampled locations at or above the selected cutoff",
    "sample_count": "Number of sampled locations",
}


def share_at_or_above(locations: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Per-grid count and share of sampled locations with score >= cutoff.

    Computed from individual location scores, never from the grid mean.
    """
    flags = locations["susceptibility_score"].to_numpy() >= cutoff
    frame = pd.DataFrame({"grid_id": locations["grid_id"].to_numpy(), "flag": flags})
    grouped = frame.groupby("grid_id", sort=False)["flag"].agg(["sum", "count"])
    return pd.DataFrame(
        {
            "grid_id": grouped.index,
            "count_at_cutoff": grouped["sum"].astype(int).to_numpy(),
            "share_at_cutoff": (grouped["sum"] / grouped["count"]).to_numpy(),
        }
    ).reset_index(drop=True)


def rank_grids(
    grid: pd.DataFrame,
    shares: pd.DataFrame,
    metric: str = "mean_score",
    min_samples: int = 1,
    score_range: tuple[float, float] = (0.0, 1.0),
    coverage: list[str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Filter and rank sampled grid cells. Unsampled cells are never included."""
    if metric not in RANKING_METRICS:
        raise ValueError(f"Unknown ranking metric: {metric}")
    data = grid.merge(shares, on="grid_id", how="left", validate="one_to_one")
    low, high = score_range
    mask = (
        (data["sample_count"] >= min_samples)
        & (data["mean_score"] >= low)
        & (data["mean_score"] <= high)
    )
    if coverage is not None:
        mask &= data["coverage_category"].isin(coverage)
    data = data[mask]
    # Ties broken deterministically by mean score, then sample count, then ID.
    secondary = [c for c in ("mean_score", "sample_count") if c != metric]
    data = data.sort_values(
        [metric, *secondary, "grid_id"],
        ascending=[False] * (1 + len(secondary)) + [True],
        kind="mergesort",
    )
    if top_n is not None:
        data = data.head(top_n)
    data = data.reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data


def summary_statistics(ranked: pd.DataFrame) -> dict[str, float]:
    if ranked.empty:
        return {"cells": 0, "locations": 0, "mean_score": float("nan"), "mean_share": float("nan")}
    return {
        "cells": int(len(ranked)),
        "locations": int(ranked["sample_count"].sum()),
        "mean_score": float(ranked["mean_score"].mean()),
        "mean_share": float(
            ranked["count_at_cutoff"].sum() / ranked["sample_count"].sum()
        ),
    }


def confusion_counts(historical: pd.DataFrame, storm: str | None = None) -> dict[str, int]:
    data = historical if storm is None else historical[historical["storm_name"] == storm]
    counts = data["classification"].value_counts()
    return {key: int(counts.get(key, 0)) for key in ("TP", "TN", "FP", "FN")}


def classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def shap_reconstruction_error(locations: pd.DataFrame) -> float:
    """Maximum absolute gap between baseline + grouped SHAP and raw model score."""
    shap_cols = [c for c in locations.columns if c.startswith("shap_") and c != "shap_base_value"]
    rebuilt = locations["shap_base_value"] + locations[shap_cols].sum(axis=1)
    return float((rebuilt - locations["raw_model_score"]).abs().max())


def cell_classification_breakdown(row: pd.Series) -> pd.DataFrame:
    rows = []
    for key in ("TP", "TN", "FP", "FN"):
        rows.append(
            {
                "classification": key,
                "count": int(row[f"{key.lower()}_count"]),
                "share": float(row[f"{key.lower()}_share"]),
            }
        )
    return pd.DataFrame(rows)
