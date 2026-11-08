"""Feature preparation and leakage-safe geographic validation splits."""

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

Split: TypeAlias = tuple[np.ndarray, np.ndarray]
StormSplit: TypeAlias = tuple[str, np.ndarray, np.ndarray]


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a clear error if required feature or grouping columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select configured model features and the target without leakage columns."""
    features = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    _require_columns(df, (*features, TARGET))
    return df.loc[:, list(features)].copy(), df[TARGET].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted transformer to fit on each fold's training rows only."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def area_group_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> Iterable[Split]:
    """Yield GroupKFold row indices with each grid cell held in one side only."""
    _require_columns(df, ("grid_id",))
    if df["grid_id"].isna().any():
        raise ValueError("grid_id must not contain missing values.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    group_count = df["grid_id"].nunique()
    if n_splits > group_count:
        raise ValueError(
            f"n_splits ({n_splits}) cannot exceed grid_id groups ({group_count})."
        )

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(df, groups=df["grid_id"])


def leave_one_storm_out_splits(
    df: pd.DataFrame, drop_overlap_areas: bool = False
) -> Iterable[StormSplit]:
    """Yield storm holdouts, optionally removing shared areas from training."""
    _require_columns(df, ("storm_name", "grid_id"))
    if df[["storm_name", "grid_id"]].isna().any().any():
        raise ValueError("storm_name and grid_id must not contain missing values.")

    storm_values = df["storm_name"].drop_duplicates().tolist()
    all_indices = np.arange(len(df))
    for storm_name in storm_values:
        validation_mask = df["storm_name"].eq(storm_name).to_numpy()
        validation_indices = all_indices[validation_mask]
        training_mask = ~validation_mask
        if drop_overlap_areas:
            validation_grid_ids = set(df.iloc[validation_indices]["grid_id"])
            training_mask &= ~df["grid_id"].isin(validation_grid_ids).to_numpy()
        training_indices = all_indices[training_mask]
        yield str(storm_name), training_indices, validation_indices


def describe_splits(
    splits: Iterable[Split | StormSplit],
    df: pd.DataFrame,
    allow_overlap: bool = False,
) -> None:
    """Print split sizes and class balances, checking grid overlap as requested."""
    _require_columns(df, ("grid_id", TARGET))

    for split_number, split in enumerate(splits, start=1):
        if len(split) == 2:
            train_indices, validation_indices = split
            split_name = f"Split {split_number}"
        elif len(split) == 3:
            split_name, train_indices, validation_indices = split
            split_name = f"Storm {split_name}"
        else:
            raise ValueError("Each split must contain 2 indices arrays or a storm name and 2 arrays.")

        train_df = df.iloc[train_indices]
        validation_df = df.iloc[validation_indices]
        train_grids = set(train_df["grid_id"].dropna())
        validation_grids = set(validation_df["grid_id"].dropna())
        shared_grids = train_grids & validation_grids
        if not allow_overlap:
            assert not shared_grids, (
                f"{split_name} has grid_id overlap between training and validation."
            )

        train_counts = train_df[TARGET].value_counts(dropna=False).sort_index()
        validation_counts = validation_df[TARGET].value_counts(dropna=False).sort_index()
        train_balance = (train_counts / len(train_df)).round(3) if len(train_df) else train_counts
        validation_balance = (
            (validation_counts / len(validation_df)).round(3)
            if len(validation_df)
            else validation_counts
        )
        print(f"{split_name}: train={len(train_df)}, validation={len(validation_df)}")
        print(f"  train class counts: {train_counts.to_dict()}")
        print(f"  train class proportions: {train_balance.to_dict()}")
        print(f"  validation class counts: {validation_counts.to_dict()}")
        print(f"  validation class proportions: {validation_balance.to_dict()}")
        if allow_overlap:
            overlap_percent = (
                100 * len(shared_grids) / len(validation_grids)
                if validation_grids
                else 0.0
            )
            print(
                f"  validation grid_ids also in training: "
                f"{len(shared_grids)}/{len(validation_grids)} "
                f"({overlap_percent:.1f}%)"
            )
        else:
            print("  grid_id overlap: none")
