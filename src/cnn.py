"""Terrain CNN models and fold-local cross-validation."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn

from src.config import SEED
from src.evaluate import CrossValidationSplit, compute_metrics, save_results
from src.features import build_preprocessor, get_xy

FoldLosses: TypeAlias = dict[int, list[float]]


class TileCNN(nn.Module):
    """CNN branches for terrain-only and terrain-plus-table inputs."""

    def __init__(self, variant: str, table_feature_count: int = 0) -> None:
        super().__init__()
        if variant not in {"cnn_tile_only", "cnn_hybrid"}:
            raise ValueError(f"Unknown CNN variant: {variant}")
        if variant == "cnn_hybrid" and table_feature_count < 1:
            raise ValueError("cnn_hybrid requires at least one table feature.")

        self.variant = variant
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        if variant == "cnn_tile_only":
            self.output = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(64 + table_feature_count, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

    def forward(
        self,
        tiles: torch.Tensor,
        table_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one flood logit per terrain tile."""
        encoded = self.convolution(tiles).flatten(start_dim=1)
        if self.variant == "cnn_hybrid":
            if table_features is None:
                raise ValueError("cnn_hybrid requires table features.")
            encoded = torch.cat((encoded, table_features), dim=1)
        return self.output(encoded).squeeze(1)


class TorchCNNClassifier:
    """Train a fixed-setting CNN on relative terrain tiles and optional features."""

    def __init__(self, variant: str, seed: int = SEED) -> None:
        if variant not in {"cnn_tile_only", "cnn_hybrid"}:
            raise ValueError(f"Unknown CNN variant: {variant}")
        self.variant = variant
        self.seed = seed
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.model_: TileCNN | None = None
        self.loss_: list[float] = []
        self.device_: torch.device | None = None

    def fit(
        self,
        tiles: np.ndarray,
        y: pd.Series | np.ndarray,
        table_features: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ) -> TorchCNNClassifier:
        """Fit for a fixed number of epochs and record mean loss per epoch."""
        if epochs != 20 or batch_size != 256 or learning_rate != 0.001:
            raise ValueError("CNN training settings are fixed at 20 epochs, batch size 256, LR 0.001.")
        if sparse.issparse(tiles):
            raise TypeError("Terrain tiles must be a dense array.")
        tile_array = np.asarray(tiles)
        labels = np.asarray(y, dtype=np.float32).reshape(-1)
        if tile_array.ndim != 3 or tile_array.shape[1:] != (32, 32):
            raise ValueError(f"Expected terrain shape (n, 32, 32), got {tile_array.shape}.")
        if len(tile_array) != len(labels) or not np.isfinite(tile_array).all():
            raise ValueError("Terrain tiles and labels must have matching rows and finite values.")
        if not set(np.unique(labels)).issubset({0.0, 1.0}):
            raise ValueError("CNN labels must contain only 0 and 1.")

        table_array: np.ndarray | None = None
        if self.variant == "cnn_hybrid":
            if table_features is None:
                raise ValueError("cnn_hybrid requires preprocessed table features.")
            if sparse.issparse(table_features):
                table_features = table_features.toarray()
            table_array = np.asarray(table_features, dtype=np.float32)
            if table_array.ndim != 2 or table_array.shape[0] != len(labels):
                raise ValueError("Table features must be a 2D array aligned with the labels.")
            if not np.isfinite(table_array).all():
                raise ValueError("Table features must be finite.")
        elif table_features is not None:
            raise ValueError("cnn_tile_only does not accept table features.")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        self.model_ = TileCNN(
            self.variant,
            table_feature_count=0 if table_array is None else table_array.shape[1],
        ).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=learning_rate)
        loss_function = nn.BCEWithLogitsLoss()
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        order = np.arange(len(labels))
        self.loss_ = []

        for _ in range(epochs):
            self.model_.train()
            order = torch.randperm(
                len(labels),
                generator=generator,
            ).numpy()
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                tile_batch = torch.as_tensor(
                    np.asarray(tile_array[batch_indices], dtype=np.float32)[:, None, :, :],
                    device=device,
                )
                label_batch = torch.as_tensor(labels[batch_indices], device=device)
                table_batch = (
                    torch.as_tensor(table_array[batch_indices], device=device)
                    if table_array is not None
                    else None
                )

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(tile_batch, table_batch)
                loss = loss_function(logits, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch_indices)
            self.loss_.append(total_loss / len(labels))
        return self

    def predict_proba(
        self,
        tiles: np.ndarray,
        table_features: np.ndarray | None = None,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Return probabilities for classes 0 and 1."""
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("The CNN must be fitted before predicting.")
        tile_array = np.asarray(tiles)
        if tile_array.ndim != 3 or tile_array.shape[1:] != (32, 32):
            raise ValueError(f"Expected terrain shape (n, 32, 32), got {tile_array.shape}.")
        table_array: np.ndarray | None = None
        if self.variant == "cnn_hybrid":
            if table_features is None:
                raise ValueError("cnn_hybrid requires preprocessed table features.")
            if sparse.issparse(table_features):
                table_features = table_features.toarray()
            table_array = np.asarray(table_features, dtype=np.float32)
            if table_array.ndim != 2 or table_array.shape[0] != len(tile_array):
                raise ValueError("Table features must be a 2D array aligned with the tiles.")
        elif table_features is not None:
            raise ValueError("cnn_tile_only does not accept table features.")

        probabilities = np.empty((len(tile_array), 2), dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(tile_array), batch_size):
                end = min(start + batch_size, len(tile_array))
                tile_batch = torch.as_tensor(
                    np.asarray(tile_array[start:end], dtype=np.float32)[:, None, :, :],
                    device=self.device_,
                )
                table_batch = (
                    torch.as_tensor(table_array[start:end], device=self.device_)
                    if table_array is not None
                    else None
                )
                logits = self.model_(tile_batch, table_batch)
                positive = torch.sigmoid(logits).cpu().numpy()
                probabilities[start:end, 1] = positive
                probabilities[start:end, 0] = 1.0 - positive
        return probabilities


def cross_validate_cnn(
    variant: str,
    df: pd.DataFrame,
    tiles: np.ndarray,
    splits: Iterable[CrossValidationSplit],
    split_name: str,
    drop_features: Iterable[str] = (),
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run fold-local CNN validation and return metrics, OOF predictions, diagnostics."""
    if len(tiles) != len(df):
        raise ValueError("Terrain array must have one row for every DataFrame row.")
    if variant == "cnn_hybrid":
        drop_features = tuple(drop_features)
        if "rainfall_inches" not in drop_features:
            raise ValueError("cnn_hybrid must exclude rainfall_inches.")
    elif variant != "cnn_tile_only":
        raise ValueError(f"Unknown CNN variant: {variant}")

    X_table, y = get_xy(df, drop_features=drop_features) if variant == "cnn_hybrid" else (None, df["flooded"])
    fold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    fold_losses: FoldLosses = {}
    estimators: dict[int, TorchCNNClassifier] = {}
    fold_seconds: dict[int, float] = {}

    for fold_number, split in enumerate(splits, start=1):
        if len(split) == 2:
            train_indices, validation_indices = split
        elif len(split) == 3:
            _, train_indices, validation_indices = split
        else:
            raise ValueError("Each split must contain two indices or storm name and two indices.")
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"Fold {fold_number} has empty training or validation indices.")

        table_train = None
        table_validation = None
        if variant == "cnn_hybrid":
            preprocessor = build_preprocessor(drop_features=drop_features)
            table_train = preprocessor.fit_transform(X_table.iloc[train_indices])
            table_validation = preprocessor.transform(X_table.iloc[validation_indices])
            if sparse.issparse(table_train):
                table_train = table_train.toarray()
                table_validation = table_validation.toarray()
            table_train = np.asarray(table_train, dtype=np.float32)
            table_validation = np.asarray(table_validation, dtype=np.float32)

        estimator = TorchCNNClassifier(variant)
        started = time.perf_counter()
        estimator.fit(
            tiles[train_indices],
            y.iloc[train_indices],
            table_features=table_train,
        )
        probabilities = estimator.predict_proba(
            tiles[validation_indices],
            table_features=table_validation,
        )[:, 1]
        elapsed = time.perf_counter() - started
        metrics = compute_metrics(y.iloc[validation_indices], probabilities)
        fold_rows.append(
            {
                "model": variant,
                "split": split_name,
                "fold": fold_number,
                **metrics,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "row_index": df.index.to_numpy()[validation_indices],
                    "y_true": y.iloc[validation_indices].to_numpy(),
                    "y_prob": probabilities,
                    "fold": fold_number,
                }
            )
        )
        fold_losses[fold_number] = estimator.loss_.copy()
        fold_seconds[fold_number] = elapsed
        estimators[fold_number] = estimator

    if not fold_rows:
        raise ValueError("splits must contain at least one fold.")
    summary = pd.DataFrame(fold_rows)
    metric_columns = (
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "tn", "fp", "fn", "tp",
    )
    means = summary[list(metric_columns)].mean()
    stds = summary[list(metric_columns)].std(ddof=1).fillna(0.0)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "model": variant,
                        "split": split_name,
                        "fold": "mean",
                        **means.to_dict(),
                    },
                    {
                        "model": variant,
                        "split": split_name,
                        "fold": "std",
                        **stds.to_dict(),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    oof = pd.concat(oof_frames, ignore_index=True)
    if save:
        save_results(summary, oof, variant, split_name)
    diagnostics: dict[str, object] = {
        "fold_losses": fold_losses,
        "fold_seconds": fold_seconds,
        "estimators": estimators,
    }
    return summary, oof, diagnostics
