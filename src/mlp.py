"""PyTorch MLP classifier compatible with scikit-learn pipelines."""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEED


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Train a fixed two-hidden-layer MLP on CPU."""

    def fit(self, X: object, y: object) -> TorchMLPClassifier:
        """Fit for 25 epochs and store the mean loss for each epoch."""
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        features = X.toarray() if sparse.issparse(X) else np.asarray(X)
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError("X and y must contain only finite values.")
        if not set(np.unique(labels)).issubset({0.0, 1.0}):
            raise ValueError("y must contain only binary labels 0 and 1.")

        self.classes_ = [0, 1]
        self.n_features_in_ = features.shape[1]
        self.model_ = nn.Sequential(
            nn.Linear(self.n_features_in_, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        ).to("cpu")

        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels),
        )
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.001)
        self.loss_ = []

        self.model_.train()
        for _ in range(25):
            total_loss = 0.0
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_features)
                loss = loss_function(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_features)
            self.loss_.append(total_loss / len(dataset))

        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Return probabilities for classes 0 and 1."""
        if not hasattr(self, "model_"):
            raise ValueError("The classifier must be fitted before prediction.")

        features = X.toarray() if sparse.issparse(X) else np.asarray(X)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} feature columns.")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")

        self.model_.eval()
        with torch.inference_mode():
            logits = self.model_(torch.from_numpy(features)).squeeze(1)
            positive_probability = torch.sigmoid(logits).numpy()
        return np.column_stack((1 - positive_probability, positive_probability))
