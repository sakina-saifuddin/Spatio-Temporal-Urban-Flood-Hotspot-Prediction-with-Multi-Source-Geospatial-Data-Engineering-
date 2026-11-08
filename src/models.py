"""Baseline model constructors and registry."""

from collections.abc import Callable

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import SEED

ModelFactory = Callable[[], object]


def get_dummy() -> DummyClassifier:
    """Return an unfitted prior-probability dummy classifier."""
    return DummyClassifier(strategy="prior")


def get_logistic_regression() -> LogisticRegression:
    """Return an unfitted, seeded logistic regression classifier."""
    return LogisticRegression(max_iter=1000, random_state=SEED)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "dummy": get_dummy,
    "logistic_regression": get_logistic_regression,
}
