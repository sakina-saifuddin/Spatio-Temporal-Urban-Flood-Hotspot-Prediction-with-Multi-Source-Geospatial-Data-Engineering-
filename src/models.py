"""Baseline model constructors and registry."""

from collections.abc import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import SEED
from src.mlp import TorchMLPClassifier

ModelFactory = Callable[[], object]


def get_dummy() -> DummyClassifier:
    """Return an unfitted prior-probability dummy classifier."""
    return DummyClassifier(strategy="prior")


def get_logistic_regression() -> LogisticRegression:
    """Return an unfitted, seeded logistic regression classifier."""
    return LogisticRegression(max_iter=1000, random_state=SEED)


def get_random_forest() -> RandomForestClassifier:
    """Return an unfitted, seeded random forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    )


def get_xgboost() -> XGBClassifier:
    """Return an unfitted, seeded XGBoost classifier."""
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def get_lightgbm() -> LGBMClassifier:
    """Return an unfitted, seeded LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def get_mlp() -> TorchMLPClassifier:
    """Return an unfitted PyTorch MLP classifier."""
    return TorchMLPClassifier()


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "dummy": get_dummy,
    "logistic_regression": get_logistic_regression,
    "random_forest": get_random_forest,
    "xgboost": get_xgboost,
    "lightgbm": get_lightgbm,
    "mlp": get_mlp,
}
