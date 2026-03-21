"""XGBoost classifier.

Wraps xgboost's XGBClassifier in a StandardScaler pipeline.
Default hyperparameters from ``configs/classifiers/xgboost.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier as _XGBClassifier

from loato_bench.classifiers.base import Classifier


class XGBoostClassifier(Classifier):
    """XGBoost with StandardScaler preprocessing.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    eval_metric : str
        Evaluation metric for training.
    tree_method : str
        Tree construction algorithm.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    _XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        eval_metric=eval_metric,
                        tree_method=tree_method,
                        random_state=42,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        preds: NDArray[np.int64] = np.asarray(self._pipeline.predict(X), dtype=np.int64)
        return preds

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = np.asarray(self._pipeline.predict_proba(X), dtype=np.float64)
        return proba
