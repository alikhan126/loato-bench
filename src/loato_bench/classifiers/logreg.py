"""Logistic Regression classifier.

Wraps sklearn's LogisticRegression in a StandardScaler pipeline.
Default hyperparameters from ``configs/classifiers/logreg.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loato_bench.classifiers.base import Classifier


class LogRegClassifier(Classifier):
    """Logistic Regression with StandardScaler preprocessing.

    Parameters
    ----------
    C : float
        Inverse regularization strength.
    max_iter : int
        Maximum iterations for the solver.
    solver : str
        Optimization algorithm.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver=solver,
                        random_state=42,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "logreg"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        preds: NDArray[np.int64] = np.asarray(self._pipeline.predict(X), dtype=np.int64)
        return preds

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = np.asarray(self._pipeline.predict_proba(X), dtype=np.float64)
        return proba
