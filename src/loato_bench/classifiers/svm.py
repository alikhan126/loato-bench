"""SVM (RBF kernel) classifier.

Wraps sklearn's SVC in a StandardScaler pipeline with ``probability=True``
for calibrated probability estimates. Default hyperparameters from
``configs/classifiers/svm.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loato_bench.classifiers.base import Classifier


class SVMClassifier(Classifier):
    """SVM with RBF kernel and StandardScaler preprocessing.

    Parameters
    ----------
    C : float
        Regularization parameter.
    kernel : str
        Kernel type (default: "rbf").
    gamma : str | float
        Kernel coefficient.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=C,
                        kernel=kernel,
                        gamma=gamma,
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "svm"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        preds: NDArray[np.int64] = np.asarray(self._pipeline.predict(X), dtype=np.int64)
        return preds

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = np.asarray(self._pipeline.predict_proba(X), dtype=np.float64)
        return proba
