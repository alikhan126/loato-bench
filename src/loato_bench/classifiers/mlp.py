"""MLP (2-layer) classifier.

Wraps sklearn's MLPClassifier in a StandardScaler pipeline with early stopping.
Default hyperparameters from ``configs/classifiers/mlp.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loato_bench.classifiers.base import Classifier


class MLPClassifier(Classifier):
    """2-layer MLP with StandardScaler preprocessing and early stopping.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Number of neurons in each hidden layer.
    learning_rate_init : float
        Initial learning rate.
    max_iter : int
        Maximum training epochs.
    early_stopping : bool
        Whether to use early stopping on a validation set.
    validation_fraction : float
        Fraction of training data for early stopping validation.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    _MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        learning_rate_init=learning_rate_init,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction,
                        random_state=42,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "mlp"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        preds: NDArray[np.int64] = np.asarray(self._pipeline.predict(X), dtype=np.int64)
        return preds

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = np.asarray(self._pipeline.predict_proba(X), dtype=np.float64)
        return proba
