"""Base class for classifiers."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Classifier(ABC):
    """Abstract base class for classifiers.

    All classifier implementations (LogReg, SVM, XGBoost, MLP)
    must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this classifier (e.g., 'logreg', 'xgboost')."""
        ...

    @abstractmethod
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        """Train the classifier on embedding vectors and labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,), values in {0, 1}.
        """
        ...

    @abstractmethod
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels for the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,), values in {0, 1}.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Predict class probabilities for the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, 2) where columns
            are [P(benign), P(injection)].
        """
        ...
