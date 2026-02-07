"""Tests for classifier base class — Sprint 2B."""

import numpy as np

from promptguard.classifiers.base import Classifier


def test_classifier_is_abstract():
    """Classifier cannot be instantiated directly."""
    try:
        Classifier()  # type: ignore[abstract]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


class DummyClassifier(Classifier):
    """Concrete dummy for testing the interface."""

    @property
    def name(self) -> str:
        return "dummy"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._classes = np.unique(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.ones(n) * 0.7, np.ones(n) * 0.3])


def test_dummy_classifier_fit_predict():
    """Concrete classifier should fit and predict with correct shapes."""
    clf = DummyClassifier()
    X = np.random.randn(50, 16).astype(np.float32)
    y = np.array([0] * 25 + [1] * 25, dtype=np.int64)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (50,)
    proba = clf.predict_proba(X)
    assert proba.shape == (50, 2)
