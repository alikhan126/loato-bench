"""SVM (RBF kernel) classifier.

Wraps sklearn's SVC in a StandardScaler pipeline with ``probability=True``
for calibrated probability estimates. Optional PCA dimensionality reduction
makes SVM tractable on high-dimensional embeddings (e.g., 4096-d E5-Mistral).

For large datasets (>10K samples), uses Nystroem kernel approximation +
SGDClassifier for O(n) training time instead of O(n²-n³) SVC, while
preserving RBF kernel semantics.

Default hyperparameters from ``configs/classifiers/svm.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loato_bench.classifiers.base import Classifier

# Threshold above which Nystroem approximation is used instead of exact SVC
_NYSTROEM_THRESHOLD = 10_000


class SVMClassifier(Classifier):
    """SVM with RBF kernel, StandardScaler, and optional PCA preprocessing.

    For datasets with ≤10K samples, uses exact SVC with probability=True.
    For larger datasets, uses Nystroem RBF kernel approximation + SGDClassifier
    with CalibratedClassifierCV for probability estimates. This gives equivalent
    RBF kernel behavior in O(n) time instead of O(n²-n³).

    Parameters
    ----------
    C : float
        Regularization parameter.
    kernel : str
        Kernel type (default: "rbf").
    gamma : str | float
        Kernel coefficient.
    pca_components : int | None
        If set, reduce dimensionality via PCA before SVM. Applied after
        StandardScaler. Use 128 for high-dimensional embeddings.
    nystroem_components : int
        Number of Nystroem components for kernel approximation (large data).
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        pca_components: int | None = None,
        nystroem_components: int = 500,
    ) -> None:
        self._pca_components = pca_components
        self._nystroem_components = nystroem_components
        self._C = C
        self._kernel = kernel
        self._gamma = gamma
        self._fitted = False
        # Build exact SVC pipeline (used for small data or as default)
        self._pipeline = self._build_exact_pipeline(C, kernel, gamma, pca_components)
        # Approximate pipeline built lazily in fit() when n > threshold
        self._approx_pipeline: Pipeline | None = None

    def _build_exact_pipeline(
        self,
        C: float,
        kernel: str,
        gamma: str | float,
        pca_components: int | None,
    ) -> Pipeline:
        """Build exact SVC pipeline for small datasets."""
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if pca_components is not None:
            steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
        steps.append(
            (
                "clf",
                SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42),
            ),
        )
        return Pipeline(steps)

    def _build_approx_pipeline(self, pca_components: int | None) -> Pipeline:
        """Build Nystroem-approximated SVM pipeline for large datasets."""
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if pca_components is not None:
            steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
        # Nystroem maps input to an approximate RBF kernel feature space
        steps.append(
            (
                "nystroem",
                Nystroem(
                    kernel=self._kernel,
                    gamma=self._gamma if isinstance(self._gamma, float) else None,
                    n_components=self._nystroem_components,
                    random_state=42,
                ),
            ),
        )
        # SGDClassifier with hinge loss = linear SVM in the Nystroem feature space
        # Wrap in CalibratedClassifierCV for predict_proba support
        sgd = SGDClassifier(
            loss="hinge",
            alpha=1.0 / (self._C * self._nystroem_components),
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        steps.append(("clf", CalibratedClassifierCV(sgd, cv=3)))
        return Pipeline(steps)

    @property
    def name(self) -> str:
        return "svm"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        n_samples = X.shape[0]
        if n_samples > _NYSTROEM_THRESHOLD:
            self._approx_pipeline = self._build_approx_pipeline(self._pca_components)
            self._approx_pipeline.fit(X, y)
        else:
            self._pipeline.fit(X, y)
        self._fitted = True

    @property
    def _active_pipeline(self) -> Pipeline:
        """Return whichever pipeline was fitted."""
        if self._approx_pipeline is not None:
            return self._approx_pipeline
        return self._pipeline

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        preds: NDArray[np.int64] = np.asarray(self._active_pipeline.predict(X), dtype=np.int64)
        return preds

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = np.asarray(
            self._active_pipeline.predict_proba(X), dtype=np.float64
        )
        return proba
