"""LOATO (Leave-One-Attack-Type-Out) evaluation protocol.

Runs the primary LOATO experiment: for each fold, train on K-1 attack
categories (plus benign), test on the held-out category (plus benign).
Also runs standard stratified 5-fold CV as the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from loato_bench.classifiers.base import Classifier
from loato_bench.evaluation.metrics import EvalMetrics, bootstrap_ci, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result for a single fold (LOATO or standard CV).

    Attributes
    ----------
    fold_id : int
        Fold index.
    held_out_category : str | None
        Attack category held out (None for standard CV).
    train_size : int
        Number of training samples.
    test_size : int
        Number of test samples.
    metrics : EvalMetrics
        Evaluation metrics (with optional bootstrap CIs).
    """

    fold_id: int
    held_out_category: str | None
    train_size: int
    test_size: int
    metrics: EvalMetrics

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "fold_id": self.fold_id,
            "held_out_category": self.held_out_category,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class ExperimentResult:
    """Aggregated result across all folds.

    Attributes
    ----------
    experiment : str
        Experiment name (e.g., "loato", "standard_cv").
    embedding : str
        Embedding model name.
    classifier : str
        Classifier name.
    folds : list[FoldResult]
        Per-fold results.
    mean_f1 : float
        Mean macro F1 across folds.
    std_f1 : float
        Standard deviation of F1 across folds.
    """

    experiment: str
    embedding: str
    classifier: str
    folds: list[FoldResult]
    mean_f1: float
    std_f1: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "experiment": self.experiment,
            "embedding": self.embedding,
            "classifier": self.classifier,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "folds": [f.to_dict() for f in self.folds],
        }


def _load_split(split_path: Path) -> dict[str, Any]:
    """Load a split JSON file, normalising flat splits to fold format.

    Fold-based splits (standard_cv, loato) have a ``"folds"`` key.
    Flat splits (direct_indirect, crosslingual) have top-level
    ``"train_indices"`` / ``"test_indices"`` — these are wrapped into
    a single-element ``"folds"`` list so callers can iterate uniformly.
    """
    result: dict[str, Any] = json.load(open(split_path))  # noqa: SIM115

    if "folds" not in result and "train_indices" in result and "test_indices" in result:
        result["folds"] = [
            {
                "train_indices": result["train_indices"],
                "test_indices": result["test_indices"],
                "held_out_category": result.get("held_out_category"),
            }
        ]

    return result


def run_experiment(
    experiment: str,
    split_path: Path,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    classifier: Classifier,
    embedding_name: str,
    with_ci: bool = False,
    n_bootstrap: int = 10_000,
) -> ExperimentResult:
    """Run a fold-based experiment (LOATO or standard CV).

    Parameters
    ----------
    experiment : str
        Experiment name ("loato" or "standard_cv").
    split_path : Path
        Path to the split JSON file.
    embeddings : array of shape (n_samples, dim)
        Pre-computed embedding matrix for the full dataset.
    labels : array of shape (n_samples,)
        Binary labels for the full dataset.
    classifier : Classifier
        Classifier instance to train/evaluate.
    embedding_name : str
        Name of the embedding model (for result metadata).
    with_ci : bool
        Whether to compute bootstrap CIs.
    n_bootstrap : int
        Number of bootstrap resamples for CIs.

    Returns
    -------
    ExperimentResult
        Aggregated results across all folds.
    """
    split_data = _load_split(split_path)
    folds = split_data["folds"]
    fold_results: list[FoldResult] = []

    for i, fold in enumerate(folds):
        train_idx = np.array(fold["train_indices"])
        test_idx = np.array(fold["test_indices"])
        held_out = fold.get("held_out_category")

        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        logger.info(
            "Fold %d/%d (held_out=%s): train=%d, test=%d",
            i + 1,
            len(folds),
            held_out,
            len(train_idx),
            len(test_idx),
        )

        # Train
        classifier.fit(X_train, y_train)

        # Predict
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)

        # Metrics
        if with_ci:
            metrics = bootstrap_ci(y_test, y_pred, y_prob, n_bootstrap=n_bootstrap)
        else:
            metrics = compute_metrics(y_test, y_pred, y_prob)

        fold_results.append(
            FoldResult(
                fold_id=i,
                held_out_category=held_out,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
            )
        )

        logger.info(
            "  F1=%.4f, Acc=%.4f, AUC-ROC=%.4f",
            metrics.f1.value,
            metrics.accuracy.value,
            metrics.auc_roc.value,
        )

    # Aggregate
    f1_scores = [f.metrics.f1.value for f in fold_results]
    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))

    logger.info(
        "%s %s×%s: mean_F1=%.4f (±%.4f)",
        experiment,
        embedding_name,
        classifier.name,
        mean_f1,
        std_f1,
    )

    return ExperimentResult(
        experiment=experiment,
        embedding=embedding_name,
        classifier=classifier.name,
        folds=fold_results,
        mean_f1=mean_f1,
        std_f1=std_f1,
    )


def run_loato(
    split_path: Path,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    classifier: Classifier,
    embedding_name: str,
    with_ci: bool = False,
    n_bootstrap: int = 10_000,
) -> ExperimentResult:
    """Convenience wrapper for LOATO experiment."""
    return run_experiment(
        experiment="loato",
        split_path=split_path,
        embeddings=embeddings,
        labels=labels,
        classifier=classifier,
        embedding_name=embedding_name,
        with_ci=with_ci,
        n_bootstrap=n_bootstrap,
    )


def run_standard_cv(
    split_path: Path,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    classifier: Classifier,
    embedding_name: str,
    with_ci: bool = False,
    n_bootstrap: int = 10_000,
) -> ExperimentResult:
    """Convenience wrapper for standard CV experiment."""
    return run_experiment(
        experiment="standard_cv",
        split_path=split_path,
        embeddings=embeddings,
        labels=labels,
        classifier=classifier,
        embedding_name=embedding_name,
        with_ci=with_ci,
        n_bootstrap=n_bootstrap,
    )


def compute_generalization_gap(
    standard_cv: ExperimentResult,
    loato: ExperimentResult,
) -> float:
    """Compute ΔF1 = Standard_CV_F1 − LOATO_F1.

    A positive gap means the model performs worse on unseen attack types.
    """
    return standard_cv.mean_f1 - loato.mean_f1
