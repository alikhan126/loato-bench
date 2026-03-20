"""Metric computation: accuracy, precision, recall, F1, AUC-ROC, AUC-PR.

Includes bootstrap confidence intervals for all metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    """A single metric with optional bootstrap confidence interval.

    Attributes
    ----------
    value : float
        Point estimate of the metric.
    ci_lower : float | None
        Lower bound of the 95% bootstrap CI.
    ci_upper : float | None
        Upper bound of the 95% bootstrap CI.
    """

    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to a JSON-serializable dict."""
        return {
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


@dataclass
class EvalMetrics:
    """Full evaluation metrics for a single run.

    Attributes
    ----------
    accuracy : MetricResult
    precision : MetricResult
    recall : MetricResult
    f1 : MetricResult
        Macro F1 score.
    auc_roc : MetricResult
    auc_pr : MetricResult
        Average precision (area under precision-recall curve).
    """

    accuracy: MetricResult
    precision: MetricResult
    recall: MetricResult
    f1: MetricResult
    auc_roc: MetricResult
    auc_pr: MetricResult

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        """Convert all metrics to a nested dict."""
        return {
            "accuracy": self.accuracy.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1": self.f1.to_dict(),
            "auc_roc": self.auc_roc.to_dict(),
            "auc_pr": self.auc_pr.to_dict(),
        }

    def summary(self) -> dict[str, float]:
        """Flat dict of point estimates (for W&B logging)."""
        return {
            "accuracy": self.accuracy.value,
            "precision": self.precision.value,
            "recall": self.recall.value,
            "f1": self.f1.value,
            "auc_roc": self.auc_roc.value,
            "auc_pr": self.auc_pr.value,
        }


def compute_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_prob: NDArray[np.float64] | None = None,
) -> EvalMetrics:
    """Compute all evaluation metrics from labels and predictions.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        Ground truth labels (0 or 1).
    y_pred : array of shape (n_samples,)
        Predicted labels (0 or 1).
    y_prob : array of shape (n_samples, 2), optional
        Predicted probabilities. Column 1 = P(injection).
        Required for AUC-ROC and AUC-PR; set to 0.0 if not provided.

    Returns
    -------
    EvalMetrics
        All metrics as point estimates (no CIs).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

    if y_prob is not None and len(np.unique(y_true)) > 1:
        prob_pos = y_prob[:, 1]
        auc_roc = float(roc_auc_score(y_true, prob_pos))
        auc_pr = float(average_precision_score(y_true, prob_pos))
    else:
        auc_roc = 0.0
        auc_pr = 0.0

    return EvalMetrics(
        accuracy=MetricResult(acc),
        precision=MetricResult(prec),
        recall=MetricResult(rec),
        f1=MetricResult(f1),
        auc_roc=MetricResult(auc_roc),
        auc_pr=MetricResult(auc_pr),
    )


def bootstrap_ci(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_prob: NDArray[np.float64] | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> EvalMetrics:
    """Compute metrics with bootstrap 95% confidence intervals.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        Ground truth labels.
    y_pred : array of shape (n_samples,)
        Predicted labels.
    y_prob : array of shape (n_samples, 2), optional
        Predicted probabilities.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    EvalMetrics
        All metrics with bootstrap CIs.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - confidence) / 2

    # Point estimates
    point = compute_metrics(y_true, y_pred, y_prob)

    # Bootstrap
    boot_metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
        "auc_pr": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        yprob = y_prob[idx] if y_prob is not None else None

        m = compute_metrics(yt, yp, yprob)
        for key in boot_metrics:
            boot_metrics[key].append(getattr(m, key).value)

    def _ci(key: str) -> MetricResult:
        vals = np.array(boot_metrics[key])
        return MetricResult(
            value=getattr(point, key).value,
            ci_lower=float(np.percentile(vals, alpha * 100)),
            ci_upper=float(np.percentile(vals, (1 - alpha) * 100)),
        )

    return EvalMetrics(
        accuracy=_ci("accuracy"),
        precision=_ci("precision"),
        recall=_ci("recall"),
        f1=_ci("f1"),
        auc_roc=_ci("auc_roc"),
        auc_pr=_ci("auc_pr"),
    )
