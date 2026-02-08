"""W&B experiment tracking utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import wandb
from numpy.typing import NDArray


def init_run(
    experiment: str,
    embedding: str,
    classifier: str,
    fold: int | str,
    config: dict[str, Any] | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with standard naming conventions.

    Run name: ``{experiment}_{embedding}_{classifier}_{fold}``
    Group:    ``{experiment}_{embedding}_{classifier}``
    """
    return wandb.init(
        project="loato-bench",
        name=f"{experiment}_{embedding}_{classifier}_{fold}",
        group=f"{experiment}_{embedding}_{classifier}",
        tags=[experiment, embedding, classifier],
        config=config,
    )


def log_metrics(
    run: wandb.sdk.wandb_run.Run,
    metrics: dict[str, float],
    prefix: str = "",
) -> None:
    """Log metrics to a W&B run, optionally prefixing all keys."""
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    run.log(metrics)


def log_confusion_matrix(
    run: wandb.sdk.wandb_run.Run,
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str],
) -> None:
    """Log a confusion matrix plot to W&B."""
    cm = wandb.plot.confusion_matrix(
        y_true=y_true.tolist(),
        preds=y_pred.tolist(),
        class_names=class_names,
    )
    run.log({"confusion_matrix": cm})


def finish_run(run: wandb.sdk.wandb_run.Run) -> None:
    """Finish a W&B run."""
    run.finish()
