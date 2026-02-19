"""W&B experiment tracking utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import wandb


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
    y_true: np.typing.NDArray[np.int64],
    y_pred: np.typing.NDArray[np.int64],
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


# ---------------------------------------------------------------------------
# EDA-specific logging
# ---------------------------------------------------------------------------


def log_eda_artifacts(
    run: wandb.sdk.wandb_run.Run,
    stats: dict[str, Any],
    figures: dict[str, Path] | None = None,
    reports: dict[str, Path] | None = None,
) -> None:
    """Log EDA summary, figures, and reports to W&B.

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    stats : dict[str, Any]
        EDA statistics (dataset stats, text properties, etc.).
    figures : dict[str, Path], optional
        Dictionary mapping figure names to file paths.
    reports : dict[str, Path], optional
        Dictionary mapping report names to file paths (JSON, CSV).
    """
    # Log summary statistics
    for key, value in stats.items():
        if isinstance(value, dict):
            # Flatten nested dicts
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, int | float | str | bool):
                    run.summary[f"{key}/{sub_key}"] = sub_value
        elif isinstance(value, int | float | str | bool):
            run.summary[key] = value

    # Log figures as artifacts
    if figures:
        for fig_name, fig_path in figures.items():
            if fig_path.exists():
                run.log({fig_name: wandb.Image(str(fig_path))})

    # Log reports as artifacts
    if reports:
        artifact = wandb.Artifact(name="eda-reports", type="dataset")
        for report_name, report_path in reports.items():
            if report_path.exists():
                artifact.add_file(str(report_path), name=report_name)
        run.log_artifact(artifact)
