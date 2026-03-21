"""Evaluation protocols: LOATO, standard CV, transfer experiments."""

from loato_bench.evaluation.loato import (
    ExperimentResult,
    FoldResult,
    compute_generalization_gap,
    run_experiment,
    run_loato,
    run_standard_cv,
)
from loato_bench.evaluation.metrics import (
    EvalMetrics,
    MetricResult,
    bootstrap_ci,
    compute_metrics,
)
from loato_bench.evaluation.transfer import (
    compute_transfer_gap,
    run_direct_indirect,
)

__all__ = [
    "EvalMetrics",
    "ExperimentResult",
    "FoldResult",
    "MetricResult",
    "bootstrap_ci",
    "compute_generalization_gap",
    "compute_metrics",
    "compute_transfer_gap",
    "run_direct_indirect",
    "run_experiment",
    "run_loato",
    "run_standard_cv",
]
