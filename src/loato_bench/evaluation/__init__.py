"""Evaluation protocols: LOATO, standard CV, transfer experiments, LLM baseline."""

from loato_bench.evaluation.llm_baseline import (
    CostTracker,
    LLMBaselineResult,
    draw_stratified_sample,
    parse_baseline_response,
    run_llm_baseline,
)
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
    "CostTracker",
    "EvalMetrics",
    "ExperimentResult",
    "FoldResult",
    "LLMBaselineResult",
    "MetricResult",
    "bootstrap_ci",
    "compute_generalization_gap",
    "compute_metrics",
    "compute_transfer_gap",
    "draw_stratified_sample",
    "parse_baseline_response",
    "run_direct_indirect",
    "run_experiment",
    "run_llm_baseline",
    "run_loato",
    "run_standard_cv",
]
