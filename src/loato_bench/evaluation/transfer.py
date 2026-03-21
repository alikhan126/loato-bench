"""Transfer experiments: direct→indirect, cross-lingual.

Convenience wrappers around ``run_experiment()`` for flat-split
(single train/test pair) transfer evaluations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from loato_bench.classifiers.base import Classifier
from loato_bench.evaluation.loato import ExperimentResult, run_experiment


def run_direct_indirect(
    split_path: Path,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    classifier: Classifier,
    embedding_name: str,
    with_ci: bool = False,
    n_bootstrap: int = 10_000,
) -> ExperimentResult:
    """Run direct→indirect transfer experiment.

    Trains on direct injections + benign, tests on indirect injections + benign.
    The split file uses a flat ``train_indices`` / ``test_indices`` format
    (normalised to a single fold by ``run_experiment``).

    Parameters
    ----------
    split_path : Path
        Path to ``direct_indirect_split.json``.
    embeddings : array of shape (n_samples, dim)
        Pre-computed embedding matrix.
    labels : array of shape (n_samples,)
        Binary labels.
    classifier : Classifier
        Classifier instance.
    embedding_name : str
        Embedding model name for result metadata.
    with_ci : bool
        Whether to compute bootstrap CIs.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    ExperimentResult
        Single-fold result for the transfer experiment.
    """
    return run_experiment(
        experiment="direct_indirect",
        split_path=split_path,
        embeddings=embeddings,
        labels=labels,
        classifier=classifier,
        embedding_name=embedding_name,
        with_ci=with_ci,
        n_bootstrap=n_bootstrap,
    )


def compute_transfer_gap(
    standard_cv: ExperimentResult,
    transfer: ExperimentResult,
) -> float:
    """Compute ΔF1 = Standard_CV_F1 − Transfer_F1.

    A positive gap means the model generalises worse on the transfer task
    (e.g., indirect injections) than on the in-distribution CV baseline.
    """
    return standard_cv.mean_f1 - transfer.mean_f1
