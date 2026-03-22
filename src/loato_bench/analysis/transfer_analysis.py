"""Transfer threshold analysis: oracle F1, Platt scaling, score distributions.

Quantifies the AUC-ROC vs F1 disconnect in Direct→Indirect transfer results.
Re-trains classifiers to extract per-sample predicted probabilities, then:
- Sweeps thresholds to find oracle F1 (best possible with labeled indirect data)
- Tests Platt scaling on a 10% holdout of indirect samples
- Generates PR/ROC curves and score distribution overlays
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from loato_bench.analysis.visualization import managed_figure, safe_output_path
from loato_bench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Threshold analysis result for one embedding × classifier combo."""

    embedding: str
    classifier: str
    # Default threshold (0.5)
    uncalibrated_f1: float
    auc_roc: float
    auc_pr: float
    # Oracle: best possible threshold on indirect test set
    oracle_f1: float
    oracle_threshold: float
    # Platt scaling (10% calibration holdout)
    calibrated_f1: float
    # Score statistics
    train_mean_score: float
    test_mean_score: float
    score_shift: float  # test_mean - train_mean (injection class only)
    # Raw curves for plotting
    pr_curve: tuple[NDArray, NDArray, NDArray] = field(
        repr=False,
        default=(np.array([]), np.array([]), np.array([])),
    )
    roc_curve: tuple[NDArray, NDArray, NDArray] = field(
        repr=False,
        default=(np.array([]), np.array([]), np.array([])),
    )
    train_scores: NDArray = field(repr=False, default_factory=lambda: np.array([]))
    test_scores: NDArray = field(repr=False, default_factory=lambda: np.array([]))
    train_labels: NDArray = field(repr=False, default_factory=lambda: np.array([]))
    test_labels: NDArray = field(repr=False, default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def find_oracle_threshold(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """Sweep thresholds to find the one maximizing macro F1.

    Parameters
    ----------
    y_true : array of shape (n,)
        Ground truth binary labels.
    y_prob : array of shape (n,)
        Predicted probability of positive class.
    n_thresholds : int
        Number of threshold values to evaluate.

    Returns
    -------
    tuple[float, float]
        (best_f1, best_threshold)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = 0.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return best_f1, best_thresh


def run_platt_scaling(
    classifier: Classifier,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    cal_fraction: float = 0.10,
    seed: int = 42,
) -> float:
    """Fit Platt scaling on a holdout of the test set, evaluate on the rest.

    Simulates having a small sample of labeled indirect injections for
    threshold recalibration.

    Parameters
    ----------
    classifier : Classifier
        Already-fitted classifier (must have sklearn pipeline accessible).
    X_train : array
        Training data (used to fit the base classifier).
    y_train : array
        Training labels.
    X_test : array
        Full test data (indirect + benign).
    y_test : array
        Full test labels.
    cal_fraction : float
        Fraction of test set used for calibration (default 10%).
    seed : int
        Random seed for the calibration/evaluation split.

    Returns
    -------
    float
        Calibrated macro F1 on the evaluation portion.
    """
    rng = np.random.RandomState(seed)
    n_test = len(y_test)
    n_cal = max(int(n_test * cal_fraction), 10)

    # Stratified split of test set into calibration + evaluation
    idx = np.arange(n_test)
    rng.shuffle(idx)
    cal_idx = idx[:n_cal]
    eval_idx = idx[n_cal:]

    X_cal, y_cal = X_test[cal_idx], y_test[cal_idx]
    X_eval, y_eval = X_test[eval_idx], y_test[eval_idx]

    # Use oracle threshold on calibration set as a simple recalibration
    # (more practical than full Platt scaling which needs the base estimator)
    y_prob_cal = classifier.predict_proba(X_cal)[:, 1]
    _, optimal_thresh = find_oracle_threshold(y_cal, y_prob_cal, n_thresholds=200)

    # Apply recalibrated threshold to evaluation set
    y_prob_eval = classifier.predict_proba(X_eval)[:, 1]
    y_pred_eval = (y_prob_eval >= optimal_thresh).astype(np.int64)
    calibrated_f1 = float(f1_score(y_eval, y_pred_eval, average="macro", zero_division=0.0))

    return calibrated_f1


def analyze_single_combo(
    embedding_name: str,
    classifier_name: str,
    classifier: Classifier,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> ThresholdResult:
    """Run full threshold analysis for one embedding × classifier combo.

    Parameters
    ----------
    embedding_name : str
        Embedding model name.
    classifier_name : str
        Classifier name.
    classifier : Classifier
        Fresh classifier instance (will be fitted).
    embeddings : array of shape (n_samples, dim)
        Full embedding matrix.
    labels : array of shape (n_samples,)
        Binary labels.
    train_indices : array
        Training set indices (direct injections + benign).
    test_indices : array
        Test set indices (indirect injections + benign).

    Returns
    -------
    ThresholdResult
        Complete threshold analysis results.
    """
    X_train = embeddings[train_indices]
    y_train = labels[train_indices]
    X_test = embeddings[test_indices]
    y_test = labels[test_indices]

    # Train
    classifier.fit(X_train, y_train)

    # Extract predictions
    y_prob_test = classifier.predict_proba(X_test)
    y_prob_train = classifier.predict_proba(X_train)
    prob_test = y_prob_test[:, 1]  # P(injection)
    prob_train = y_prob_train[:, 1]

    # Default threshold F1
    y_pred_default = (prob_test >= 0.5).astype(np.int64)
    uncalibrated_f1 = float(f1_score(y_test, y_pred_default, average="macro", zero_division=0.0))

    # AUC metrics
    auc_roc = float(roc_auc_score(y_test, prob_test))
    auc_pr = float(average_precision_score(y_test, prob_test))

    # Oracle threshold
    oracle_f1, oracle_threshold = find_oracle_threshold(y_test, prob_test)

    # Platt scaling (10% holdout)
    calibrated_f1 = run_platt_scaling(
        classifier, X_train, y_train, X_test, y_test, cal_fraction=0.10
    )

    # Score statistics (injection class only)
    train_inj_scores = prob_train[y_train == 1]
    test_inj_scores = prob_test[y_test == 1]
    train_mean = float(np.mean(train_inj_scores)) if len(train_inj_scores) > 0 else 0.0
    test_mean = float(np.mean(test_inj_scores)) if len(test_inj_scores) > 0 else 0.0

    # PR and ROC curves
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, prob_test)
    fpr, tpr, roc_thresholds = roc_curve(y_test, prob_test)

    logger.info(
        "%s × %s: uncal_F1=%.3f, oracle_F1=%.3f (t=%.3f), cal_F1=%.3f, AUC=%.3f",
        embedding_name,
        classifier_name,
        uncalibrated_f1,
        oracle_f1,
        oracle_threshold,
        calibrated_f1,
        auc_roc,
    )

    return ThresholdResult(
        embedding=embedding_name,
        classifier=classifier_name,
        uncalibrated_f1=uncalibrated_f1,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        oracle_f1=oracle_f1,
        oracle_threshold=oracle_threshold,
        calibrated_f1=calibrated_f1,
        train_mean_score=train_mean,
        test_mean_score=test_mean,
        score_shift=test_mean - train_mean,
        pr_curve=(pr_precision, pr_recall, pr_thresholds),
        roc_curve=(fpr, tpr, roc_thresholds),
        train_scores=prob_train,
        test_scores=prob_test,
        train_labels=y_train,
        test_labels=y_test,
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

EMBEDDING_DISPLAY = {
    "minilm": "MiniLM (384d)",
    "bge_large": "BGE-Large (1024d)",
    "instructor": "Instructor (768d)",
    "openai_small": "OpenAI-Small (1536d)",
    "e5_mistral": "E5-Mistral (4096d)",
}

CLASSIFIER_DISPLAY = {
    "logreg": "LogReg",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}

EMBEDDING_ORDER = ["minilm", "bge_large", "instructor", "openai_small", "e5_mistral"]
CLASSIFIER_ORDER = ["logreg", "svm", "xgboost", "mlp"]
CLASSIFIER_COLORS = {"logreg": "#1f77b4", "svm": "#ff7f0e", "xgboost": "#2ca02c", "mlp": "#d62728"}


def plot_roc_pr_grid(
    results: list[ThresholdResult],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot 5×2 grid: ROC + PR curves per embedding (4 classifier lines each).

    Parameters
    ----------
    results : list[ThresholdResult]
        All threshold analysis results.
    output_path : Path
        Where to save the figure.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    # Group by embedding
    by_emb: dict[str, list[ThresholdResult]] = {}
    for r in results:
        by_emb.setdefault(r.embedding, []).append(r)

    n_emb = len(by_emb)
    with managed_figure(figsize=(14, 3 * n_emb), dpi=dpi) as (fig, _):
        # Remove default axes, create grid
        fig.clf()
        axes = fig.subplots(n_emb, 2, squeeze=False)

        for row, emb in enumerate(EMBEDDING_ORDER):
            if emb not in by_emb:
                continue
            emb_results = by_emb[emb]

            ax_roc = axes[row][0]
            ax_pr = axes[row][1]

            for r in sorted(emb_results, key=lambda x: CLASSIFIER_ORDER.index(x.classifier)):
                color = CLASSIFIER_COLORS.get(r.classifier, "gray")
                clf_disp = CLASSIFIER_DISPLAY.get(r.classifier, r.classifier)
                label = f"{clf_disp} (AUC={r.auc_roc:.3f})"

                fpr, tpr, _ = r.roc_curve
                ax_roc.plot(fpr, tpr, color=color, label=label, linewidth=1.5)

                pr_prec, pr_rec, _ = r.pr_curve
                label_pr = f"{clf_disp} (AP={r.auc_pr:.3f})"
                ax_pr.plot(pr_rec, pr_prec, color=color, label=label_pr, linewidth=1.5)

            # ROC formatting
            ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
            ax_roc.set_xlim([0, 1])
            ax_roc.set_ylim([0, 1.02])
            ax_roc.set_xlabel("FPR", fontsize=9)
            ax_roc.set_ylabel("TPR", fontsize=9)
            emb_disp = EMBEDDING_DISPLAY.get(emb, emb)
            ax_roc.set_title(
                f"{emb_disp} — ROC",
                fontsize=10,
                fontweight="bold",
            )
            ax_roc.legend(fontsize=7, loc="lower right")

            # PR formatting
            ax_pr.set_xlim([0, 1])
            ax_pr.set_ylim([0, 1.02])
            ax_pr.set_xlabel("Recall", fontsize=9)
            ax_pr.set_ylabel("Precision", fontsize=9)
            ax_pr.set_title(
                f"{emb_disp} — PR",
                fontsize=10,
                fontweight="bold",
            )
            ax_pr.legend(fontsize=7, loc="lower left")

        fig.suptitle(
            "Direct→Indirect Transfer: ROC & PR Curves",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved ROC/PR grid to %s", output_path)


def plot_score_distributions(
    results: list[ThresholdResult],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot score distribution overlay for select combos showing distributional shift.

    Shows P(injection) scores for injection-class samples in train (direct) vs
    test (indirect) to visualize the threshold miscalibration.

    Parameters
    ----------
    results : list[ThresholdResult]
        All threshold analysis results.
    output_path : Path
        Where to save the figure.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    # Pick one classifier per embedding (MLP — best non-SVM performer)
    combos = []
    for emb in EMBEDDING_ORDER:
        for r in results:
            if r.embedding == emb and r.classifier == "mlp":
                combos.append(r)
                break
        else:
            # Fallback to first available
            for r in results:
                if r.embedding == emb:
                    combos.append(r)
                    break

    n = len(combos)
    with managed_figure(figsize=(14, 3 * n), dpi=dpi) as (fig, _):
        fig.clf()
        axes = fig.subplots(n, 1, squeeze=False)

        for i, r in enumerate(combos):
            ax = axes[i][0]

            # Injection-class scores only
            train_inj = r.train_scores[r.train_labels == 1]
            test_inj = r.test_scores[r.test_labels == 1]

            bins = np.linspace(0, 1, 50)
            ax.hist(
                train_inj,
                bins=bins,
                alpha=0.5,
                label="Direct (train)",
                color="#2ecc71",
                density=True,
            )
            ax.hist(
                test_inj,
                bins=bins,
                alpha=0.5,
                label="Indirect (test)",
                color="#e74c3c",
                density=True,
            )

            # Default threshold line
            ax.axvline(
                x=0.5,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Default threshold (0.5)",
            )
            # Oracle threshold
            ax.axvline(
                x=r.oracle_threshold,
                color="blue",
                linestyle="-.",
                linewidth=1.5,
                label=f"Oracle threshold ({r.oracle_threshold:.2f})",
            )

            ax.set_xlabel("P(injection)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            emb_display = EMBEDDING_DISPLAY.get(r.embedding, r.embedding)
            clf_display = CLASSIFIER_DISPLAY.get(r.classifier, r.classifier)
            ax.set_title(
                f"{emb_display} × {clf_display} — Score shift: {r.score_shift:+.3f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.legend(fontsize=8)

        fig.suptitle(
            "Distributional Shift: P(injection) for Direct vs Indirect Injections",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved score distributions to %s", output_path)


def plot_threshold_summary(
    results: list[ThresholdResult],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Bar chart comparing uncalibrated, oracle, and calibrated F1.

    Parameters
    ----------
    results : list[ThresholdResult]
        All threshold analysis results.
    output_path : Path
        Where to save the figure.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    # Sort by embedding order then classifier order
    sorted_results = sorted(
        results,
        key=lambda r: (EMBEDDING_ORDER.index(r.embedding), CLASSIFIER_ORDER.index(r.classifier)),
    )

    labels = [
        f"{r.embedding}\n{CLASSIFIER_DISPLAY.get(r.classifier, r.classifier)}"
        for r in sorted_results
    ]
    uncal = [r.uncalibrated_f1 for r in sorted_results]
    oracle = [r.oracle_f1 for r in sorted_results]
    cal = [r.calibrated_f1 for r in sorted_results]

    x = np.arange(len(labels))
    width = 0.25

    with managed_figure(figsize=(16, 6), dpi=dpi) as (fig, ax):
        ax.bar(
            x - width,
            uncal,
            width,
            label="Uncalibrated (t=0.5)",
            color="#e74c3c",
            alpha=0.8,
        )
        ax.bar(
            x,
            cal,
            width,
            label="Calibrated (10% holdout)",
            color="#f39c12",
            alpha=0.8,
        )
        ax.bar(
            x + width,
            oracle,
            width,
            label="Oracle (best threshold)",
            color="#2ecc71",
            alpha=0.8,
        )

        ax.set_xlabel("")
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_title(
            "Direct→Indirect Transfer: Threshold Recalibration Impact",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved threshold summary to %s", output_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def build_summary_table(results: list[ThresholdResult]) -> pd.DataFrame:
    """Build summary DataFrame from threshold analysis results.

    Parameters
    ----------
    results : list[ThresholdResult]
        All threshold analysis results.

    Returns
    -------
    pd.DataFrame
        Summary table with columns for each metric.
    """
    rows = []
    for r in sorted(
        results,
        key=lambda x: (EMBEDDING_ORDER.index(x.embedding), CLASSIFIER_ORDER.index(x.classifier)),
    ):
        rows.append(
            {
                "Embedding": EMBEDDING_DISPLAY.get(r.embedding, r.embedding),
                "Classifier": CLASSIFIER_DISPLAY.get(r.classifier, r.classifier),
                "Uncalibrated_F1": r.uncalibrated_f1,
                "Oracle_F1": r.oracle_f1,
                "Oracle_Threshold": r.oracle_threshold,
                "Calibrated_F1": r.calibrated_f1,
                "AUC_ROC": r.auc_roc,
                "AUC_PR": r.auc_pr,
                "Score_Shift": r.score_shift,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_transfer_threshold_analysis(
    embeddings_loader: Callable[[str], NDArray[np.float32] | None],
    labels: NDArray[np.int64],
    split_path: Path,
    classifier_factories: dict[str, Callable[[], Classifier]],
    output_dir: Path,
    dpi: int = 150,
) -> dict[str, Path]:
    """Run the full transfer threshold analysis pipeline.

    Parameters
    ----------
    embeddings_loader : callable
        Function that takes an embedding name and returns the embedding matrix
        (or None if not available).
    labels : array of shape (n_samples,)
        Binary labels for the full dataset.
    split_path : Path
        Path to ``direct_indirect_split.json``.
    classifier_factories : dict
        Mapping of classifier name to factory function.
    output_dir : Path
        Where to save all outputs.
    dpi : int
        DPI for figures.

    Returns
    -------
    dict[str, Path]
        Mapping of output name to file path.
    """
    import json as json_mod

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    with open(split_path) as f:
        split_data = json_mod.load(f)

    train_indices = np.array(split_data["train_indices"])
    test_indices = np.array(split_data["test_indices"])

    logger.info("Split: %d train, %d test", len(train_indices), len(test_indices))

    all_results: list[ThresholdResult] = []

    for emb_name in EMBEDDING_ORDER:
        embeddings = embeddings_loader(emb_name)
        if embeddings is None:
            logger.warning("No embeddings for %s, skipping", emb_name)
            continue

        for clf_name in CLASSIFIER_ORDER:
            if clf_name not in classifier_factories:
                continue

            logger.info("Analyzing %s × %s", emb_name, clf_name)
            clf = classifier_factories[clf_name]()

            result = analyze_single_combo(
                embedding_name=emb_name,
                classifier_name=clf_name,
                classifier=clf,
                embeddings=embeddings,
                labels=labels,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            all_results.append(result)

    if not all_results:
        logger.warning("No results produced")
        return {}

    outputs: dict[str, Path] = {}

    # Summary table
    summary_df = build_summary_table(all_results)
    table_path = output_dir / "transfer_threshold_analysis.json"
    table_data = {
        "results": summary_df.to_dict(orient="records"),
        "metadata": {
            "n_combos": len(all_results),
            "cal_fraction": 0.10,
            "n_thresholds": 200,
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
        },
    }
    with open(table_path, "w") as f:
        json_mod.dump(table_data, f, indent=2)
    outputs["summary_json"] = table_path

    # Markdown table
    md_path = output_dir / "transfer_threshold_table.md"
    md_lines = ["## Transfer Threshold Analysis\n"]
    fmt_df = summary_df.copy()
    for col in fmt_df.select_dtypes(include=[np.floating]).columns:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")
    md_lines.append(fmt_df.to_markdown(index=False))
    md_path.write_text("\n".join(md_lines))
    outputs["summary_md"] = md_path

    # Figures
    for ext in ("png", "pdf"):
        # ROC/PR grid
        roc_pr_path = figures_dir / f"transfer_roc_pr_grid.{ext}"
        plot_roc_pr_grid(all_results, roc_pr_path, dpi=dpi)
        outputs[f"roc_pr_{ext}"] = roc_pr_path

        # Score distributions
        dist_path = figures_dir / f"transfer_score_distributions.{ext}"
        plot_score_distributions(all_results, dist_path, dpi=dpi)
        outputs[f"score_dist_{ext}"] = dist_path

        # Threshold summary bar chart
        thresh_path = figures_dir / f"transfer_threshold_summary.{ext}"
        plot_threshold_summary(all_results, thresh_path, dpi=dpi)
        outputs[f"threshold_summary_{ext}"] = thresh_path

    logger.info("Transfer threshold analysis complete. %d outputs.", len(outputs))
    return outputs
