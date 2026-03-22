"""Cost-performance analysis: classifier vs GPT-4o regime map and layered defense model.

Builds the quantitative backing for §6.5 deployment recommendations:
- Side-by-side F1 comparison (Standard CV + Direct→Indirect)
- Cost-per-prediction for classifier vs GPT-4o
- Regime map: F1 vs attack novelty with crossover
- Layered defense cost model as a function of escalation rate
- GPT-4o precision-recall interpretation
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from loato_bench.analysis.report import (
    CLASSIFIER_DISPLAY,
    CLASSIFIER_ORDER,
    EMBEDDING_DISPLAY,
    EMBEDDING_ORDER,
)
from loato_bench.analysis.visualization import managed_figure, safe_output_path

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    """One row of the side-by-side comparison table."""

    test_pool: str
    system: str
    embedding: str
    classifier: str
    f1: float
    precision: float
    recall: float
    auc_roc: float
    cost_per_query: float


@dataclass
class LayeredDefensePoint:
    """One point on the layered defense cost curve."""

    escalation_rate: float
    total_cost_per_query: float
    estimated_f1: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPT-4o cost from LLM baseline experiment (March 2026 pricing)
GPT4O_COST_PER_QUERY = 0.0008  # USD

# Classifier cost is effectively zero at inference (embedding + sklearn predict).
# Training cost amortized over millions of queries is negligible.
# We use a small number to make log-scale plots work.
CLASSIFIER_COST_PER_QUERY = 0.000001  # USD (~$0 after training)

# For cost ratio display
COST_RATIO = GPT4O_COST_PER_QUERY / CLASSIFIER_COST_PER_QUERY


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_classifier_results(
    results_dir: Path,
) -> pd.DataFrame:
    """Load all classifier experiment results into a DataFrame.

    Parameters
    ----------
    results_dir : Path
        Directory containing experiment result JSONs.

    Returns
    -------
    pd.DataFrame
        Columns: experiment, embedding, classifier, f1, precision, recall, auc_roc.
    """
    rows: list[dict[str, object]] = []

    for experiment in ("standard_cv", "direct_indirect"):
        for emb in EMBEDDING_ORDER:
            for clf in CLASSIFIER_ORDER:
                fname = f"{experiment}_{emb}_{clf}.json"
                fpath = results_dir / fname
                if not fpath.exists():
                    logger.warning("Missing result file: %s", fpath)
                    continue

                with open(fpath) as f:
                    data = json.load(f)

                folds = data.get("folds", [])
                if not folds:
                    continue

                n = len(folds)
                mean_f1 = data["mean_f1"]
                mean_prec = sum(fold["metrics"]["precision"]["value"] for fold in folds) / n
                mean_rec = sum(fold["metrics"]["recall"]["value"] for fold in folds) / n
                mean_auc = sum(fold["metrics"]["auc_roc"]["value"] for fold in folds) / n

                rows.append(
                    {
                        "experiment": experiment,
                        "embedding": emb,
                        "classifier": clf,
                        "f1": mean_f1,
                        "precision": mean_prec,
                        "recall": mean_rec,
                        "auc_roc": mean_auc,
                    }
                )

    return pd.DataFrame(rows)


def load_llm_results(llm_results_path: Path) -> pd.DataFrame:
    """Load GPT-4o baseline results.

    Parameters
    ----------
    llm_results_path : Path
        Path to llm_baseline_gpt_4o_all.json.

    Returns
    -------
    pd.DataFrame
        Columns: test_pool, f1, precision, recall, auc_roc, cost_usd, n_samples.
    """
    with open(llm_results_path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        rows.append(
            {
                "test_pool": entry["test_pool"],
                "f1": entry["metrics"]["f1"]["value"],
                "precision": entry["metrics"]["precision"]["value"],
                "recall": entry["metrics"]["recall"]["value"],
                "auc_roc": entry["metrics"]["auc_roc"]["value"],
                "cost_usd": entry["cost"]["estimated_cost_usd"],
                "n_samples": entry["n_samples"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def build_comparison_table(
    clf_df: pd.DataFrame,
    llm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build side-by-side comparison of best classifier vs GPT-4o.

    Parameters
    ----------
    clf_df : pd.DataFrame
        Classifier results from load_classifier_results().
    llm_df : pd.DataFrame
        LLM results from load_llm_results().

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: test_pool, system, f1, precision, recall,
        auc_roc, cost_per_query, winner, f1_gap.
    """
    pool_map = {"standard_cv": "Standard CV", "direct_indirect": "Direct→Indirect"}
    rows: list[dict[str, object]] = []

    for experiment, display_name in pool_map.items():
        exp_df = clf_df[clf_df["experiment"] == experiment]

        # Best overall classifier
        best_clf = exp_df.loc[exp_df["f1"].idxmax()]

        # GPT-4o
        llm_pool = "standard_cv" if experiment == "standard_cv" else "direct_indirect"
        llm_row = llm_df[llm_df["test_pool"] == llm_pool].iloc[0]

        # Add best classifier row
        emb_disp = EMBEDDING_DISPLAY.get(str(best_clf["embedding"]), str(best_clf["embedding"]))
        clf_disp = CLASSIFIER_DISPLAY.get(str(best_clf["classifier"]), str(best_clf["classifier"]))
        rows.append(
            {
                "test_pool": display_name,
                "system": f"Best Classifier ({emb_disp} × {clf_disp})",
                "f1": float(best_clf["f1"]),
                "precision": float(best_clf["precision"]),
                "recall": float(best_clf["recall"]),
                "auc_roc": float(best_clf["auc_roc"]),
                "cost_per_query": CLASSIFIER_COST_PER_QUERY,
            }
        )

        # Add GPT-4o row
        rows.append(
            {
                "test_pool": display_name,
                "system": "GPT-4o (zero-shot)",
                "f1": float(llm_row["f1"]),
                "precision": float(llm_row["precision"]),
                "recall": float(llm_row["recall"]),
                "auc_roc": float(llm_row["auc_roc"]),
                "cost_per_query": GPT4O_COST_PER_QUERY,
            }
        )

    df = pd.DataFrame(rows)

    # Add winner and gap columns
    winners: list[str] = []
    gaps: list[float] = []
    for pool in pool_map.values():
        pool_rows = df[df["test_pool"] == pool]
        f1_vals = pool_rows["f1"].tolist()
        if len(f1_vals) == 2:
            gap = f1_vals[0] - f1_vals[1]  # classifier - GPT-4o
            winners.append("Classifier" if gap > 0 else "GPT-4o")
            gaps.append(abs(gap))
            winners.append("Classifier" if gap > 0 else "GPT-4o")
            gaps.append(abs(gap))
        else:
            winners.extend(["—"] * len(pool_rows))
            gaps.extend([0.0] * len(pool_rows))

    df["winner"] = winners
    df["f1_gap"] = gaps

    return df


# ---------------------------------------------------------------------------
# Regime map
# ---------------------------------------------------------------------------


def build_regime_data(
    clf_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    threshold_results_path: Path | None = None,
) -> pd.DataFrame:
    """Build data for the regime map visualization.

    Parameters
    ----------
    clf_df : pd.DataFrame
        Classifier results.
    llm_df : pd.DataFrame
        LLM results.
    threshold_results_path : Path, optional
        Path to transfer_threshold_analysis.json for calibrated F1.

    Returns
    -------
    pd.DataFrame
        Columns: regime, classifier_f1, gpt4o_f1, calibrated_f1 (optional).
    """
    regimes: list[dict[str, object]] = []

    # Standard CV (known attacks)
    scv = clf_df[clf_df["experiment"] == "standard_cv"]
    best_scv = float(scv["f1"].max())
    llm_scv = float(llm_df[llm_df["test_pool"] == "standard_cv"]["f1"].iloc[0])

    regimes.append(
        {
            "regime": "Known Attacks\n(Standard CV)",
            "regime_order": 0,
            "classifier_f1": best_scv,
            "gpt4o_f1": llm_scv,
        }
    )

    # Direct→Indirect uncalibrated (novel attacks)
    di = clf_df[clf_df["experiment"] == "direct_indirect"]
    best_di = float(di["f1"].max())
    llm_di = float(llm_df[llm_df["test_pool"] == "direct_indirect"]["f1"].iloc[0])

    regimes.append(
        {
            "regime": "Novel Attacks\n(Direct→Indirect)",
            "regime_order": 1,
            "classifier_f1": best_di,
            "gpt4o_f1": llm_di,
        }
    )

    # Add calibrated point if threshold results available
    if threshold_results_path and threshold_results_path.exists():
        with open(threshold_results_path) as f:
            threshold_data = json.load(f)

        calibrated_f1s = [r["calibrated_f1"] for r in threshold_data if "calibrated_f1" in r]
        if calibrated_f1s:
            best_calibrated = max(calibrated_f1s)
            mean_calibrated = sum(calibrated_f1s) / len(calibrated_f1s)

            regimes.append(
                {
                    "regime": "Novel + Calibrated\n(10% holdout)",
                    "regime_order": 2,
                    "classifier_f1": best_calibrated,
                    "gpt4o_f1": llm_di,
                    "note": f"mean calibrated={mean_calibrated:.2f}",
                }
            )

    return pd.DataFrame(regimes).sort_values("regime_order")


def plot_regime_map(
    regime_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Plot the cost-performance regime map.

    X-axis: attack novelty (known → novel → novel+calibrated).
    Y-axis: F1 score.
    Two lines: classifier, GPT-4o.

    Parameters
    ----------
    regime_df : pd.DataFrame
        From build_regime_data().
    output_dir : Path
        Directory for output figures.
    dpi : int
        Figure DPI.

    Returns
    -------
    Path
        Path to saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with managed_figure(figsize=(10, 6), dpi=dpi) as (fig, ax):
        x = range(len(regime_df))
        regimes = regime_df["regime"].tolist()

        # Classifier line
        clf_f1s = regime_df["classifier_f1"].tolist()
        ax.plot(
            x,
            clf_f1s,
            "o-",
            color="#2196F3",
            linewidth=2.5,
            markersize=10,
            label="Best Embedding Classifier",
            zorder=3,
        )

        # GPT-4o line
        gpt_f1s = regime_df["gpt4o_f1"].tolist()
        ax.plot(
            x,
            gpt_f1s,
            "s--",
            color="#FF5722",
            linewidth=2.5,
            markersize=10,
            label="GPT-4o (zero-shot)",
            zorder=3,
        )

        # Annotate F1 values
        for i, (cf1, gf1) in enumerate(zip(clf_f1s, gpt_f1s)):
            ax.annotate(
                f"{cf1:.2f}",
                (i, cf1),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=10,
                color="#2196F3",
                fontweight="bold",
            )
            ax.annotate(
                f"{gf1:.2f}",
                (i, gf1),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=10,
                color="#FF5722",
                fontweight="bold",
            )

        # Shade the "GPT-4o wins" region
        if len(clf_f1s) >= 2 and len(gpt_f1s) >= 2:
            # Find where lines cross (between known and novel)
            ax.fill_between(
                x,
                clf_f1s,
                gpt_f1s,
                where=[cf > gf for cf, gf in zip(clf_f1s, gpt_f1s)],
                alpha=0.1,
                color="#2196F3",
                label="Classifier advantage",
            )
            ax.fill_between(
                x,
                clf_f1s,
                gpt_f1s,
                where=[cf <= gf for cf, gf in zip(clf_f1s, gpt_f1s)],
                alpha=0.1,
                color="#FF5722",
                label="GPT-4o advantage",
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(regimes, fontsize=11)
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_xlabel("Attack Novelty →", fontsize=12)
        ax.set_title(
            "Cost-Performance Regime Map: Classifier vs GPT-4o",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add cost annotation
        ax.text(
            0.98,
            0.02,
            f"Classifier: ~$0/query\nGPT-4o: ${GPT4O_COST_PER_QUERY}/query",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )

        out_path = safe_output_path(output_dir / "cost_performance_regime_map.png")
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")

    logger.info("Saved regime map to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Layered defense cost model
# ---------------------------------------------------------------------------


def compute_layered_defense_curve(
    classifier_f1_known: float,
    classifier_f1_novel: float,
    gpt4o_f1_novel: float,
    n_points: int = 50,
) -> list[LayeredDefensePoint]:
    """Model cost and F1 of a layered defense at varying escalation rates.

    Assumes: classifier screens all inputs. A fraction (escalation_rate) of
    inputs are escalated to GPT-4o. Escalated inputs get GPT-4o's F1;
    non-escalated get the classifier's F1.

    Parameters
    ----------
    classifier_f1_known : float
        Classifier F1 on known attacks (high).
    classifier_f1_novel : float
        Classifier F1 on novel attacks (low).
    gpt4o_f1_novel : float
        GPT-4o F1 on novel attacks.
    n_points : int
        Number of escalation rates to sample.

    Returns
    -------
    list[LayeredDefensePoint]
        Cost and estimated F1 at each escalation rate.
    """
    points: list[LayeredDefensePoint] = []

    for esc_rate in np.linspace(0.0, 1.0, n_points):
        # Cost: classifier always runs + GPT-4o for escalated fraction
        cost = CLASSIFIER_COST_PER_QUERY + esc_rate * GPT4O_COST_PER_QUERY

        # F1 model: linear interpolation between classifier and GPT-4o
        # on the novel attack surface. Known attacks always use classifier.
        # This is a simplification — real F1 depends on which samples
        # are escalated. We model the "oracle escalation" scenario where
        # the classifier's uncertain predictions are the ones escalated.
        f1_novel = (1 - esc_rate) * classifier_f1_novel + esc_rate * gpt4o_f1_novel

        points.append(
            LayeredDefensePoint(
                escalation_rate=float(esc_rate),
                total_cost_per_query=float(cost),
                estimated_f1=float(f1_novel),
            )
        )

    return points


def plot_layered_defense(
    points: list[LayeredDefensePoint],
    classifier_f1_novel: float,
    gpt4o_f1_novel: float,
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Plot layered defense cost vs F1 curve.

    Parameters
    ----------
    points : list[LayeredDefensePoint]
        From compute_layered_defense_curve().
    classifier_f1_novel : float
        Classifier-only F1 baseline.
    gpt4o_f1_novel : float
        GPT-4o-only F1 baseline.
    output_dir : Path
        Output directory.
    dpi : int
        Figure DPI.

    Returns
    -------
    Path
        Path to saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    esc_rates = [p.escalation_rate for p in points]
    costs = [p.total_cost_per_query for p in points]
    f1s = [p.estimated_f1 for p in points]

    with managed_figure(figsize=(12, 5), dpi=dpi) as (fig, ax):
        # Create twin axes
        ax2 = ax.twinx()

        # F1 curve
        ln1 = ax.plot(
            esc_rates,
            f1s,
            "-",
            color="#2196F3",
            linewidth=2.5,
            label="Estimated F1 (novel)",
        )
        ax.axhline(
            y=classifier_f1_novel,
            color="#2196F3",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )
        ax.axhline(
            y=gpt4o_f1_novel,
            color="#FF5722",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )

        # Cost curve
        ln2 = ax2.plot(
            esc_rates,
            [c * 1000 for c in costs],
            "--",
            color="#FF5722",
            linewidth=2.5,
            label="Cost ($/1K queries)",
        )

        # Mark key escalation rates
        for esc, label, color in [
            (0.10, "10% escalation", "#4CAF50"),
            (0.25, "25% escalation", "#FF9800"),
            (0.50, "50% escalation", "#9C27B0"),
        ]:
            f1_at_esc = (1 - esc) * classifier_f1_novel + esc * gpt4o_f1_novel
            cost_at_esc = CLASSIFIER_COST_PER_QUERY + esc * GPT4O_COST_PER_QUERY
            ax.plot(esc, f1_at_esc, "o", color=color, markersize=8, zorder=5)
            ax.annotate(
                f"{label}\nF1={f1_at_esc:.3f}\n${cost_at_esc * 1000:.2f}/1K",
                (esc, f1_at_esc),
                textcoords="offset points",
                xytext=(15, -5),
                fontsize=8,
                color=color,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
            )

        ax.set_xlabel("Escalation Rate (fraction sent to GPT-4o)", fontsize=12)
        ax.set_ylabel("Estimated F1 (novel attacks)", fontsize=12, color="#2196F3")
        ax2.set_ylabel("Cost per 1K queries ($)", fontsize=12, color="#FF5722")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.0)

        # Combined legend
        lns = ln1 + ln2
        labs: list[str] = [str(ln.get_label()) for ln in lns]
        ax.legend(lns, labs, loc="center right", fontsize=10)

        ax.set_title(
            "Layered Defense: Cost vs F1 by Escalation Rate",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        out_path = safe_output_path(output_dir / "layered_defense_cost_curve.png")
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")

    logger.info("Saved layered defense curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# GPT-4o precision-recall analysis
# ---------------------------------------------------------------------------


def build_precision_recall_interpretation(
    llm_df: pd.DataFrame,
) -> dict[str, object]:
    """Interpret GPT-4o's precision-recall tradeoff for deployment.

    Parameters
    ----------
    llm_df : pd.DataFrame
        LLM results from load_llm_results().

    Returns
    -------
    dict
        Interpretation with key metrics and deployment implications.
    """
    interp: dict[str, object] = {}

    for _, row in llm_df.iterrows():
        pool = str(row["test_pool"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        f1 = float(row["f1"])

        # False positive rate = 1 - precision (approximate)
        fpr_approx = 1 - precision
        # Miss rate = 1 - recall
        miss_rate = 1 - recall

        interp[pool] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": round(fpr_approx, 4),
            "miss_rate": round(miss_rate, 4),
            "interpretation": (
                f"Precision {precision:.3f}: ~{fpr_approx * 100:.1f}% of flagged "
                f"inputs are false positives (benign text wrongly blocked). "
                f"Recall {recall:.3f}: ~{miss_rate * 100:.1f}% of actual attacks "
                f"are missed (pass through undetected)."
            ),
        }

    return interp


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def build_cost_performance_summary(
    comparison_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    defense_points: list[LayeredDefensePoint],
    pr_interpretation: dict[str, object],
) -> dict[str, object]:
    """Build JSON summary of all cost-performance findings.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Side-by-side comparison table.
    regime_df : pd.DataFrame
        Regime map data.
    defense_points : list[LayeredDefensePoint]
        Layered defense cost curve.
    pr_interpretation : dict
        GPT-4o precision-recall interpretation.

    Returns
    -------
    dict
        Complete summary for JSON serialization.
    """
    # Key escalation points
    key_rates: dict[float, dict[str, float] | None] = {
        0.10: None,
        0.25: None,
        0.50: None,
        1.00: None,
    }
    for p in defense_points:
        for rate in key_rates:
            if abs(p.escalation_rate - rate) < 0.02:
                key_rates[rate] = {
                    "escalation_rate": p.escalation_rate,
                    "cost_per_query": p.total_cost_per_query,
                    "cost_per_1k": p.total_cost_per_query * 1000,
                    "estimated_f1": p.estimated_f1,
                }

    return {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "cost_per_query": {
            "classifier": CLASSIFIER_COST_PER_QUERY,
            "gpt4o": GPT4O_COST_PER_QUERY,
            "ratio": f"{COST_RATIO:.0f}x",
        },
        "regime_map": regime_df.to_dict(orient="records"),
        "layered_defense": {
            "model": (
                "Linear interpolation: F1 = (1-esc)*clf_f1 + esc*gpt4o_f1. "
                "Cost = clf_cost + esc*gpt4o_cost."
            ),
            "key_escalation_points": key_rates,
        },
        "gpt4o_precision_recall": pr_interpretation,
    }


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------


def generate_comparison_markdown(comparison_df: pd.DataFrame) -> str:
    """Generate markdown comparison table.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        From build_comparison_table().

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "## Cost-Performance Comparison: Classifier vs GPT-4o",
        "",
        "| Test Pool | System | F1 | Precision | Recall | Cost/Query |",
        "|-----------|--------|-----|-----------|--------|------------|",
    ]

    for _, row in comparison_df.iterrows():
        cost_str = f"${row['cost_per_query']:.4f}" if row["cost_per_query"] >= 0.0001 else "~$0"
        f1_str = f"**{row['f1']:.4f}**" if row["winner"] != "—" else f"{row['f1']:.4f}"
        lines.append(
            f"| {row['test_pool']} | {row['system']} | "
            f"{f1_str} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {cost_str} |"
        )

    lines.extend(
        [
            "",
            "## Layered Defense Cost Model",
            "",
            "| Escalation Rate | Cost/1K Queries | Est. F1 (novel) | F1 Gain vs Classifier-Only |",
            "|-----------------|-----------------|-----------------|---------------------------|",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_cost_performance_analysis(
    results_dir: Path,
    llm_results_path: Path,
    output_dir: Path,
    threshold_results_path: Path | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Run the full cost-performance analysis pipeline.

    Parameters
    ----------
    results_dir : Path
        Directory containing experiment result JSONs.
    llm_results_path : Path
        Path to llm_baseline_gpt_4o_all.json.
    output_dir : Path
        Output directory for tables, figures, and JSON.
    threshold_results_path : Path, optional
        Path to transfer_threshold_analysis.json.
    dpi : int
        Figure DPI.

    Returns
    -------
    dict[str, Path]
        Map of output name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # 1. Load data
    logger.info("Loading classifier results from %s", results_dir)
    clf_df = load_classifier_results(results_dir)
    if clf_df.empty:
        logger.error("No classifier results found in %s", results_dir)
        return outputs

    logger.info("Loading LLM results from %s", llm_results_path)
    llm_df = load_llm_results(llm_results_path)

    # 2. Side-by-side comparison table
    logger.info("Building comparison table")
    comparison_df = build_comparison_table(clf_df, llm_df)

    # 3. Regime map data + figure
    logger.info("Building regime map")
    regime_df = build_regime_data(clf_df, llm_df, threshold_results_path)
    regime_path = plot_regime_map(regime_df, figures_dir, dpi=dpi)
    outputs["regime_map_figure"] = regime_path

    # 4. Layered defense cost model
    logger.info("Computing layered defense cost curve")
    scv_df = clf_df[clf_df["experiment"] == "standard_cv"]
    di_df = clf_df[clf_df["experiment"] == "direct_indirect"]
    best_clf_known = float(scv_df["f1"].max())
    best_clf_novel = float(di_df["f1"].max())
    gpt4o_novel = float(llm_df[llm_df["test_pool"] == "direct_indirect"]["f1"].iloc[0])

    defense_points = compute_layered_defense_curve(
        classifier_f1_known=best_clf_known,
        classifier_f1_novel=best_clf_novel,
        gpt4o_f1_novel=gpt4o_novel,
    )

    defense_path = plot_layered_defense(
        defense_points,
        classifier_f1_novel=best_clf_novel,
        gpt4o_f1_novel=gpt4o_novel,
        output_dir=figures_dir,
        dpi=dpi,
    )
    outputs["layered_defense_figure"] = defense_path

    # 5. GPT-4o precision-recall interpretation
    logger.info("Building precision-recall interpretation")
    pr_interp = build_precision_recall_interpretation(llm_df)

    # 6. Summary JSON
    summary = build_cost_performance_summary(
        comparison_df,
        regime_df,
        defense_points,
        pr_interp,
    )
    json_path = output_dir / "cost_performance_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    outputs["summary_json"] = json_path

    # 7. Markdown table
    md_content = generate_comparison_markdown(comparison_df)

    # Add layered defense rows
    md_lines = [md_content]
    base_clf_f1 = best_clf_novel
    for p in defense_points:
        rate = p.escalation_rate
        if rate in (0.0, 0.10, 0.25, 0.50, 1.0) or abs(rate % 0.25) < 0.02:
            if abs(rate - round(rate, 2)) < 0.005:
                f1_gain = p.estimated_f1 - base_clf_f1
                gain_str = f"+{f1_gain:.3f}" if f1_gain > 0 else f"{f1_gain:.3f}"
                md_lines.append(
                    f"| {rate:.0%} | ${p.total_cost_per_query * 1000:.3f} | "
                    f"{p.estimated_f1:.4f} | {gain_str} |"
                )

    md_lines.extend(
        [
            "",
            "## GPT-4o Precision-Recall Interpretation",
            "",
        ]
    )
    for pool, data in pr_interp.items():
        if isinstance(data, dict):
            md_lines.append(f"**{pool}**: {data.get('interpretation', '')}")
            md_lines.append("")

    md_path = output_dir / "cost_performance_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    outputs["markdown_table"] = md_path

    logger.info("Cost-performance analysis complete. %d outputs.", len(outputs))
    return outputs
