"""Auto-generate markdown and LaTeX summary tables, heatmaps, and per-fold charts.

Consolidates Sprint 3 experiment results into publication-ready outputs:
- Master results table (markdown + LaTeX)
- ΔF1 heatmap (embedding × classifier)
- Per-fold bar chart (F1 by held-out category)
- Statistical significance tests (Wilcoxon signed-rank)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from loato_bench.analysis.visualization import managed_figure, safe_output_path

logger = logging.getLogger(__name__)

# Set non-GUI backend
plt.switch_backend("Agg")
sns.set_style("whitegrid")

# Ordered lists for consistent table/heatmap layout
EMBEDDING_ORDER = ["minilm", "bge_large", "instructor", "openai_small", "e5_mistral"]
CLASSIFIER_ORDER = ["logreg", "svm", "xgboost", "mlp"]

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ResultRow:
    """Flat representation of one experiment run for tabular analysis."""

    experiment: str
    embedding: str
    classifier: str
    mean_f1: float
    std_f1: float
    fold_f1s: list[float] = field(default_factory=list)
    fold_categories: list[str | None] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    auc_roc: float = 0.0


def load_result_json(path: Path) -> ResultRow:
    """Load a single experiment result JSON into a ResultRow.

    Parameters
    ----------
    path : Path
        Path to an experiment result JSON file.

    Returns
    -------
    ResultRow
        Flattened result for tabular analysis.
    """
    with open(path) as f:
        data = json.load(f)

    folds = data.get("folds", [])
    fold_f1s = [fold["metrics"]["f1"]["value"] for fold in folds]
    fold_cats = [fold.get("held_out_category") for fold in folds]

    # Average metrics across folds
    avg_prec = float(np.mean([f["metrics"]["precision"]["value"] for f in folds])) if folds else 0.0
    avg_rec = float(np.mean([f["metrics"]["recall"]["value"] for f in folds])) if folds else 0.0
    avg_acc = float(np.mean([f["metrics"]["accuracy"]["value"] for f in folds])) if folds else 0.0
    avg_auc = float(np.mean([f["metrics"]["auc_roc"]["value"] for f in folds])) if folds else 0.0

    return ResultRow(
        experiment=data["experiment"],
        embedding=data["embedding"],
        classifier=data["classifier"],
        mean_f1=data["mean_f1"],
        std_f1=data["std_f1"],
        fold_f1s=fold_f1s,
        fold_categories=fold_cats,
        precision=avg_prec,
        recall=avg_rec,
        accuracy=avg_acc,
        auc_roc=avg_auc,
    )


def load_all_results(results_dir: Path) -> list[ResultRow]:
    """Load all individual experiment result JSONs from a directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``{experiment}_{embedding}_{classifier}.json`` files.

    Returns
    -------
    list[ResultRow]
        All loaded results (excludes aggregated ``*_all_results.json`` files).
    """
    rows: list[ResultRow] = []
    for path in sorted(results_dir.glob("*.json")):
        if "all_results" in path.name:
            continue
        try:
            rows.append(load_result_json(path))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
    return rows


# ---------------------------------------------------------------------------
# Master results table
# ---------------------------------------------------------------------------


def build_master_table(results: list[ResultRow]) -> pd.DataFrame:
    """Build the master results table with Standard CV F1, LOATO F1, and ΔF1.

    Parameters
    ----------
    results : list[ResultRow]
        All experiment results (standard_cv + loato).

    Returns
    -------
    pd.DataFrame
        Table with columns: Embedding, Classifier, CV_F1, LOATO_F1, Delta_F1,
        Precision, Recall, AUC_ROC.
    """
    cv_map: dict[tuple[str, str], ResultRow] = {}
    loato_map: dict[tuple[str, str], ResultRow] = {}

    for r in results:
        key = (r.embedding, r.classifier)
        if r.experiment == "standard_cv":
            cv_map[key] = r
        elif r.experiment == "loato":
            loato_map[key] = r

    rows = []
    for emb in EMBEDDING_ORDER:
        for clf in CLASSIFIER_ORDER:
            key = (emb, clf)
            cv = cv_map.get(key)
            loato = loato_map.get(key)
            if cv is None and loato is None:
                continue

            cv_f1 = cv.mean_f1 if cv else float("nan")
            loato_f1 = loato.mean_f1 if loato else float("nan")
            delta_f1 = cv_f1 - loato_f1

            rows.append(
                {
                    "Embedding": EMBEDDING_DISPLAY.get(emb, emb),
                    "Classifier": CLASSIFIER_DISPLAY.get(clf, clf),
                    "CV_F1": cv_f1,
                    "CV_F1_std": cv.std_f1 if cv else float("nan"),
                    "LOATO_F1": loato_f1,
                    "LOATO_F1_std": loato.std_f1 if loato else float("nan"),
                    "Delta_F1": delta_f1,
                    "CV_Precision": cv.precision if cv else float("nan"),
                    "CV_Recall": cv.recall if cv else float("nan"),
                    "CV_AUC_ROC": cv.auc_roc if cv else float("nan"),
                    "LOATO_Precision": loato.precision if loato else float("nan"),
                    "LOATO_Recall": loato.recall if loato else float("nan"),
                    "LOATO_AUC_ROC": loato.auc_roc if loato else float("nan"),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-fold breakdown
# ---------------------------------------------------------------------------


def build_per_fold_table(results: list[ResultRow]) -> pd.DataFrame:
    """Build per-fold F1 breakdown for LOATO experiments.

    Parameters
    ----------
    results : list[ResultRow]
        All experiment results (filters to LOATO only).

    Returns
    -------
    pd.DataFrame
        Table with columns: Embedding, Classifier, and one column per held-out category.
    """
    loato_results = [r for r in results if r.experiment == "loato"]
    if not loato_results:
        return pd.DataFrame()

    # Get category order from first result
    categories = [c for c in loato_results[0].fold_categories if c is not None]

    rows = []
    for r in loato_results:
        row: dict[str, str | float] = {
            "Embedding": EMBEDDING_DISPLAY.get(r.embedding, r.embedding),
            "Classifier": CLASSIFIER_DISPLAY.get(r.classifier, r.classifier),
        }
        for cat, f1 in zip(r.fold_categories, r.fold_f1s, strict=False):
            if cat is not None:
                row[cat] = f1
        row["Mean"] = r.mean_f1
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns
    fixed = ["Embedding", "Classifier"]
    cat_cols = [c for c in categories if c in df.columns]
    return df[fixed + cat_cols + ["Mean"]]


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


@dataclass
class SignificanceResult:
    """Result of a paired statistical test across folds.

    Attributes
    ----------
    embedding : str
        Embedding model name.
    classifier : str
        Classifier name.
    statistic : float
        Test statistic (Wilcoxon W or paired t).
    p_value : float
        Two-sided p-value.
    test_name : str
        Name of the test used.
    significant : bool
        Whether p < 0.05.
    cv_folds : list[float]
        Per-fold F1 scores for standard CV.
    loato_folds : list[float]
        Per-fold F1 scores for LOATO.
    """

    embedding: str
    classifier: str
    statistic: float
    p_value: float
    test_name: str
    significant: bool
    cv_folds: list[float] = field(default_factory=list)
    loato_folds: list[float] = field(default_factory=list)


def run_significance_tests(results: list[ResultRow]) -> list[SignificanceResult]:
    """Run paired significance tests (Wilcoxon or paired t-test) per combination.

    Uses Wilcoxon signed-rank test when fold counts match and n >= 6.
    Falls back to paired t-test for smaller fold counts.

    Parameters
    ----------
    results : list[ResultRow]
        All experiment results.

    Returns
    -------
    list[SignificanceResult]
        One result per (embedding, classifier) pair that has both CV and LOATO data.
    """
    cv_map: dict[tuple[str, str], ResultRow] = {}
    loato_map: dict[tuple[str, str], ResultRow] = {}

    for r in results:
        key = (r.embedding, r.classifier)
        if r.experiment == "standard_cv":
            cv_map[key] = r
        elif r.experiment == "loato":
            loato_map[key] = r

    sig_results: list[SignificanceResult] = []

    for emb in EMBEDDING_ORDER:
        for clf in CLASSIFIER_ORDER:
            key = (emb, clf)
            cv = cv_map.get(key)
            loato = loato_map.get(key)
            if cv is None or loato is None:
                continue

            cv_folds = cv.fold_f1s
            loato_folds = loato.fold_f1s

            # Need matched pairs — use min length
            n = min(len(cv_folds), len(loato_folds))
            if n < 2:
                continue

            cv_arr = np.array(cv_folds[:n])
            loato_arr = np.array(loato_folds[:n])

            # Choose test based on sample size
            if n >= 6:
                try:
                    stat_result = stats.wilcoxon(cv_arr, loato_arr, alternative="two-sided")
                    test_name = "Wilcoxon signed-rank"
                    statistic = float(stat_result.statistic)
                    p_value = float(stat_result.pvalue)
                except ValueError:
                    # Wilcoxon fails if all differences are zero
                    stat_result = stats.ttest_rel(cv_arr, loato_arr)
                    test_name = "Paired t-test"
                    statistic = float(stat_result.statistic)
                    p_value = float(stat_result.pvalue)
            else:
                stat_result = stats.ttest_rel(cv_arr, loato_arr)
                test_name = "Paired t-test"
                statistic = float(stat_result.statistic)
                p_value = float(stat_result.pvalue)

            sig_results.append(
                SignificanceResult(
                    embedding=EMBEDDING_DISPLAY.get(emb, emb),
                    classifier=CLASSIFIER_DISPLAY.get(clf, clf),
                    statistic=statistic,
                    p_value=p_value,
                    test_name=test_name,
                    significant=p_value < 0.05,
                    cv_folds=cv_folds[:n],
                    loato_folds=loato_folds[:n],
                )
            )

    return sig_results


# ---------------------------------------------------------------------------
# Table export (Markdown + LaTeX)
# ---------------------------------------------------------------------------


def table_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convert a DataFrame to a markdown table string.

    Parameters
    ----------
    df : pd.DataFrame
        Table to convert.
    title : str
        Optional title to prepend as a markdown heading.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = []
    if title:
        lines.append(f"## {title}\n")

    # Format float columns to 4 decimal places
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[np.floating]).columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    lines.append(formatted.to_markdown(index=False))
    return "\n".join(lines)


def table_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert a DataFrame to a LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
        Table to convert.
    caption : str
        LaTeX table caption.
    label : str
        LaTeX table label for cross-referencing.

    Returns
    -------
    str
        LaTeX-formatted table.
    """
    # Format float columns
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[np.floating]).columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "---")

    latex = formatted.to_latex(index=False, escape=True)

    if caption or label:
        # Wrap in table environment
        parts = ["\\begin{table}[htbp]", "\\centering"]
        if caption:
            parts.append(f"\\caption{{{caption}}}")
        if label:
            parts.append(f"\\label{{{label}}}")
        parts.append(latex)
        parts.append("\\end{table}")
        return "\n".join(parts)

    return latex


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_delta_f1_heatmap(
    master_table: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (10, 7),
    dpi: int = 150,
) -> None:
    """Plot ΔF1 heatmap (embedding × classifier).

    Parameters
    ----------
    master_table : pd.DataFrame
        Master results table from ``build_master_table``.
    output_path : Path
        Where to save the figure (PNG or PDF).
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    # Pivot to embedding × classifier
    pivot = master_table.pivot(index="Embedding", columns="Classifier", values="Delta_F1")

    # Reorder
    emb_order = [
        EMBEDDING_DISPLAY[e] for e in EMBEDDING_ORDER if EMBEDDING_DISPLAY[e] in pivot.index
    ]
    clf_order = [
        CLASSIFIER_DISPLAY[c] for c in CLASSIFIER_ORDER if CLASSIFIER_DISPLAY[c] in pivot.columns
    ]
    pivot = pivot.loc[emb_order, clf_order]

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": "ΔF1 (CV − LOATO)"},
            ax=ax,
            vmin=0,
        )
        ax.set_title(
            "Generalization Gap: ΔF1 = Standard CV − LOATO",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel("Embedding", fontsize=12)
        ax.set_xlabel("Classifier", fontsize=12)

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved ΔF1 heatmap to %s", output_path)


def plot_per_fold_f1(
    results: list[ResultRow],
    output_path: Path,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 150,
) -> None:
    """Plot per-fold F1 bar chart showing F1 by held-out category.

    Each (embedding, classifier) pair is a group; bars are colored by held-out category.

    Parameters
    ----------
    results : list[ResultRow]
        All experiment results (filters to LOATO only).
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    loato_results = [r for r in results if r.experiment == "loato"]
    if not loato_results:
        logger.warning("No LOATO results found. Skipping per-fold plot.")
        return

    # Build long-form DataFrame for seaborn
    rows = []
    for r in loato_results:
        combo = f"{r.embedding}\n{CLASSIFIER_DISPLAY.get(r.classifier, r.classifier)}"
        for cat, f1 in zip(r.fold_categories, r.fold_f1s, strict=False):
            if cat is not None:
                rows.append({"Combination": combo, "Held-Out Category": cat, "F1": f1})

    plot_df = pd.DataFrame(rows)

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        categories = plot_df["Held-Out Category"].unique()
        palette = sns.color_palette("Set2", len(categories))

        sns.barplot(
            data=plot_df,
            x="Combination",
            y="F1",
            hue="Held-Out Category",
            palette=palette,
            ax=ax,
        )

        ax.set_title(
            "LOATO Per-Fold F1 by Held-Out Category",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.legend(title="Held-Out Category", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved per-fold F1 chart to %s", output_path)


def plot_cv_vs_loato_comparison(
    master_table: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (12, 7),
    dpi: int = 150,
) -> None:
    """Plot grouped bar chart comparing Standard CV vs LOATO F1 per combination.

    Parameters
    ----------
    master_table : pd.DataFrame
        Master results table.
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    df = master_table.copy()
    df["Combo"] = df["Embedding"] + "\n" + df["Classifier"]

    # Melt for grouped bar
    melted = df.melt(
        id_vars=["Combo"],
        value_vars=["CV_F1", "LOATO_F1"],
        var_name="Protocol",
        value_name="F1",
    )
    melted["Protocol"] = melted["Protocol"].map({"CV_F1": "Standard CV", "LOATO_F1": "LOATO"})

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.barplot(
            data=melted,
            x="Combo",
            y="F1",
            hue="Protocol",
            palette=["#2ecc71", "#e74c3c"],
            ax=ax,
        )

        ax.set_title(
            "Standard CV vs LOATO F1 by Embedding × Classifier",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Protocol")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved CV vs LOATO comparison to %s", output_path)


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------


def generate_narrative_summary(
    master_table: pd.DataFrame,
    per_fold: pd.DataFrame,
    sig_results: list[SignificanceResult],
) -> str:
    """Generate a 1-2 paragraph narrative summary of key findings.

    Parameters
    ----------
    master_table : pd.DataFrame
        Master results table.
    per_fold : pd.DataFrame
        Per-fold LOATO breakdown.
    sig_results : list[SignificanceResult]
        Statistical significance results.

    Returns
    -------
    str
        Markdown-formatted narrative summary.
    """
    if master_table.empty:
        return "No results available for summary generation."

    best_cv = master_table.loc[master_table["CV_F1"].idxmax()]
    max_gap = master_table.loc[master_table["Delta_F1"].idxmax()]
    min_gap = master_table.loc[master_table["Delta_F1"].idxmin()]

    mean_cv = master_table["CV_F1"].mean()
    mean_loato = master_table["LOATO_F1"].mean()
    mean_gap = master_table["Delta_F1"].mean()

    n_significant = sum(1 for s in sig_results if s.significant)
    n_total = len(sig_results)

    lines = [
        "# LOATO-4B-01: Core Results & ΔF1 Analysis — Summary\n",
        f"Across {len(master_table)} embedding-classifier combinations, "
        f"the average Standard CV F1 is **{mean_cv:.4f}** while the average LOATO F1 is "
        f"**{mean_loato:.4f}**, yielding a mean generalization gap of "
        f"**ΔF1 = {mean_gap:.4f}**. "
        f"The best Standard CV performance is {best_cv['Embedding']} × {best_cv['Classifier']} "
        f"(F1 = {best_cv['CV_F1']:.4f}), "
        f"while the largest generalization gap belongs to {max_gap['Embedding']} × "
        f"{max_gap['Classifier']} (ΔF1 = {max_gap['Delta_F1']:.4f}). "
        f"The smallest gap is {min_gap['Embedding']} × {min_gap['Classifier']} "
        f"(ΔF1 = {min_gap['Delta_F1']:.4f}).\n",
        f"Statistical testing shows **{n_significant}/{n_total}** combinations exhibit a "
        f"statistically significant drop from Standard CV to LOATO (p < 0.05). ",
    ]

    # Add per-fold insight if available
    if not per_fold.empty:
        # Find which held-out category causes the biggest average drop
        cat_cols = [c for c in per_fold.columns if c not in ("Embedding", "Classifier", "Mean")]
        if cat_cols:
            cat_means = per_fold[cat_cols].mean()
            hardest = cat_means.idxmin()
            easiest = cat_means.idxmax()
            lines.append(
                f"Per-fold analysis reveals **{hardest}** as the hardest held-out category "
                f"(mean F1 = {cat_means[hardest]:.4f}) and **{easiest}** as the easiest "
                f"(mean F1 = {cat_means[easiest]:.4f}). "
                f"This connects to the template homogeneity analysis from Sprint 2A, where "
                f"categories with higher homogeneity (surface-level patterns) are easier to "
                f"detect even when held out, while semantically diverse categories expose the "
                f"generalization gap most starkly.\n"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_report(
    results_dir: Path,
    output_dir: Path,
    dpi: int = 150,
) -> dict[str, Path]:
    """Generate the full 4B-01 report: tables, figures, and narrative summary.

    Parameters
    ----------
    results_dir : Path
        Directory containing experiment result JSONs.
    output_dir : Path
        Directory to save all outputs (creates ``figures/`` and ``tables/`` subdirs).
    dpi : int
        DPI for saved figures.

    Returns
    -------
    dict[str, Path]
        Mapping of output name to file path.
    """
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # Load all results
    results = load_all_results(results_dir)
    logger.info("Loaded %d experiment results from %s", len(results), results_dir)

    if not results:
        logger.warning("No results found in %s", results_dir)
        return outputs

    # 1. Master results table
    master = build_master_table(results)
    if not master.empty:
        # Compact table for display: Embedding, Classifier, CV F1, LOATO F1, ΔF1
        display = master[["Embedding", "Classifier", "CV_F1", "LOATO_F1", "Delta_F1"]].copy()

        md_path = tables_dir / "master_results.md"
        md_path.write_text(table_to_markdown(display, title="Master Results Table"))
        outputs["master_table_md"] = md_path

        latex_path = tables_dir / "master_results.tex"
        latex_path.write_text(
            table_to_latex(
                display,
                caption="Standard CV vs LOATO F1 across all embedding-classifier combinations.",
                label="tab:master-results",
            )
        )
        outputs["master_table_tex"] = latex_path

        # Full table with all metrics
        full_md_path = tables_dir / "master_results_full.md"
        full_md_path.write_text(table_to_markdown(master, title="Full Results Table"))
        outputs["master_table_full_md"] = full_md_path

        logger.info("Generated master results tables")

    # 2. Per-fold breakdown
    per_fold = build_per_fold_table(results)
    if not per_fold.empty:
        md_path = tables_dir / "per_fold_breakdown.md"
        md_path.write_text(table_to_markdown(per_fold, title="LOATO Per-Fold F1 Breakdown"))
        outputs["per_fold_md"] = md_path

        latex_path = tables_dir / "per_fold_breakdown.tex"
        latex_path.write_text(
            table_to_latex(
                per_fold,
                caption="LOATO per-fold F1 by held-out attack category.",
                label="tab:per-fold",
            )
        )
        outputs["per_fold_tex"] = latex_path

        logger.info("Generated per-fold breakdown tables")

    # 3. Statistical significance
    sig_results = run_significance_tests(results)
    if sig_results:
        sig_rows = [
            {
                "Embedding": s.embedding,
                "Classifier": s.classifier,
                "Test": s.test_name,
                "Statistic": s.statistic,
                "p-value": s.p_value,
                "Significant": "Yes" if s.significant else "No",
            }
            for s in sig_results
        ]
        sig_df = pd.DataFrame(sig_rows)

        md_path = tables_dir / "significance_tests.md"
        md_path.write_text(
            table_to_markdown(sig_df, title="Statistical Significance Tests (CV vs LOATO)")
        )
        outputs["significance_md"] = md_path

        latex_path = tables_dir / "significance_tests.tex"
        latex_path.write_text(
            table_to_latex(
                sig_df,
                caption="Paired significance tests: Standard CV F1 vs LOATO F1.",
                label="tab:significance",
            )
        )
        outputs["significance_tex"] = latex_path

        logger.info("Generated significance test tables")

    # 4. Figures
    if not master.empty:
        # ΔF1 heatmap
        for ext in ("png", "pdf"):
            heatmap_path = figures_dir / f"delta_f1_heatmap.{ext}"
            plot_delta_f1_heatmap(master, heatmap_path, dpi=dpi)
            outputs[f"heatmap_{ext}"] = heatmap_path

        # CV vs LOATO comparison
        for ext in ("png", "pdf"):
            comp_path = figures_dir / f"cv_vs_loato_comparison.{ext}"
            plot_cv_vs_loato_comparison(master, comp_path, dpi=dpi)
            outputs[f"comparison_{ext}"] = comp_path

    if results:
        for ext in ("png", "pdf"):
            fold_path = figures_dir / f"per_fold_f1.{ext}"
            plot_per_fold_f1(results, fold_path, dpi=dpi)
            outputs[f"per_fold_{ext}"] = fold_path

    # 5. Narrative summary
    summary = generate_narrative_summary(master, per_fold, sig_results)
    summary_path = output_dir / "4b_01_summary.md"
    summary_path.write_text(summary)
    outputs["summary"] = summary_path

    # 6. Save raw data as JSON for downstream use
    raw_path = output_dir / "4b_01_raw_data.json"
    raw_data = {
        "master_table": master.to_dict(orient="records") if not master.empty else [],
        "per_fold": per_fold.to_dict(orient="records") if not per_fold.empty else [],
        "significance": [
            {
                "embedding": s.embedding,
                "classifier": s.classifier,
                "statistic": s.statistic,
                "p_value": s.p_value,
                "test_name": s.test_name,
                "significant": s.significant,
            }
            for s in sig_results
        ],
    }
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    outputs["raw_data"] = raw_path

    logger.info("Report generation complete. %d outputs saved to %s", len(outputs), output_dir)
    return outputs
