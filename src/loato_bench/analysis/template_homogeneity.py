"""Template homogeneity analysis: correlate category-level template reuse with ΔF1.

Implements LOATO-4B-04:
- template_homogeneity_score per LOATO fold (mean max cosine sim, test→train)
- Scatter plot: homogeneity vs ΔF1 with regression line and R²
- Inter-category centroid distance matrix heatmap
- UMAP 2D projection colored by attack category
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from loato_bench.analysis.visualization import managed_figure, safe_output_path

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")
sns.set_style("whitegrid")

# LOATO-eligible categories (C1–C5); excludes C7 "other"
LOATO_CATEGORIES = [
    "instruction_override",
    "jailbreak_roleplay",
    "obfuscation_encoding",
    "information_extraction",
    "social_engineering",
]

CATEGORY_SHORT: dict[str, str] = {
    "instruction_override": "C1",
    "jailbreak_roleplay": "C2",
    "obfuscation_encoding": "C3",
    "information_extraction": "C4",
    "social_engineering": "C5",
    "context_manipulation": "C6",
    "other": "C7",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FoldHomogeneity:
    """Template homogeneity score and ΔF1 for one LOATO fold."""

    category: str
    template_homogeneity_score: float
    mean_delta_f1: float
    n_test: int
    n_train: int


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_template_homogeneity(
    embeddings: NDArray[np.float32],
    train_indices: list[int],
    test_indices: list[int],
) -> float:
    """Compute template homogeneity score for a single LOATO fold.

    For each test sample, finds its nearest neighbor in the train set
    (cosine similarity) and returns the mean of max similarities.

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        Full embedding matrix (N, D).
    train_indices : list[int]
        Row indices of training samples.
    test_indices : list[int]
        Row indices of test samples.

    Returns
    -------
    float
        Mean max cosine similarity (0–1). Higher = more template reuse.
    """
    train_emb = embeddings[train_indices]
    test_emb = embeddings[test_indices]

    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(train_emb)
    distances, _ = nn.kneighbors(test_emb)

    # cosine distance → cosine similarity
    similarities = 1.0 - distances[:, 0]
    return float(np.mean(similarities))


def load_loato_delta_f1(results_dir: Path) -> dict[str, float]:
    """Load mean ΔF1 per LOATO fold, averaged across all embedding-classifier combos.

    Parameters
    ----------
    results_dir : Path
        Directory containing experiment result JSONs.

    Returns
    -------
    dict[str, float]
        Mapping of held-out category → mean ΔF1.
    """
    cv_f1s: dict[tuple[str, str], float] = {}
    loato_fold_f1s: dict[tuple[str, str], dict[str, float]] = {}

    for path in sorted(results_dir.glob("*.json")):
        if "all_results" in path.name:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        key = (data["embedding"], data["classifier"])

        if data["experiment"] == "standard_cv":
            cv_f1s[key] = data["mean_f1"]
        elif data["experiment"] == "loato":
            fold_map: dict[str, float] = {}
            for fold in data["folds"]:
                cat = fold.get("held_out_category")
                if cat:
                    fold_map[cat] = fold["metrics"]["f1"]["value"]
            loato_fold_f1s[key] = fold_map

    # Mean ΔF1 per category across all combos
    category_deltas: dict[str, list[float]] = {}
    for key, fold_map in loato_fold_f1s.items():
        cv_f1 = cv_f1s.get(key)
        if cv_f1 is None:
            continue
        for cat, loato_f1 in fold_map.items():
            if cat not in category_deltas:
                category_deltas[cat] = []
            category_deltas[cat].append(cv_f1 - loato_f1)

    return {cat: float(np.mean(deltas)) for cat, deltas in category_deltas.items()}


def compute_centroid_distances(
    embeddings: NDArray[np.float32],
    categories: NDArray[np.str_],
    labels: NDArray[np.int64],
) -> tuple[pd.DataFrame, dict[str, NDArray[np.float32]]]:
    """Compute inter-category embedding centroid cosine distance matrix.

    Uses injection samples only (label=1) for centroid computation.

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        Full embedding matrix (N, D).
    categories : NDArray[np.str_]
        Attack category per sample.
    labels : NDArray[np.int64]
        Binary labels (0=benign, 1=injection).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, NDArray[np.float32]]]
        Distance matrix DataFrame and centroid dict.
    """
    injection_mask = labels == 1
    inj_cats = categories[injection_mask]
    inj_emb = embeddings[injection_mask]

    unique_cats = sorted(set(inj_cats) - {"", "nan", "None"})

    centroids: dict[str, NDArray[np.float32]] = {}
    for cat in unique_cats:
        cat_mask = inj_cats == cat
        if np.sum(cat_mask) > 0:
            centroids[cat] = np.mean(inj_emb[cat_mask], axis=0)

    cats = sorted(centroids.keys())
    n = len(cats)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = cosine_distance(centroids[cats[i]], centroids[cats[j]])

    display_cats = [CATEGORY_SHORT.get(c, c) for c in cats]
    df = pd.DataFrame(dist_matrix, index=display_cats, columns=display_cats)
    return df, centroids


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_homogeneity_vs_delta_f1(
    fold_data: list[FoldHomogeneity],
    output_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """Scatter plot of template homogeneity vs mean ΔF1 with regression line.

    Parameters
    ----------
    fold_data : list[FoldHomogeneity]
        One entry per LOATO-eligible fold.
    output_dir : Path
        Directory to save figures.
    dpi : int
        Figure DPI.

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    saved: list[Path] = []

    x = np.array([f.template_homogeneity_score for f in fold_data])
    y = np.array([f.mean_delta_f1 for f in fold_data])
    point_labels = [CATEGORY_SHORT.get(f.category, f.category) for f in fold_data]

    slope, intercept, r_value, p_value, _std_err = stats.linregress(x, y)
    r_squared = r_value**2

    for ext in ("png", "pdf"):
        out = safe_output_path(output_dir / f"homogeneity_vs_delta_f1.{ext}")

        with managed_figure(figsize=(8, 6), dpi=dpi) as (fig, ax):
            ax.scatter(x, y, s=120, c="#e74c3c", edgecolors="black", zorder=5)

            for xi, yi, label in zip(x, y, point_labels, strict=True):
                ax.annotate(
                    label,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=11,
                    fontweight="bold",
                )

            # Regression line
            x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "--", color="#3498db", linewidth=2, alpha=0.7)

            stats_text = (
                f"R² = {r_squared:.3f}\nr = {r_value:.3f}\np = {p_value:.3f}\nslope = {slope:.3f}"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel(
                "Template Homogeneity Score\n(mean max cosine sim, test → train)", fontsize=12
            )
            ax.set_ylabel("Mean ΔF1 (Standard CV − LOATO)", fontsize=12)
            ax.set_title(
                "Template Homogeneity Predicts Generalization Gap",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            plt.tight_layout()
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            saved.append(out)

    logger.info("Saved homogeneity vs ΔF1 scatter to %s", output_dir)
    return saved


def plot_centroid_distance_heatmap(
    dist_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """Plot inter-category centroid cosine distance heatmap.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Square distance matrix (category × category).
    output_dir : Path
        Directory to save figures.
    dpi : int
        Figure DPI.

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    saved: list[Path] = []

    for ext in ("png", "pdf"):
        out = safe_output_path(output_dir / f"category_centroid_distances.{ext}")

        with managed_figure(figsize=(8, 7), dpi=dpi) as (fig, ax):
            sns.heatmap(
                dist_df,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd_r",
                linewidths=0.5,
                cbar_kws={"label": "Cosine Distance"},
                ax=ax,
                vmin=0,
                vmax=float(dist_df.values.max()),
                square=True,
            )
            ax.set_title(
                "Inter-Category Centroid Cosine Distance\n(MiniLM embeddings, injection samples)",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )

            plt.tight_layout()
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            saved.append(out)

    logger.info("Saved centroid distance heatmap to %s", output_dir)
    return saved


def plot_umap_projection(
    embeddings: NDArray[np.float32],
    categories: NDArray[np.str_],
    labels: NDArray[np.int64],
    output_dir: Path,
    max_samples: int = 8000,
    seed: int = 42,
    dpi: int = 150,
) -> list[Path]:
    """UMAP 2D projection of injection samples colored by attack category.

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        Full embedding matrix (N, D).
    categories : NDArray[np.str_]
        Attack category per sample.
    labels : NDArray[np.int64]
        Binary labels (0=benign, 1=injection).
    output_dir : Path
        Directory to save figures.
    max_samples : int
        Max injection samples to project (subsampled if larger).
    seed : int
        Random seed for UMAP and subsampling.
    dpi : int
        Figure DPI.

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    import umap

    saved: list[Path] = []
    rng = np.random.RandomState(seed)

    injection_mask = labels == 1
    inj_indices = np.where(injection_mask)[0]
    inj_emb = embeddings[inj_indices]
    inj_cats = categories[inj_indices]

    if len(inj_emb) > max_samples:
        sample_idx = rng.choice(len(inj_emb), max_samples, replace=False)
        inj_emb = inj_emb[sample_idx]
        inj_cats = inj_cats[sample_idx]

    logger.info("Running UMAP on %d injection samples...", len(inj_emb))
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    )
    coords = reducer.fit_transform(inj_emb)

    plot_df = pd.DataFrame(
        {
            "UMAP-1": coords[:, 0],
            "UMAP-2": coords[:, 1],
            "Category": [CATEGORY_SHORT.get(str(c), str(c)) for c in inj_cats],
        }
    )

    for ext in ("png", "pdf"):
        out = safe_output_path(output_dir / f"umap_category_projection.{ext}")

        with managed_figure(figsize=(10, 8), dpi=dpi) as (fig, ax):
            n_cats = plot_df["Category"].nunique()
            palette = sns.color_palette("Set2", n_cats)
            sns.scatterplot(
                data=plot_df,
                x="UMAP-1",
                y="UMAP-2",
                hue="Category",
                palette=palette,
                s=15,
                alpha=0.6,
                linewidth=0,
                ax=ax,
            )
            ax.set_title(
                "UMAP Projection of Injection Samples by Category\n(MiniLM embeddings)",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )
            ax.legend(
                title="Category",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                markerscale=3,
            )

            plt.tight_layout()
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            saved.append(out)

    logger.info("Saved UMAP projection to %s", output_dir)
    return saved


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def build_analysis_summary(
    fold_data: list[FoldHomogeneity],
    dist_df: pd.DataFrame,
    r_squared: float,
    r_value: float,
    p_value: float,
) -> dict[str, object]:
    """Build JSON-serializable summary of the template homogeneity analysis.

    Parameters
    ----------
    fold_data : list[FoldHomogeneity]
        Per-fold homogeneity and ΔF1 data.
    dist_df : pd.DataFrame
        Inter-category centroid distance matrix.
    r_squared : float
        R² from homogeneity–ΔF1 regression.
    r_value : float
        Pearson r from regression.
    p_value : float
        p-value from regression.

    Returns
    -------
    dict[str, object]
        Summary suitable for JSON serialization.
    """
    return {
        "analysis": "template_homogeneity_4B-04",
        "folds": [
            {
                "category": f.category,
                "category_id": CATEGORY_SHORT.get(f.category, f.category),
                "template_homogeneity_score": round(f.template_homogeneity_score, 6),
                "mean_delta_f1": round(f.mean_delta_f1, 4),
                "n_test": f.n_test,
                "n_train": f.n_train,
            }
            for f in fold_data
        ],
        "regression": {
            "r_squared": round(r_squared, 4),
            "pearson_r": round(r_value, 4),
            "p_value": round(p_value, 4),
            "interpretation": (
                "Negative r: higher homogeneity → lower ΔF1 (easier to generalize). "
                "Positive r: higher homogeneity → higher ΔF1 (unexpected)."
            ),
        },
        "centroid_distances": {
            "metric": "cosine_distance",
            "embedding": "minilm_384d",
            "matrix": dist_df.to_dict(),
        },
        "hypothesis": (
            "Categories with high template homogeneity (test samples resemble training "
            "patterns from other categories) should have low ΔF1. Categories with low "
            "homogeneity (unique patterns) should have high ΔF1."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_template_homogeneity_analysis(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    categories: NDArray[np.str_],
    splits_path: Path,
    results_dir: Path,
    output_dir: Path,
    dpi: int = 150,
    seed: int = 42,
) -> dict[str, Path]:
    """Run the full template homogeneity analysis (LOATO-4B-04).

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        MiniLM embeddings for all samples (N, 384).
    labels : NDArray[np.int64]
        Binary labels (0=benign, 1=injection).
    categories : NDArray[np.str_]
        Attack category per sample.
    splits_path : Path
        Path to loato_splits.json.
    results_dir : Path
        Directory containing experiment result JSONs.
    output_dir : Path
        Output directory for figures and JSON.
    dpi : int
        Figure DPI.
    seed : int
        Random seed for UMAP.

    Returns
    -------
    dict[str, Path]
        Mapping of output name to file path.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # 1. Load LOATO splits
    with open(splits_path) as f:
        split_data = json.load(f)

    # 2. Load ΔF1 per category
    delta_f1_map = load_loato_delta_f1(results_dir)
    logger.info("ΔF1 per category: %s", {k: f"{v:.4f}" for k, v in delta_f1_map.items()})

    # 3. Compute template homogeneity per LOATO-eligible fold
    fold_data: list[FoldHomogeneity] = []
    for fold in split_data["folds"]:
        cat = fold["held_out_category"]
        if cat not in LOATO_CATEGORIES:
            logger.info("Skipping non-eligible category: %s", cat)
            continue

        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]

        logger.info(
            "Computing homogeneity for %s (test=%d, train=%d)...",
            cat,
            len(test_idx),
            len(train_idx),
        )
        score = compute_template_homogeneity(embeddings, train_idx, test_idx)

        delta_f1 = delta_f1_map.get(cat, float("nan"))
        fold_data.append(
            FoldHomogeneity(
                category=cat,
                template_homogeneity_score=score,
                mean_delta_f1=delta_f1,
                n_test=len(test_idx),
                n_train=len(train_idx),
            )
        )

    logger.info("Computed homogeneity for %d folds", len(fold_data))
    for fh in fold_data:
        logger.info(
            "  %s: homogeneity=%.4f, ΔF1=%.4f",
            CATEGORY_SHORT.get(fh.category, fh.category),
            fh.template_homogeneity_score,
            fh.mean_delta_f1,
        )

    # 4. Scatter plot: homogeneity vs ΔF1
    scatter_paths = plot_homogeneity_vs_delta_f1(fold_data, figures_dir, dpi=dpi)
    for p in scatter_paths:
        outputs[f"scatter_{p.suffix[1:]}"] = p

    # Regression stats for summary
    x = np.array([fh.template_homogeneity_score for fh in fold_data])
    y = np.array([fh.mean_delta_f1 for fh in fold_data])
    _slope, _intercept, r_value, p_value, _std_err = stats.linregress(x, y)
    r_squared = r_value**2

    # 5. Centroid distance matrix
    dist_df, _centroids = compute_centroid_distances(embeddings, categories, labels)
    heatmap_paths = plot_centroid_distance_heatmap(dist_df, figures_dir, dpi=dpi)
    for p in heatmap_paths:
        outputs[f"heatmap_{p.suffix[1:]}"] = p

    # 6. UMAP projection
    umap_paths = plot_umap_projection(
        embeddings,
        categories,
        labels,
        figures_dir,
        max_samples=8000,
        seed=seed,
        dpi=dpi,
    )
    for p in umap_paths:
        outputs[f"umap_{p.suffix[1:]}"] = p

    # 7. Save summary JSON
    summary = build_analysis_summary(fold_data, dist_df, r_squared, r_value, p_value)
    summary_path = output_dir / "template_homogeneity_analysis.json"
    with open(summary_path, "w") as out_f:
        json.dump(summary, out_f, indent=2)
    outputs["summary_json"] = summary_path

    logger.info("Template homogeneity analysis complete. %d outputs.", len(outputs))
    return outputs
