"""Visualization functions for EDA: distributions, heatmaps, and dashboards.

All plotting functions use managed_figure context manager for memory safety.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set non-GUI backend for server compatibility
plt.switch_backend("Agg")

# Set seaborn style
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Context manager for memory-safe figure handling
# ---------------------------------------------------------------------------


@contextmanager
def managed_figure(
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> Generator[tuple[Figure, Axes], None, None]:
    """Context manager for matplotlib figures with guaranteed cleanup.

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size (width, height) in inches.
    dpi : int
        DPI for saved figures.

    Yields
    ------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    try:
        yield fig, ax
    finally:
        plt.close(fig)
        # Explicit garbage collection for large figures
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def safe_output_path(output_path: Path, base_dir: Path | None = None) -> Path:
    """Validate output path stays within allowed directory.

    Parameters
    ----------
    output_path : Path
        Desired output path.
    base_dir : Path, optional
        Base directory (defaults to CWD). Output must be within this dir.

    Returns
    -------
    Path
        Validated absolute path.

    Raises
    ------
    ValueError
        If path is outside base_dir or has invalid extension.
    """
    # Validate extension whitelist
    allowed_extensions = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
    if output_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Invalid file extension: {output_path.suffix}. Allowed: {allowed_extensions}"
        )

    # Resolve to absolute path
    resolved = output_path.resolve()

    # Validate within base_dir if specified
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        if not str(resolved).startswith(str(base_resolved)):
            raise ValueError(f"Security: Path {output_path} is outside base directory {base_dir}")

    # Create parent directory if it doesn't exist
    resolved.parent.mkdir(parents=True, exist_ok=True)

    return resolved


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


def plot_label_distribution(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> None:
    """Plot label distribution (benign vs injection).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'label' column.
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size (width, height).
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Count labels
        counts = df["label"].value_counts().sort_index()
        labels_text = ["Benign (0)", "Injection (1)"]

        # Bar plot
        ax.bar(range(len(counts)), counts.values.tolist(), color=["#2ecc71", "#e74c3c"])
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels_text)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Label Distribution", fontsize=16, fontweight="bold")

        # Add count annotations
        for i, count in enumerate(counts.values):
            percentage = count / len(df) * 100
            ax.text(
                i,
                count + len(df) * 0.01,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved label distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Source breakdown
# ---------------------------------------------------------------------------


def plot_source_breakdown(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 150,
) -> None:
    """Plot sample counts by source.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'source' column.
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size (width, height).
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Count by source
        counts = df["source"].value_counts()

        # Horizontal bar plot
        colors = sns.color_palette("Set2", len(counts))
        ax.barh(range(len(counts)), counts.values.tolist(), color=colors)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index)
        ax.set_xlabel("Sample Count", fontsize=12)
        ax.set_title("Sample Distribution by Source", fontsize=16, fontweight="bold")

        # Add count annotations
        for i, count in enumerate(counts.values):
            percentage = count / len(df) * 100
            ax.text(
                count + len(df) * 0.01,
                i,
                f"{count:,} ({percentage:.1f}%)",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved source breakdown plot to {output_path}")


# ---------------------------------------------------------------------------
# Text length distribution
# ---------------------------------------------------------------------------


def plot_text_length_distribution(
    df: pd.DataFrame,
    output_path: Path,
    bins: list[int] | None = None,
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 150,
) -> None:
    """Plot text length distribution histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'text' column.
    output_path : Path
        Where to save the figure.
    bins : list[int], optional
        Bin edges for histogram. Defaults to [0, 50, 100, 200, 500, 1000, 2000, 5000].
    figsize : tuple[int, int]
        Figure size (width, height).
    dpi : int
        DPI for saved figure.
    """
    if bins is None:
        bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]

    output_path = safe_output_path(output_path)

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Compute text lengths
        lengths = np.array(df["text"].str.len(), dtype=np.int64)

        # Histogram
        ax.hist(lengths, bins=bins, color="#3498db", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Text Length (characters)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Text Length Distribution", fontsize=16, fontweight="bold")

        # Add statistics annotation
        stats_text = (
            f"Mean: {lengths.mean():.0f} chars\n"
            f"Median: {np.median(lengths):.0f} chars\n"
            f"Max: {lengths.max():,} chars"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved text length distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Language heatmap
# ---------------------------------------------------------------------------


def plot_language_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> None:
    """Plot heatmap of language × label distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'language' and 'label' columns.
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size (width, height).
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Create contingency table
        crosstab = pd.crosstab(df["language"], df["label"])
        crosstab.columns = ["Benign", "Injection"]

        # Sort by total count descending
        crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index]

        # Keep top 20 languages if more than 20
        if len(crosstab) > 20:
            crosstab = crosstab.head(20)

        # Heatmap
        sns.heatmap(
            crosstab, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={"label": "Sample Count"}, ax=ax
        )
        ax.set_xlabel("Label", fontsize=12)
        ax.set_ylabel("Language", fontsize=12)
        ax.set_title("Language × Label Distribution", fontsize=16, fontweight="bold")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved language heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Attack category distribution
# ---------------------------------------------------------------------------


def plot_attack_category_distribution(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> None:
    """Plot attack category distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'attack_category' column.
    output_path : Path
        Where to save the figure.
    figsize : tuple[int, int]
        Figure size (width, height).
    dpi : int
        DPI for saved figure.
    """
    output_path = safe_output_path(output_path)

    # Filter to injection samples only (label=1)
    injection_df = df[df["label"] == 1]

    # Count by attack_category (exclude nulls)
    counts = injection_df["attack_category"].value_counts().dropna()

    if counts.empty:
        logger.warning("No attack categories found. Skipping plot.")
        return

    with managed_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Horizontal bar plot
        colors = sns.color_palette("Set3", len(counts))
        ax.barh(range(len(counts)), counts.values.tolist(), color=colors)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index)
        ax.set_xlabel("Sample Count", fontsize=12)
        ax.set_title(
            "Attack Category Distribution (Injection Samples Only)", fontsize=16, fontweight="bold"
        )

        # Add count annotations
        for i, count in enumerate(counts.values):
            percentage = count / len(injection_df) * 100
            ax.text(
                count + len(injection_df) * 0.01,
                i,
                f"{count:,} ({percentage:.1f}%)",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved attack category distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Dashboard creation
# ---------------------------------------------------------------------------


def create_eda_dashboard(
    df: pd.DataFrame,
    output_dir: Path,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> list[Path]:
    """Create full EDA dashboard with all plots.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset.
    output_dir : Path
        Directory to save all figures.
    figsize : tuple[int, int]
        Default figure size for all plots.
    dpi : int
        DPI for saved figures.

    Returns
    -------
    list[Path]
        List of paths to saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    logger.info(f"Creating EDA dashboard in {output_dir}")

    # 1. Label distribution
    try:
        path = output_dir / "label_distribution.png"
        plot_label_distribution(df, path, figsize=figsize, dpi=dpi)
        saved_paths.append(path)
    except Exception as e:
        logger.error(f"Failed to create label distribution plot: {e}")

    # 2. Source breakdown
    try:
        path = output_dir / "source_breakdown.png"
        plot_source_breakdown(df, path, figsize=figsize, dpi=dpi)
        saved_paths.append(path)
    except Exception as e:
        logger.error(f"Failed to create source breakdown plot: {e}")

    # 3. Text length distribution
    try:
        path = output_dir / "text_length_distribution.png"
        plot_text_length_distribution(df, path, figsize=figsize, dpi=dpi)
        saved_paths.append(path)
    except Exception as e:
        logger.error(f"Failed to create text length distribution plot: {e}")

    # 4. Language heatmap
    try:
        path = output_dir / "language_heatmap.png"
        plot_language_heatmap(df, path, figsize=figsize, dpi=dpi)
        saved_paths.append(path)
    except Exception as e:
        logger.error(f"Failed to create language heatmap: {e}")

    # 5. Attack category distribution (if available)
    try:
        if "attack_category" in df.columns and df["attack_category"].notna().any():
            path = output_dir / "attack_category_distribution.png"
            plot_attack_category_distribution(df, path, figsize=figsize, dpi=dpi)
            saved_paths.append(path)
    except Exception as e:
        logger.error(f"Failed to create attack category plot: {e}")

    logger.info(f"Created {len(saved_paths)} plots in {output_dir}")
    return saved_paths
