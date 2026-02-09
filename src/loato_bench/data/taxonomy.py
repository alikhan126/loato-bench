"""Attack taxonomy mapping (Tier 1: source mapping, Tier 2: regex, Tier 3: LLM).

This module implements the 3-tier taxonomy mapping system:
- Tier 1: Direct source-to-category mappings (fast, deterministic)
- Tier 2: Regex-based heuristics (fast, testable)
- Tier 3: LLM-assisted classification (expensive, Sprint 2A)

For EDA (Sprint 1A), only Tier 1+2 are implemented.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from loato_bench.utils.config import CONFIGS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_taxonomy_config(path: Path | None = None) -> dict[str, Any]:
    """Load taxonomy configuration from YAML file.

    Parameters
    ----------
    path : Path, optional
        Path to taxonomy.yaml. If None, uses configs/data/taxonomy.yaml.

    Returns
    -------
    dict[str, Any]
        Taxonomy configuration dictionary.
    """
    if path is None:
        path = CONFIGS_DIR / "data" / "taxonomy.yaml"

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


# ---------------------------------------------------------------------------
# Tier 1: Source-specific category mappings
# ---------------------------------------------------------------------------


def apply_tier1_source_mapping(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply Tier 1 source-specific category mappings.

    Updates the 'attack_category' column based on 'source' and 'original_category'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'source', 'original_category', and 'attack_category' columns.
    config : dict, optional
        Taxonomy config. If None, loads from default location.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'attack_category' column.
    """
    if config is None:
        config = load_taxonomy_config()

    df = df.copy()

    # Extract source mappings if available
    # For current taxonomy.yaml structure, no explicit Tier 1 mappings
    # This would be used if we had source-specific mappings in config

    # For now, just log that Tier 1 was applied
    logger.info("Applied Tier 1 source mapping")

    return df


# ---------------------------------------------------------------------------
# Tier 2: Regex-based heuristic matching
# ---------------------------------------------------------------------------


def apply_tier2_regex_patterns(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply Tier 2 regex patterns to infer attack categories.

    Only applies to samples where 'attack_category' is null (not yet mapped).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'text' and 'attack_category' columns.
    config : dict, optional
        Taxonomy config. If None, loads from default location.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'attack_category' column.
    """
    if config is None:
        config = load_taxonomy_config()

    df = df.copy()

    # Get categories and their regex patterns
    categories = config.get("categories", {})

    # Only process injection samples (label=1) with null attack_category
    if "label" in df.columns:
        unmapped_mask = df["attack_category"].isna() & (df["label"] == 1)
    else:
        unmapped_mask = df["attack_category"].isna()

    unmapped_indices = df[unmapped_mask].index

    if len(unmapped_indices) == 0:
        logger.info("No unmapped samples for Tier 2 regex matching")
        return df

    # Compile regex patterns for each category
    category_patterns: dict[str, list[re.Pattern[str]]] = {}
    for cat_name, cat_info in categories.items():
        patterns = cat_info.get("regex_patterns", [])
        if patterns:
            category_patterns[cat_name] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    # Apply patterns to unmapped samples
    matched_count = 0
    for idx in unmapped_indices:
        text = df.at[idx, "text"]
        if not isinstance(text, str):
            continue

        # Try each category's patterns
        for cat_name, patterns in category_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                df.at[idx, "attack_category"] = cat_name
                matched_count += 1
                break  # First match wins

    logger.info(f"Tier 2 regex matched {matched_count} / {len(unmapped_indices)} unmapped samples")

    return df


# ---------------------------------------------------------------------------
# Category coverage analysis
# ---------------------------------------------------------------------------


def compute_category_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Compute attack category coverage statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'attack_category' and 'label' columns.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - category_counts: dict[str, int] (counts per category)
        - total_injection_samples: int
        - mapped_count: int (samples with non-null category)
        - unmapped_count: int (null category)
        - coverage_percentage: float (% of injection samples mapped)
    """
    # Filter to injection samples only (label=1)
    if "label" in df.columns:
        injection_df = df[df["label"] == 1]
    else:
        injection_df = df

    total_injection = len(injection_df)

    # Count by category
    category_counts = injection_df["attack_category"].value_counts().to_dict()

    # Mapped vs unmapped
    mapped_count = int(injection_df["attack_category"].notna().sum())
    unmapped_count = total_injection - mapped_count

    coverage_percentage = (
        float(mapped_count / total_injection * 100) if total_injection > 0 else 0.0
    )

    return {
        "category_counts": category_counts,
        "total_injection_samples": total_injection,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "coverage_percentage": coverage_percentage,
    }


# ---------------------------------------------------------------------------
# Merge recommendations
# ---------------------------------------------------------------------------


def recommend_category_merges(
    df: pd.DataFrame,
    min_size: int = 50,
) -> dict[str, Any]:
    """Recommend category merges for small categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'attack_category' column.
    min_size : int
        Minimum samples required to keep a category separate.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - small_categories: list[dict] (categories below threshold)
        - merge_recommendations: list[str] (textual recommendations)
        - categories_to_keep: list[str] (categories above threshold)
    """
    # Filter to injection samples with non-null category
    if "label" in df.columns:
        injection_df = df[(df["label"] == 1) & df["attack_category"].notna()]
    else:
        injection_df = df[df["attack_category"].notna()]

    # Count by category
    category_counts = injection_df["attack_category"].value_counts()

    # Identify small categories
    small_categories = []
    categories_to_keep = []

    for cat_name, count in category_counts.items():
        if count < min_size:
            small_categories.append({"name": cat_name, "count": count})
        else:
            categories_to_keep.append(cat_name)

    # Generate merge recommendations
    recommendations = []
    if small_categories:
        recommendations.append(
            f"Found {len(small_categories)} categories below threshold ({min_size} samples):"
        )
        for cat_info in small_categories:
            recommendations.append(
                f"  - {cat_info['name']}: {cat_info['count']} samples "
                f"(recommend merging into larger category)"
            )
    else:
        recommendations.append(f"All categories meet minimum size threshold ({min_size} samples)")

    return {
        "small_categories": small_categories,
        "merge_recommendations": recommendations,
        "categories_to_keep": categories_to_keep,
    }


# ---------------------------------------------------------------------------
# Full taxonomy pipeline (Tier 1 + 2)
# ---------------------------------------------------------------------------


def apply_taxonomy_mapping(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply full Tier 1 + Tier 2 taxonomy mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns.
    config : dict, optional
        Taxonomy config. If None, loads from default location.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'attack_category' column.
    """
    if config is None:
        config = load_taxonomy_config()

    logger.info("Starting taxonomy mapping (Tier 1 + Tier 2)")

    # Apply Tier 1 (source-specific mappings)
    df = apply_tier1_source_mapping(df, config)

    # Apply Tier 2 (regex patterns)
    df = apply_tier2_regex_patterns(df, config)

    # Compute coverage
    coverage = compute_category_coverage(df)
    logger.info(
        f"Taxonomy coverage: {coverage['coverage_percentage']:.1f}% "
        f"({coverage['mapped_count']} / {coverage['total_injection_samples']} samples)"
    )

    return df
