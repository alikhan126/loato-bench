"""Attack taxonomy mapping (Tier 1: source mapping, Tier 2: regex, Tier 3: LLM).

This module implements the 3-tier taxonomy mapping system:
- Tier 1: Direct source-to-category mappings (fast, deterministic)
- Tier 2: Regex-based heuristics (fast, testable)
- Tier 3: LLM-assisted classification (uses LLMProvider ABC)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from loato_bench.utils.config import CONFIGS_DIR

if TYPE_CHECKING:
    from loato_bench.llm.base import LLMProvider

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
# Tier 3: LLM-assisted classification
# ---------------------------------------------------------------------------


def build_tier3_prompt(
    text: str,
    categories: dict[str, Any],
) -> str:
    """Build classification prompt for a single sample.

    Parameters
    ----------
    text : str
        The sample text to classify.
    categories : dict[str, Any]
        Category definitions from taxonomy config (name → {description, ...}).

    Returns
    -------
    str
        The formatted user prompt for the LLM.
    """
    return f"Classify this text into one attack category.\n\nText: {text}"


def _build_tier3_system(categories: dict[str, Any]) -> str:
    """Build system message listing valid categories and descriptions."""
    lines = ["Classify the given text into exactly one attack category.", ""]
    lines.append("Valid categories:")
    for name, info in categories.items():
        desc = info.get("description", "")
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("Respond with ONLY the category name. Nothing else.")
    return "\n".join(lines)


def parse_tier3_response(
    response: str,
    valid_categories: list[str],
) -> str | None:
    """Parse LLM response into a valid category name or None.

    Parameters
    ----------
    response : str
        Raw LLM response text.
    valid_categories : list[str]
        List of acceptable category names.

    Returns
    -------
    str | None
        Matched category name (original casing from valid list) or None.
    """
    cleaned = response.strip().lower()
    lookup = {c.lower(): c for c in valid_categories}
    return lookup.get(cleaned)


def load_tier3_cache(cache_path: Path) -> dict[str, str]:
    """Load cached LLM classifications from JSON.

    Parameters
    ----------
    cache_path : Path
        Path to the JSON cache file.

    Returns
    -------
    dict[str, str]
        Mapping of text hash → category name.  Empty dict if file missing
        or corrupt.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt Tier 3 cache at %s — starting fresh", cache_path)
        return {}


def save_tier3_cache(cache: dict[str, str], cache_path: Path) -> None:
    """Save LLM classifications to JSON cache.

    Parameters
    ----------
    cache : dict[str, str]
        Mapping of text hash → category name.
    cache_path : Path
        Destination file path (parent dirs created automatically).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _text_hash(text: str) -> str:
    """Compute a short hash for cache keying."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def apply_tier3_llm_mapping(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    llm_provider: LLMProvider | None = None,
    max_calls: int = 2000,
    cache_path: Path | None = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """Apply Tier 3 LLM-assisted taxonomy mapping to unmapped samples.

    Only processes injection samples (label=1) with null ``attack_category``.
    Uses a JSON cache to avoid redundant API calls.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'text', 'label', and 'attack_category' columns.
    config : dict, optional
        Taxonomy config. If None, loads from default location.
    llm_provider : LLMProvider, optional
        LLM provider instance.  If None, creates one via ``get_llm_provider()``.
    max_calls : int
        Maximum number of LLM invocations to make.
    cache_path : Path, optional
        Path to JSON cache file.  If None, uses ``data/taxonomy_tier3_cache.json``.
    batch_size : int
        Reserved for future batched classification (currently unused).

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'attack_category' column.
    """
    if config is None:
        config = load_taxonomy_config()

    df = df.copy()

    categories = config.get("categories", {})
    valid_categories = list(categories.keys())

    if not valid_categories:
        logger.warning("No categories defined in config — skipping Tier 3")
        return df

    # Identify unmapped injection samples
    unmapped_mask = df["attack_category"].isna() & (df["label"] == 1)
    unmapped_indices = df[unmapped_mask].index.tolist()

    if not unmapped_indices:
        logger.info("No unmapped injection samples for Tier 3")
        return df

    # Load cache
    if cache_path is None:
        from loato_bench.utils.config import DATA_DIR

        cache_path = DATA_DIR / "taxonomy_tier3_cache.json"
    cache = load_tier3_cache(cache_path)

    # Lazy-init LLM provider
    if llm_provider is None:
        from loato_bench.llm import get_llm_provider

        llm_provider = get_llm_provider()

    system_msg = _build_tier3_system(categories)
    calls_made = 0
    mapped_count = 0

    for idx in unmapped_indices:
        text = df.at[idx, "text"]
        if not isinstance(text, str):
            continue

        th = _text_hash(text)

        # Check cache first
        if th in cache:
            category = parse_tier3_response(cache[th], valid_categories)
            if category is not None:
                df.at[idx, "attack_category"] = category
                mapped_count += 1
            continue

        # Respect max_calls
        if calls_made >= max_calls:
            break

        # Call LLM
        prompt = build_tier3_prompt(text, categories)
        response = llm_provider.complete(prompt, system=system_msg, max_tokens=50)
        calls_made += 1

        category = parse_tier3_response(response, valid_categories)
        if category is not None:
            df.at[idx, "attack_category"] = category
            cache[th] = category
            mapped_count += 1
        else:
            logger.debug("Tier 3 could not parse response %r for idx %d", response, idx)

    # Persist cache
    save_tier3_cache(cache, cache_path)

    logger.info("Tier 3 LLM mapped %d samples (%d API calls)", mapped_count, calls_made)
    return df


# ---------------------------------------------------------------------------
# Category merging
# ---------------------------------------------------------------------------

DEFAULT_MERGE_MAP: dict[str, str] = {
    "obfuscation_encoding": "instruction_override",
    "context_manipulation": "instruction_override",
    "information_extraction": "instruction_override",
}


def merge_small_categories(
    df: pd.DataFrame,
    merge_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Merge small attack categories into larger ones.

    Applied **after** Tier 3 LLM mapping, based on final category counts.
    Only merges ``attack_category`` values present in *merge_map*.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``attack_category`` column.
    merge_map : dict[str, str] | None
        Mapping of ``source_category → target_category``.
        If None, uses :data:`DEFAULT_MERGE_MAP`.

    Returns
    -------
    pd.DataFrame
        DataFrame with merged categories (new copy).
    """
    if merge_map is None:
        merge_map = DEFAULT_MERGE_MAP

    if not merge_map:
        return df.copy()

    df = df.copy()

    merged_count = 0
    for src, tgt in merge_map.items():
        mask = df["attack_category"] == src
        n = mask.sum()
        if n > 0:
            df.loc[mask, "attack_category"] = tgt
            merged_count += n
            logger.info("Merged %d samples: %s → %s", n, src, tgt)

    if merged_count:
        logger.info("Total merged: %d samples", merged_count)

    return df


# ---------------------------------------------------------------------------
# Full taxonomy pipeline (Tier 1 + 2, optionally Tier 3)
# ---------------------------------------------------------------------------


def apply_taxonomy_mapping(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    llm_provider: LLMProvider | None = None,
    apply_tier3: bool = False,
    max_calls: int = 2000,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Apply taxonomy mapping pipeline.

    Always applies Tier 1 + Tier 2.  Optionally applies Tier 3 LLM
    classification for remaining unmapped injection samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns.
    config : dict, optional
        Taxonomy config. If None, loads from default location.
    llm_provider : LLMProvider, optional
        LLM provider for Tier 3.  Ignored unless *apply_tier3* is True.
    apply_tier3 : bool
        Whether to run Tier 3 LLM classification.
    max_calls : int
        Max LLM calls for Tier 3.
    cache_path : Path, optional
        Tier 3 cache path.

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

    # Optionally apply Tier 3 (LLM)
    if apply_tier3:
        logger.info("Applying Tier 3 LLM classification")
        df = apply_tier3_llm_mapping(
            df,
            config=config,
            llm_provider=llm_provider,
            max_calls=max_calls,
            cache_path=cache_path,
        )

    # Compute coverage
    coverage = compute_category_coverage(df)
    logger.info(
        f"Taxonomy coverage: {coverage['coverage_percentage']:.1f}% "
        f"({coverage['mapped_count']} / {coverage['total_injection_samples']} samples)"
    )

    return df
