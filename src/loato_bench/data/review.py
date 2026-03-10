"""Manual review & spot-check module (LOATO-2A-03).

Provides export/import tooling for human spot-checking of LLM-assigned labels
and application of manual overrides back to the labeled dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from loato_bench.data.taxonomy import _text_hash
from loato_bench.data.taxonomy_spec import CATEGORY_ID_TO_SLUG, VALID_SLUGS

logger = logging.getLogger(__name__)

# Label sources that represent confident LLM labeling
LLM_LABEL_SOURCES: frozenset[str] = frozenset({"llm", "gpt_4_1_mini"})

# Columns exported in spot-check and uncertain CSVs
EXPORT_COLUMNS: list[str] = [
    "sample_hash",
    "text",
    "source",
    "attack_category",
    "confidence",
    "label_source",
    "correct_category",
]


def export_spot_check_samples(
    df: pd.DataFrame,
    n_per_category: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Export a stratified sample of confident LLM-labeled rows for spot-checking.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataset (must contain ``label_source``, ``attack_category``, ``text``).
    n_per_category : int
        Maximum samples to draw per attack category.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Spot-check DataFrame with columns per :data:`EXPORT_COLUMNS`.
    """
    # Filter to confident LLM pool (not tier1_2, not uncertain, not None)
    llm_mask = df["label_source"].isin(LLM_LABEL_SOURCES)
    llm_pool = df[llm_mask].copy()

    if llm_pool.empty:
        logger.warning("No confident LLM-labeled samples found")
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    # Stratified sample by attack_category
    parts = []
    for _cat, group in llm_pool.groupby("attack_category"):
        n = min(n_per_category, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)

    # Build export frame
    result = pd.DataFrame()
    result["sample_hash"] = sampled["text"].apply(_text_hash)
    result["text"] = sampled["text"].str[:500]
    result["source"] = sampled["source"].values
    result["attack_category"] = sampled["attack_category"].values
    result["confidence"] = sampled["confidence"].values
    result["label_source"] = sampled["label_source"].values
    result["correct_category"] = ""

    logger.info(
        "Exported %d spot-check samples across %d categories",
        len(result),
        result["attack_category"].nunique(),
    )
    return result.reset_index(drop=True)


def export_uncertain_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Export all uncertain-labeled rows for human review.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataset (must contain ``label_source``, ``text``).

    Returns
    -------
    pd.DataFrame
        Uncertain pool DataFrame with columns per :data:`EXPORT_COLUMNS`.
    """
    uncertain_mask = df["label_source"] == "uncertain"
    uncertain = df[uncertain_mask].copy()

    if uncertain.empty:
        logger.warning("No uncertain samples found")
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    result = pd.DataFrame()
    result["sample_hash"] = uncertain["text"].apply(_text_hash)
    result["text"] = uncertain["text"].str[:500]
    result["source"] = uncertain["source"].values
    result["attack_category"] = uncertain["attack_category"].values
    if "confidence" in uncertain.columns:
        result["confidence"] = uncertain["confidence"].values
    else:
        result["confidence"] = None
    result["label_source"] = uncertain["label_source"].values
    result["correct_category"] = ""

    logger.info("Exported %d uncertain samples", len(result))
    return result.reset_index(drop=True)


def load_manual_overrides(path: Path) -> pd.DataFrame:
    """Load and validate a manual overrides CSV.

    The CSV must have ``sample_hash`` and ``correct_category`` columns.
    Category IDs (e.g. ``"C1"``) are normalized to slugs.

    Parameters
    ----------
    path : Path
        Path to the overrides CSV file.

    Returns
    -------
    pd.DataFrame
        Validated overrides with ``sample_hash`` and ``correct_category``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or unknown categories are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Overrides file not found: {path}")

    overrides = pd.read_csv(path)

    # Validate required columns
    required = {"sample_hash", "correct_category"}
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to rows with non-empty correct_category
    has_value = overrides["correct_category"].notna() & (overrides["correct_category"] != "")
    overrides = overrides[has_value]
    overrides = overrides.copy()

    # Normalize C-IDs to slugs
    overrides["correct_category"] = overrides["correct_category"].apply(
        lambda x: CATEGORY_ID_TO_SLUG.get(str(x).strip(), str(x).strip())
    )

    # Validate all categories
    unknown = set(overrides["correct_category"]) - VALID_SLUGS
    if unknown:
        raise ValueError(
            f"Unknown categories in overrides: {unknown}. Valid: {sorted(VALID_SLUGS)}"
        )

    logger.info("Loaded %d manual overrides from %s", len(overrides), path)
    return overrides[["sample_hash", "correct_category"]].reset_index(drop=True)


def apply_manual_overrides(
    df: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """Apply manual overrides to the labeled dataset.

    Matches on ``sample_hash`` (computed from ``text``). Overridden rows get:
    - ``attack_category`` updated to the override value
    - ``label_source`` set to ``"manual"``
    - ``confidence`` set to ``1.0``

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataset with ``text``, ``attack_category``, ``label_source``,
        ``confidence`` columns.
    overrides : pd.DataFrame
        Must have ``sample_hash`` and ``correct_category`` columns.

    Returns
    -------
    pd.DataFrame
        Updated dataset (new copy).
    """
    df = df.copy()

    # Compute hashes for the dataset
    df["_hash"] = df["text"].apply(_text_hash)

    # Build override lookup
    override_map = dict(zip(overrides["sample_hash"], overrides["correct_category"]))

    matched = 0
    for idx in df.index:
        h = df.at[idx, "_hash"]
        if h in override_map:
            df.at[idx, "attack_category"] = override_map[h]
            df.at[idx, "label_source"] = "manual"
            df.at[idx, "confidence"] = 1.0
            matched += 1

    df = df.drop(columns=["_hash"])

    # Warn about remaining uncertain rows
    if "label_source" in df.columns:
        remaining_uncertain = (df["label_source"] == "uncertain").sum()
        if remaining_uncertain > 0:
            logger.warning(
                "%d uncertain samples remain after applying overrides", remaining_uncertain
            )

    logger.info("Applied %d manual overrides (%d in override file)", matched, len(overrides))
    return df


def compute_error_rates(
    spot_check: pd.DataFrame,
) -> dict[str, Any]:
    """Compute per-category error rates from a completed spot-check CSV.

    Compares ``attack_category`` (LLM-assigned) vs ``correct_category`` (human).
    Rows with blank ``correct_category`` are skipped.

    Parameters
    ----------
    spot_check : pd.DataFrame
        Completed spot-check DataFrame with ``attack_category`` and
        ``correct_category`` columns.

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``per_category``: dict mapping category → {total, errors, error_rate}
        - ``overall_error_rate``: float
        - ``high_error_categories``: list of categories with error_rate > 0.20
    """
    # Filter to reviewed rows
    reviewed = spot_check[
        spot_check["correct_category"].notna() & (spot_check["correct_category"] != "")
    ].copy()

    if reviewed.empty:
        return {
            "per_category": {},
            "overall_error_rate": 0.0,
            "high_error_categories": [],
        }

    # Compare per category
    per_category: dict[str, dict[str, Any]] = {}
    total_reviewed = 0
    total_errors = 0

    for cat, group in reviewed.groupby("attack_category"):
        n = len(group)
        errors = (group["attack_category"] != group["correct_category"]).sum()
        rate = float(errors / n) if n > 0 else 0.0
        per_category[str(cat)] = {
            "total": int(n),
            "errors": int(errors),
            "error_rate": round(rate, 4),
        }
        total_reviewed += n
        total_errors += errors

    overall_rate = float(total_errors / total_reviewed) if total_reviewed > 0 else 0.0
    high_error = [c for c, s in per_category.items() if s["error_rate"] > 0.20]

    return {
        "per_category": per_category,
        "overall_error_rate": round(overall_rate, 4),
        "high_error_categories": high_error,
    }


def generate_coverage_report_v2(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a v2 coverage report after manual overrides.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataset with ``label``, ``attack_category``, ``label_source`` columns.

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``total_injection``: int
        - ``per_category``: dict[str, int]
        - ``per_label_source``: dict[str, int]
        - ``coverage_pct``: float
        - ``meets_90_pct_threshold``: bool
    """
    injection = df[df["label"] == 1]
    total_inj = len(injection)

    mapped = injection[injection["attack_category"].notna()]
    coverage_pct = float(len(mapped) / total_inj * 100) if total_inj > 0 else 0.0

    per_category = injection["attack_category"].value_counts().to_dict()
    # Convert keys to str for JSON serialization
    per_category = {str(k): int(v) for k, v in per_category.items()}

    per_label_source: dict[str, int] = {}
    if "label_source" in df.columns:
        source_counts = injection["label_source"].value_counts()
        per_label_source = {str(k): int(v) for k, v in source_counts.items()}

    return {
        "total_injection": int(total_inj),
        "per_category": per_category,
        "per_label_source": per_label_source,
        "coverage_pct": round(coverage_pct, 2),
        "meets_90_pct_threshold": coverage_pct >= 90.0,
    }
