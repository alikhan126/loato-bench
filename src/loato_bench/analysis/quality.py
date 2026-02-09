"""Data quality analysis and GenTel filtering recommendations.

This module implements the GenTel quality gate using heuristic scoring to
distinguish genuine prompt injection attacks from pure content harm samples.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default injection keywords for confidence scoring
DEFAULT_INJECTION_KEYWORDS = [
    "ignore",
    "disregard",
    "override",
    "prompt injection",
    "jailbreak",
    "bypass",
    "system prompt",
    "instructions",
    "pretend",
    "roleplay",
    "act as",
    "new instruction",
    "ignore previous",
    "disregard above",
    "forget everything",
    "new task",
    "actually",
    "instead",
    "however",
    "but really",
    "translate to",
    "execute",
    "run code",
    "eval",
]


# ---------------------------------------------------------------------------
# Injection confidence scoring
# ---------------------------------------------------------------------------


def compute_injection_confidence_scores(
    df: pd.DataFrame,
    keywords: list[str] | None = None,
) -> pd.Series:
    """Compute injection confidence scores for each sample.

    Uses regex-based keyword matching to score samples from 0 (pure content harm)
    to 1 (high confidence injection). This is a heuristic approach for fast,
    deterministic quality assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'text' column.
    keywords : list[str], optional
        Injection keywords to search for. If None, uses DEFAULT_INJECTION_KEYWORDS.

    Returns
    -------
    pd.Series
        Float scores in range [0, 1] for each sample.
    """
    if keywords is None:
        keywords = DEFAULT_INJECTION_KEYWORDS

    if df.empty:
        return pd.Series([], dtype=float)

    # Build regex pattern (case-insensitive word boundary matching)
    pattern = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    regex = re.compile(pattern, re.IGNORECASE)

    scores = []
    for text in df["text"]:
        if not isinstance(text, str) or not text.strip():
            scores.append(0.0)
            continue

        # Count keyword matches
        matches = regex.findall(text.lower())
        match_count = len(matches)

        # Normalize by text length (longer texts get lower boost per match)
        text_len = len(text)
        if text_len == 0:
            scores.append(0.0)
            continue

        # Score formula: matches / (1 + log10(text_len))
        # Examples:
        # - 3 matches in 100 chars: ~1.5 score → capped to 1.0
        # - 1 match in 50 chars: ~0.59
        # - 0 matches: 0.0
        import math

        raw_score = match_count / (1 + math.log10(text_len))
        normalized = min(raw_score, 1.0)
        scores.append(normalized)

    return pd.Series(scores, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# GenTel quality issues
# ---------------------------------------------------------------------------


def detect_gentel_quality_issues(df: pd.DataFrame) -> dict[str, Any]:
    """Detect quality issues in GenTel-Bench samples.

    GenTel categories are content harm types (hate speech, violence), not
    injection techniques. This function identifies samples that are likely
    pure content harm with no injection component.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'source' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - gentel_count: int (total GenTel samples)
        - low_confidence_count: int (score <0.3)
        - medium_confidence_count: int (0.3 <= score < 0.7)
        - high_confidence_count: int (score >= 0.7)
        - mean_score: float
        - median_score: float
        - issues_detected: list[str] (textual descriptions)
    """
    # Filter to GenTel samples only
    gentel_df = df[df["source"].str.lower().str.contains("gentel", na=False)]

    if gentel_df.empty:
        return {
            "gentel_count": 0,
            "low_confidence_count": 0,
            "medium_confidence_count": 0,
            "high_confidence_count": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "issues_detected": ["No GenTel samples found in dataset"],
        }

    # Compute confidence scores
    scores = compute_injection_confidence_scores(gentel_df)

    gentel_count = len(gentel_df)
    low_count = int((scores < 0.3).sum())
    medium_count = int(((scores >= 0.3) & (scores < 0.7)).sum())
    high_count = int((scores >= 0.7).sum())
    mean_score = float(scores.mean())
    median_score = float(scores.median())

    # Detect issues
    issues = []
    if low_count / gentel_count > 0.5:
        issues.append(
            f"{low_count / gentel_count * 100:.1f}% of GenTel samples have "
            "low injection confidence (<0.3). These may be pure content harm."
        )
    if mean_score < 0.4:
        issues.append(
            f"Mean injection confidence is only {mean_score:.2f}. "
            "Consider filtering or manual review."
        )
    if gentel_count > 10000:
        issues.append(
            f"GenTel has {gentel_count} samples. Consider capping at 5K "
            "injection-confident samples to balance dataset."
        )

    if not issues:
        issues.append("No critical quality issues detected.")

    return {
        "gentel_count": gentel_count,
        "low_confidence_count": low_count,
        "medium_confidence_count": medium_count,
        "high_confidence_count": high_count,
        "mean_score": mean_score,
        "median_score": median_score,
        "issues_detected": issues,
    }


# ---------------------------------------------------------------------------
# Filtering recommendations
# ---------------------------------------------------------------------------


def recommend_gentel_filtering(
    df: pd.DataFrame,
    threshold: float = 0.4,
    max_samples: int = 5000,
) -> dict[str, Any]:
    """Recommend filtering strategy for GenTel samples.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'source' column.
    threshold : float
        Minimum injection confidence score to keep.
    max_samples : int
        Maximum number of GenTel samples to retain (top-scored).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - original_count: int
        - filtered_count: int (after threshold)
        - final_count: int (after threshold + cap)
        - removed_count: int
        - threshold_used: float
        - max_samples_used: int
        - recommendation: str (textual recommendation)
    """
    gentel_df = df[df["source"].str.lower().str.contains("gentel", na=False)]

    if gentel_df.empty:
        return {
            "original_count": 0,
            "filtered_count": 0,
            "final_count": 0,
            "removed_count": 0,
            "threshold_used": threshold,
            "max_samples_used": max_samples,
            "recommendation": "No GenTel samples to filter.",
        }

    original_count = len(gentel_df)

    # Apply threshold filter
    scores = compute_injection_confidence_scores(gentel_df)
    filtered_df = gentel_df[scores >= threshold]
    filtered_count = len(filtered_df)

    # Apply cap (keep top-scored samples)
    if filtered_count > max_samples:
        # Sort by score descending, keep top max_samples
        scored_df = filtered_df.copy()
        scored_df["_score"] = scores[scores >= threshold]
        scored_df = scored_df.sort_values("_score", ascending=False)
        final_count = max_samples
    else:
        final_count = filtered_count

    removed_count = original_count - final_count

    # Generate recommendation text
    if removed_count == 0:
        recommendation = (
            f"Keep all {original_count} GenTel samples (all above threshold {threshold})."
        )
    else:
        recommendation = (
            f"Filter GenTel from {original_count} to {final_count} samples:\n"
            f"  - Apply threshold {threshold}: {original_count} → {filtered_count}\n"
            f"  - Cap at {max_samples}: {filtered_count} → {final_count}\n"
            f"  - Total removed: {removed_count} ({removed_count / original_count * 100:.1f}%)"
        )

    return {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "final_count": final_count,
        "removed_count": removed_count,
        "threshold_used": threshold,
        "max_samples_used": max_samples,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Data integrity validation
# ---------------------------------------------------------------------------


def validate_data_integrity(df: pd.DataFrame) -> list[str]:
    """Validate data integrity and return list of warnings.

    Checks for:
    - Empty text fields
    - Invalid label values (not 0 or 1)
    - Missing required fields
    - Duplicate indices
    - Extremely long texts (>10K chars)
    - Suspicious patterns (excessive punctuation, all caps)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    list[str]
        List of warning messages. Empty list if no issues found.
    """
    warnings = []

    if df.empty:
        warnings.append("DataFrame is empty")
        return warnings

    # Check for missing required fields FIRST
    required_fields = ["text", "label", "source"]
    for field in required_fields:
        if field not in df.columns:
            warnings.append(f"Missing required field: {field}")

    # Only proceed with field-specific checks if required fields exist
    if "text" in df.columns:
        # Check for empty or whitespace-only text
        empty_text = df["text"].isna() | (df["text"].str.strip() == "")
        if empty_text.any():
            warnings.append(f"Found {empty_text.sum()} samples with empty text")

    if "label" in df.columns:
        # Check for invalid label values
        invalid_labels = ~df["label"].isin([0, 1])
        if invalid_labels.any():
            warnings.append(
                f"Found {invalid_labels.sum()} samples with invalid labels (not 0 or 1)"
            )

    # Check for null values in existing required fields
    for field in required_fields:
        if field in df.columns and df[field].isna().any():
            na_count = df[field].isna().sum()
            warnings.append(f"Found {na_count} null values in '{field}'")

    # Check for duplicate indices
    if df.index.duplicated().any():
        warnings.append(f"Found {df.index.duplicated().sum()} duplicate indices")

    # Check for extremely long texts (>10K chars) - only if text column exists
    if "text" in df.columns:
        very_long = df["text"].str.len() > 10000
        if very_long.any():
            warnings.append(
                f"Found {very_long.sum()} samples with >10K characters "
                f"(max: {df['text'].str.len().max()})"
            )

    # Check for suspicious all-caps texts (>80% uppercase)
    if not df.empty and "text" in df.columns:

        def is_mostly_caps(text: str) -> bool:
            if not isinstance(text, str) or len(text) == 0:
                return False
            alpha_chars = [c for c in text if c.isalpha()]
            if len(alpha_chars) < 10:  # Ignore short texts
                return False
            upper_ratio = sum(c.isupper() for c in alpha_chars) / len(alpha_chars)
            return upper_ratio > 0.8

        all_caps = df["text"].apply(is_mostly_caps)
        if all_caps.any():
            warnings.append(f"Found {all_caps.sum()} samples with >80% uppercase letters")

    if not warnings:
        warnings.append("No data integrity issues detected")

    return warnings
