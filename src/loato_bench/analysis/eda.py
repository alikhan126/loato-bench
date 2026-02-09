"""Core exploratory data analysis functions for the unified dataset.

This module provides statistics and analysis functions that inform Sprint 2A
decisions (taxonomy mapping, split generation). All functions follow security
best practices: input validation, path traversal prevention, no eval/exec.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loato_bench.utils.config import DATA_DIR

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE_MB = 500
MAX_ROWS = 1_000_000


# ---------------------------------------------------------------------------
# Input validation and security
# ---------------------------------------------------------------------------


def load_parquet_safely(path: Path) -> pd.DataFrame:
    """Load Parquet file with security checks.

    Validates file size, prevents path traversal, checks row count and schema.

    Parameters
    ----------
    path : Path
        Path to Parquet file (must be within DATA_DIR).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If path is outside DATA_DIR, file too large, or invalid schema.
    FileNotFoundError
        If file doesn't exist.
    """
    # 1. Resolve path and prevent traversal outside DATA_DIR
    resolved_path = path.resolve()
    data_dir_resolved = DATA_DIR.resolve()

    if not str(resolved_path).startswith(str(data_dir_resolved)):
        raise ValueError(f"Security: Path {path} is outside DATA_DIR {DATA_DIR}")

    if not resolved_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {resolved_path}")

    # 2. Check file size before loading (prevent DoS via large files)
    file_size_mb = resolved_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")

    # 3. Load with error handling (catch corrupted files)
    try:
        df = pd.read_parquet(resolved_path)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet file: {e}") from e

    # 4. Validate row count
    if len(df) > MAX_ROWS:
        raise ValueError(f"DataFrame too large: {len(df)} rows (max {MAX_ROWS})")

    # 5. Validate schema (required columns)
    required_columns = {
        "text",
        "label",
        "source",
        "attack_category",
        "original_category",
        "language",
        "is_indirect",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 6. Validate dtypes
    if not pd.api.types.is_string_dtype(df["text"]):
        raise ValueError("Column 'text' must be string dtype")
    if not pd.api.types.is_integer_dtype(df["label"]):
        raise ValueError("Column 'label' must be integer dtype")

    logger.info("Loaded %d samples from %s (%.1f MB)", len(df), resolved_path.name, file_size_mb)
    return df


def sanitize_text_for_display(text: str, max_len: int = 100) -> str:
    """Sanitize text for safe display in plots and reports.

    Removes invisible Unicode, escapes HTML, truncates length.

    Parameters
    ----------
    text : str
        Raw text (may contain malicious Unicode or HTML).
    max_len : int
        Maximum length after truncation.

    Returns
    -------
    str
        Sanitized text safe for display.
    """
    # 1. Remove zero-width characters and other invisible Unicode
    invisible_ranges = [
        (0x200B, 0x200F),  # Zero-width spaces, joiners
        (0x202A, 0x202E),  # Direction override characters
        (0x2060, 0x206F),  # Word joiners, invisible operators
    ]

    cleaned = []
    for char in text:
        code = ord(char)
        is_invisible = any(start <= code <= end for start, end in invisible_ranges)
        if not is_invisible:
            cleaned.append(char)

    result = "".join(cleaned)

    # 2. Replace control characters with space
    result = "".join(c if c.isprintable() or c.isspace() else " " for c in result)

    # 3. Truncate to max_len
    if len(result) > max_len:
        result = result[:max_len] + "..."

    # 4. Escape HTML entities (for HTML exports)
    result = (
        result.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

    return result


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------


def load_unified_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the unified harmonized dataset.

    Parameters
    ----------
    path : Path, optional
        Path to unified_dataset.parquet. If None, uses DATA_DIR/processed/.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with all UnifiedSample fields.

    Raises
    ------
    FileNotFoundError
        If the unified dataset doesn't exist.
    ValueError
        If the file is invalid or too large.
    """
    if path is None:
        path = DATA_DIR / "processed" / "unified_dataset.parquet"

    return load_parquet_safely(path)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def compute_dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate statistics for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - total_samples: int
        - num_sources: int
        - num_languages: int
        - class_balance: dict[int, int] (label counts)
        - sources: dict[str, int] (source counts)
        - languages: dict[str, int] (language counts)
        - indirect_count: int
        - attack_category_coverage: float (% non-null)
    """
    if df.empty:
        return {
            "total_samples": 0,
            "num_sources": 0,
            "num_languages": 0,
            "class_balance": {},
            "sources": {},
            "languages": {},
            "indirect_count": 0,
            "attack_category_coverage": 0.0,
        }

    # Basic counts
    total_samples = len(df)
    num_sources = df["source"].nunique()
    num_languages = df["language"].nunique()

    # Class balance (benign vs injection)
    class_balance = df["label"].value_counts().to_dict()

    # Source distribution
    sources = df["source"].value_counts().to_dict()

    # Language distribution
    languages = df["language"].value_counts().to_dict()

    # Indirect injection count
    indirect_count = int(df["is_indirect"].sum())

    # Attack category coverage (% with non-null attack_category)
    attack_category_coverage = float((df["attack_category"].notna()).sum() / total_samples * 100)

    return {
        "total_samples": total_samples,
        "num_sources": num_sources,
        "num_languages": num_languages,
        "class_balance": class_balance,
        "sources": sources,
        "languages": languages,
        "indirect_count": indirect_count,
        "attack_category_coverage": attack_category_coverage,
    }


def analyze_text_properties(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze text length and vocabulary properties.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with 'text' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - char_lengths: dict (min, max, mean, median, std)
        - word_lengths: dict (min, max, mean, median, std)
        - outliers_short: int (texts <10 chars)
        - outliers_long: int (texts >5000 chars)
        - empty_count: int
    """
    if df.empty:
        return {
            "char_lengths": {},
            "word_lengths": {},
            "outliers_short": 0,
            "outliers_long": 0,
            "empty_count": 0,
        }

    # Character lengths
    char_lens = np.array(df["text"].str.len(), dtype=np.int64)
    char_stats = {
        "min": int(char_lens.min()),
        "max": int(char_lens.max()),
        "mean": float(char_lens.mean()),
        "median": float(np.median(char_lens)),
        "std": float(char_lens.std()),
    }

    # Word lengths (split on whitespace)
    word_lens = np.array(df["text"].str.split().str.len().fillna(0), dtype=np.int64)
    word_stats = {
        "min": int(word_lens.min()),
        "max": int(word_lens.max()),
        "mean": float(word_lens.mean()),
        "median": float(np.median(word_lens)),
        "std": float(word_lens.std()),
    }

    # Outliers
    outliers_short = int((char_lens < 10).sum())
    outliers_long = int((char_lens > 5000).sum())
    empty_count = int((df["text"].str.strip() == "").sum())

    return {
        "char_lengths": char_stats,
        "word_lengths": word_stats,
        "outliers_short": outliers_short,
        "outliers_long": outliers_long,
        "empty_count": empty_count,
    }


def analyze_label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze label (benign vs injection) distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with 'label' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - counts: dict[int, int] (label counts)
        - percentages: dict[int, float] (label percentages)
        - balance_ratio: float (minority / majority)
    """
    if df.empty:
        return {
            "counts": {},
            "percentages": {},
            "balance_ratio": 0.0,
        }

    counts = df["label"].value_counts().to_dict()
    total = len(df)
    percentages = {k: float(v / total * 100) for k, v in counts.items()}

    # Balance ratio (minority / majority)
    if len(counts) >= 2:
        minority = min(counts.values())
        majority = max(counts.values())
        balance_ratio = float(minority / majority)
    else:
        balance_ratio = 1.0

    return {
        "counts": counts,
        "percentages": percentages,
        "balance_ratio": balance_ratio,
    }


def analyze_source_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze sample distribution across sources.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with 'source' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - counts: dict[str, int] (source counts)
        - percentages: dict[str, float] (source percentages)
        - num_sources: int
    """
    if df.empty:
        return {
            "counts": {},
            "percentages": {},
            "num_sources": 0,
        }

    counts = df["source"].value_counts().to_dict()
    total = len(df)
    percentages = {k: float(v / total * 100) for k, v in counts.items()}

    return {
        "counts": counts,
        "percentages": percentages,
        "num_sources": len(counts),
    }


def analyze_language_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze language distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with 'language' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - counts: dict[str, int] (language counts)
        - percentages: dict[str, float] (language percentages)
        - num_languages: int
        - english_percentage: float
        - non_english_count: int
    """
    if df.empty:
        return {
            "counts": {},
            "percentages": {},
            "num_languages": 0,
            "english_percentage": 0.0,
            "non_english_count": 0,
        }

    counts = df["language"].value_counts().to_dict()
    total = len(df)
    percentages = {k: float(v / total * 100) for k, v in counts.items()}

    english_count = counts.get("en", 0)
    english_percentage = float(english_count / total * 100)
    non_english_count = total - english_count

    return {
        "counts": counts,
        "percentages": percentages,
        "num_languages": len(counts),
        "english_percentage": english_percentage,
        "non_english_count": non_english_count,
    }
