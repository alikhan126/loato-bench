"""Data harmonization and deduplication pipeline.

Processes raw :class:`UnifiedSample` lists from all dataset loaders into a
clean, deduplicated pandas DataFrame ready for embedding and evaluation.

Pipeline steps:
1. Text normalization — NFC unicode, strip whitespace, collapse spaces
2. Exact dedup — SHA-256 on normalized text (keep first occurrence)
3. Near dedup — MinHash LSH, Jaccard threshold 0.85, word 3-grams
4. Language detection — ``langdetect`` on all samples
5. Output — pandas DataFrame (saved as parquet by CLI)
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import asdict

import pandas as pd
from datasketch import MinHash, MinHashLSH
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from promptguard.data.base import UnifiedSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text: NFC unicode, strip whitespace, collapse spaces.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Normalized text.
    """
    # NFC unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Replace any whitespace character (tab, newline, etc.) with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Exact deduplication (SHA-256)
# ---------------------------------------------------------------------------


def exact_dedup(samples: list[UnifiedSample]) -> list[UnifiedSample]:
    """Remove exact duplicates based on SHA-256 hash of normalized text.

    Keeps the first occurrence when duplicates are found.

    Parameters
    ----------
    samples : list[UnifiedSample]
        Input samples (may contain duplicates).

    Returns
    -------
    list[UnifiedSample]
        Deduplicated samples.
    """
    if not samples:
        return []

    seen: set[str] = set()
    result: list[UnifiedSample] = []
    for sample in samples:
        normalized = normalize_text(sample.text)
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if text_hash not in seen:
            seen.add(text_hash)
            result.append(sample)

    removed = len(samples) - len(result)
    if removed > 0:
        logger.info(
            "Exact dedup removed %d duplicates (%d → %d)",
            removed,
            len(samples),
            len(result),
        )
    return result


# ---------------------------------------------------------------------------
# Near deduplication (MinHash LSH)
# ---------------------------------------------------------------------------


def _word_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def near_dedup(
    samples: list[UnifiedSample],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[UnifiedSample]:
    """Remove near-duplicates using MinHash LSH with Jaccard similarity.

    Uses word 3-grams as shingles for MinHash computation.

    Parameters
    ----------
    samples : list[UnifiedSample]
        Input samples (exact dupes should be removed first).
    threshold : float
        Jaccard similarity threshold for near-duplicate detection.
    num_perm : int
        Number of permutations for MinHash.

    Returns
    -------
    list[UnifiedSample]
        Samples with near-duplicates removed.
    """
    if len(samples) <= 1:
        return list(samples)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []
    keep_indices: list[int] = []

    for i, sample in enumerate(samples):
        normalized = normalize_text(sample.text)
        ngrams = _word_ngrams(normalized)

        mh = MinHash(num_perm=num_perm)
        for gram in ngrams:
            mh.update(gram.encode("utf-8"))
        minhashes.append(mh)

        # Check if this sample is a near-duplicate of an already-kept sample
        candidates = lsh.query(mh)
        if not candidates:
            lsh.insert(str(i), mh)
            keep_indices.append(i)

    result = [samples[i] for i in keep_indices]
    removed = len(samples) - len(result)
    if removed > 0:
        logger.info(
            "Near dedup removed %d samples (%d → %d)",
            removed,
            len(samples),
            len(result),
        )
    return result


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """Detect the language of a text string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        ISO 639-1 language code (e.g., "en", "fr") or "unknown".
    """
    if not text or not text.strip():
        return "unknown"
    try:
        result: str = detect(text)
        return result
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


def samples_to_dataframe(samples: list[UnifiedSample]) -> pd.DataFrame:
    """Convert a list of UnifiedSample to a pandas DataFrame.

    The ``metadata`` dict is excluded from the main columns — it can be
    stored separately or expanded as needed.

    Parameters
    ----------
    samples : list[UnifiedSample]
        Input samples.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample.
    """
    if not samples:
        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "source",
                "attack_category",
                "original_category",
                "language",
                "is_indirect",
            ]
        )

    records = []
    for s in samples:
        d = asdict(s)
        d.pop("metadata", None)
        records.append(d)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full harmonization pipeline
# ---------------------------------------------------------------------------


def harmonize_samples(
    samples: list[UnifiedSample],
    near_dedup_threshold: float = 0.85,
) -> pd.DataFrame:
    """Run the full harmonization pipeline on a list of UnifiedSample.

    Steps:
    1. Normalize text in-place
    2. Exact deduplication (SHA-256)
    3. Near deduplication (MinHash LSH)
    4. Language detection
    5. Convert to DataFrame

    Parameters
    ----------
    samples : list[UnifiedSample]
        Raw samples from all dataset loaders.
    near_dedup_threshold : float
        Jaccard threshold for near-duplicate removal.

    Returns
    -------
    pd.DataFrame
        Harmonized, deduplicated DataFrame.
    """
    if not samples:
        return samples_to_dataframe([])

    logger.info("Starting harmonization of %d samples", len(samples))

    # Step 1: Normalize text
    for s in samples:
        s.text = normalize_text(s.text)

    # Step 2: Exact dedup
    samples = exact_dedup(samples)

    # Step 3: Near dedup
    samples = near_dedup(samples, threshold=near_dedup_threshold)

    # Step 4: Language detection (update language field)
    logger.info("Running language detection on %d samples", len(samples))
    for s in samples:
        detected = detect_language(s.text)
        if detected != "unknown":
            s.language = detected

    # Step 5: Convert to DataFrame
    df = samples_to_dataframe(samples)
    logger.info(
        "Harmonization complete: %d samples, %d unique sources",
        len(df),
        df["source"].nunique(),
    )
    return df
