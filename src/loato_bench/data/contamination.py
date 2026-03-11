"""Split contamination checks: lexical (Jaccard) and semantic (cosine).

Validates that train/test splits are free of near-duplicate or paraphrased
samples that could inflate classifier performance.  Runs against materialized
parquet split pairs produced by LOATO-2A-06.

Two checks per split pair:

1. **Lexical** — MinHash LSH with word-level Jaccard ≥ 0.8
2. **Semantic** — cosine similarity ≥ 0.95 using all-MiniLM-L6-v2 (K=5 NN)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
from typing import Any
import unicodedata

from datasketch import MinHash, MinHashLSH
import numpy as np
from numpy.typing import NDArray
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers (mirrors harmonize.py but operates on raw strings)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """NFC unicode, collapse whitespace, strip."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_tokens(text: str) -> set[str]:
    """Lower-cased word token set for Jaccard computation."""
    return set(text.lower().split())


def _word_ngrams(text: str, n: int = 5) -> list[str]:
    """Word n-grams for MinHash (same shingle size as harmonize.py)."""
    words = text.lower().split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _text_id(text: str) -> str:
    """Stable 12-char hex ID from normalized text (for CSV output)."""
    norm = _normalize(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Lexical contamination (MinHash LSH)
# ---------------------------------------------------------------------------


def lexical_check(
    train_texts: list[str],
    test_texts: list[str],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> list[dict[str, Any]]:
    """Find near-duplicate test↔train pairs via MinHash LSH.

    Inserts all *train_texts* into an LSH index, then queries each
    *test_texts* entry.  Confirmed matches (exact Jaccard ≥ threshold)
    are returned.

    Parameters
    ----------
    train_texts : list[str]
        Training set texts.
    test_texts : list[str]
        Test set texts.
    threshold : float
        Jaccard similarity threshold (default 0.8).
    num_perm : int
        MinHash permutation count.

    Returns
    -------
    list[dict]
        Each dict: ``{test_idx, train_idx, jaccard}``.
    """
    if not train_texts or not test_texts:
        return []

    # Build LSH index from train texts
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    train_minhashes: list[MinHash] = []
    train_token_sets: list[set[str]] = []

    for i, text in enumerate(train_texts):
        norm = _normalize(text)
        tokens = _word_tokens(norm)

        mh = MinHash(num_perm=num_perm)
        for token in tokens:
            mh.update(token.encode("utf-8"))

        train_minhashes.append(mh)
        train_token_sets.append(tokens)

        try:
            lsh.insert(str(i), mh)
        except ValueError:
            # Duplicate MinHash — skip (identical train text)
            pass

    # Query each test text
    flags: list[dict[str, Any]] = []
    for ti, text in enumerate(test_texts):
        norm = _normalize(text)
        test_tokens = _word_tokens(norm)

        mh = MinHash(num_perm=num_perm)
        for token in test_tokens:
            mh.update(token.encode("utf-8"))

        candidates = lsh.query(mh)
        for cand_key in candidates:
            cand_idx = int(cand_key)
            # Compute exact Jaccard to confirm
            intersection = len(test_tokens & train_token_sets[cand_idx])
            union = len(test_tokens | train_token_sets[cand_idx])
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= threshold:
                flags.append(
                    {
                        "test_idx": ti,
                        "train_idx": cand_idx,
                        "jaccard": round(jaccard, 4),
                    }
                )

    return flags


# ---------------------------------------------------------------------------
# Semantic contamination (cosine KNN)
# ---------------------------------------------------------------------------


def _knn_max_similarities(
    train_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
    k: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Compute max cosine similarity of each test vector to K nearest train neighbors.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(max_sims, best_train_indices)`` — both shape ``(n_test,)``.
    """
    from sklearn.neighbors import NearestNeighbors

    actual_k = min(k, train_embeddings.shape[0])
    nn = NearestNeighbors(n_neighbors=actual_k, metric="cosine", algorithm="brute")
    nn.fit(train_embeddings)

    distances, indices = nn.kneighbors(test_embeddings)
    similarities = 1.0 - distances  # cosine distance → cosine similarity

    best_per_row = np.argmax(similarities, axis=1)
    max_sims = similarities[np.arange(len(similarities)), best_per_row]
    best_indices = indices[np.arange(len(indices)), best_per_row]

    return max_sims, best_indices


def semantic_check(
    train_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
    k: int = 5,
    threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], NDArray[np.float64]]:
    """Find semantically similar test↔train pairs via cosine KNN.

    For each test vector, finds *k* nearest train neighbors.  Flags
    test samples where max cosine similarity ≥ *threshold*.

    Parameters
    ----------
    train_embeddings : NDArray
        Shape ``(n_train, dim)``.
    test_embeddings : NDArray
        Shape ``(n_test, dim)``.
    k : int
        Number of nearest neighbors to check.
    threshold : float
        Cosine similarity threshold (default 0.95).

    Returns
    -------
    tuple[list[dict], NDArray]
        ``(flags, max_sims)`` where flags is a list of dicts with
        ``{test_idx, train_idx, cosine_sim}`` and max_sims is the
        per-test-sample max cosine similarity (shape ``(n_test,)``).
    """
    if train_embeddings.shape[0] == 0 or test_embeddings.shape[0] == 0:
        return [], np.array([], dtype=np.float64)

    max_sims, best_indices = _knn_max_similarities(train_embeddings, test_embeddings, k)

    flags: list[dict[str, Any]] = []
    for ti in range(test_embeddings.shape[0]):
        if max_sims[ti] >= threshold:
            flags.append(
                {
                    "test_idx": ti,
                    "train_idx": int(best_indices[ti]),
                    "cosine_sim": round(float(max_sims[ti]), 6),
                }
            )

    return flags, max_sims


# ---------------------------------------------------------------------------
# Per-split contamination check
# ---------------------------------------------------------------------------


def check_split_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_id: str,
    train_embeddings: NDArray[np.float32] | None = None,
    test_embeddings: NDArray[np.float32] | None = None,
    jaccard_threshold: float = 0.8,
    cosine_threshold: float = 0.95,
    k: int = 5,
) -> dict[str, Any]:
    """Run lexical + semantic contamination checks on one split pair.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (must have ``text`` column).
    test_df : pd.DataFrame
        Test split (must have ``text`` column).
    split_id : str
        Identifier for this split (e.g. ``"standard_cv/fold_0"``).
    train_embeddings : NDArray | None
        Pre-computed train embeddings.  If ``None``, semantic check is skipped.
    test_embeddings : NDArray | None
        Pre-computed test embeddings.  If ``None``, semantic check is skipped.
    jaccard_threshold : float
        Lexical Jaccard threshold.
    cosine_threshold : float
        Semantic cosine threshold.
    k : int
        Number of nearest neighbors for semantic check.

    Returns
    -------
    dict
        Report entry with flag counts, rates, pass/fail, and flag details.
    """
    n_test = len(test_df)
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    # Lexical check
    logger.info("  [%s] Lexical check (%d train × %d test)...", split_id, len(train_texts), n_test)
    lex_flags = lexical_check(train_texts, test_texts, threshold=jaccard_threshold)

    # Deduplicate by test_idx (keep highest score)
    lex_by_test: dict[int, dict[str, Any]] = {}
    for f in lex_flags:
        ti = f["test_idx"]
        if ti not in lex_by_test or f["jaccard"] > lex_by_test[ti]["jaccard"]:
            lex_by_test[ti] = f
    lex_unique = list(lex_by_test.values())

    # Semantic check
    sem_unique: list[dict[str, Any]] = []
    max_sims: NDArray[np.float64] | None = None
    if train_embeddings is not None and test_embeddings is not None:
        logger.info(
            "  [%s] Semantic check (K=%d, threshold=%.2f)...", split_id, k, cosine_threshold
        )
        sem_flags, max_sims = semantic_check(
            train_embeddings, test_embeddings, k=k, threshold=cosine_threshold
        )
        sem_by_test: dict[int, dict[str, Any]] = {}
        for f in sem_flags:
            ti = f["test_idx"]
            if ti not in sem_by_test or f["cosine_sim"] > sem_by_test[ti]["cosine_sim"]:
                sem_by_test[ti] = f
        sem_unique = list(sem_by_test.values())

    # Combine unique flagged test indices
    all_flagged_test_ids = set(f["test_idx"] for f in lex_unique) | set(
        f["test_idx"] for f in sem_unique
    )
    flagged_rate = len(all_flagged_test_ids) / n_test if n_test > 0 else 0.0
    passed = flagged_rate <= 0.01

    # Build flag records for CSV
    flag_records: list[dict[str, Any]] = []
    for f in lex_unique:
        ti = f["test_idx"]
        flag_records.append(
            {
                "split_id": split_id,
                "test_idx": ti,
                "sample_id": _text_id(test_texts[ti]),
                "check_type": "lexical",
                "score": f["jaccard"],
                "nearest_neighbor_idx": f["train_idx"],
                "nearest_neighbor_id": _text_id(train_texts[f["train_idx"]]),
            }
        )
    for f in sem_unique:
        ti = f["test_idx"]
        flag_records.append(
            {
                "split_id": split_id,
                "test_idx": ti,
                "sample_id": _text_id(test_texts[ti]),
                "check_type": "semantic",
                "score": f["cosine_sim"],
                "nearest_neighbor_idx": f["train_idx"],
                "nearest_neighbor_id": _text_id(train_texts[f["train_idx"]]),
            }
        )

    # Worst offenders for report
    worst_lex = sorted(lex_unique, key=lambda x: x["jaccard"], reverse=True)[:3]
    worst_sem = sorted(sem_unique, key=lambda x: x["cosine_sim"], reverse=True)[:3]

    # Template homogeneity score: mean max cosine similarity across all test samples.
    # Higher = more cross-set templating. Predicts ΔF1 gap in LOATO analysis.
    homogeneity_score: float | None = None
    if max_sims is not None and len(max_sims) > 0:
        homogeneity_score = round(float(np.mean(max_sims)), 6)

    report_entry: dict[str, Any] = {
        "split_id": split_id,
        "n_test": n_test,
        "n_train": len(train_df),
        "lexical_flagged": len(lex_unique),
        "semantic_flagged": len(sem_unique),
        "total_flagged": len(all_flagged_test_ids),
        "flagged_rate_pct": round(flagged_rate * 100, 4),
        "template_homogeneity_score": homogeneity_score,
        "pass": passed,
        "worst_lexical": [
            {"test_idx": w["test_idx"], "train_idx": w["train_idx"], "jaccard": w["jaccard"]}
            for w in worst_lex
        ],
        "worst_semantic": [
            {
                "test_idx": w["test_idx"],
                "train_idx": w["train_idx"],
                "cosine_sim": w["cosine_sim"],
            }
            for w in worst_sem
        ],
    }

    return {"report": report_entry, "flags": flag_records}


# ---------------------------------------------------------------------------
# Batch: check all materialized splits
# ---------------------------------------------------------------------------


def discover_split_pairs(splits_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find all train/test parquet pairs under *splits_dir*.

    Returns
    -------
    list[tuple[str, Path, Path]]
        ``(split_id, train_path, test_path)`` sorted by split_id.
    """
    pairs: list[tuple[str, Path, Path]] = []
    for train_path in sorted(splits_dir.rglob("train.parquet")):
        test_path = train_path.parent / "test.parquet"
        if test_path.exists():
            rel = str(train_path.parent.relative_to(splits_dir))
            pairs.append((rel, train_path, test_path))
    return pairs


def embed_texts_minilm(texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
    """Embed texts using all-MiniLM-L6-v2 (contamination check only).

    Parameters
    ----------
    texts : list[str]
        Raw texts to embed.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    NDArray
        Shape ``(len(texts), 384)``.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings: NDArray[np.float32] = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def check_all_splits(
    splits_dir: Path,
    jaccard_threshold: float = 0.8,
    cosine_threshold: float = 0.95,
    k: int = 5,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run contamination checks on all split pairs found under *splits_dir*.

    Embeds all unique texts once using MiniLM, then checks each split pair.

    Parameters
    ----------
    splits_dir : Path
        Root directory containing split subdirectories.
    jaccard_threshold : float
        Lexical Jaccard threshold.
    cosine_threshold : float
        Semantic cosine threshold.
    k : int
        Nearest neighbors for semantic check.

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        ``(report_entries, flags_df)`` where flags_df has columns:
        ``split_id, test_idx, sample_id, check_type, score,
        nearest_neighbor_idx, nearest_neighbor_id``.
    """
    pairs = discover_split_pairs(splits_dir)
    if not pairs:
        logger.warning("No split pairs found in %s", splits_dir)
        return [], pd.DataFrame()

    logger.info("Found %d split pairs to check", len(pairs))

    # Collect all unique texts for a single embedding pass
    all_texts_set: dict[str, int] = {}  # text -> index
    split_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    for split_id, train_path, test_path in pairs:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        split_data.append((split_id, train_df, test_df))

        for text in train_df["text"]:
            if text not in all_texts_set:
                all_texts_set[text] = len(all_texts_set)
        for text in test_df["text"]:
            if text not in all_texts_set:
                all_texts_set[text] = len(all_texts_set)

    unique_texts = [""] * len(all_texts_set)
    for text, idx in all_texts_set.items():
        unique_texts[idx] = text

    logger.info("Embedding %d unique texts with MiniLM...", len(unique_texts))
    all_embeddings = embed_texts_minilm(unique_texts)

    # Check each split pair
    report_entries: list[dict[str, Any]] = []
    all_flags: list[dict[str, Any]] = []

    for split_id, train_df, test_df in split_data:
        logger.info("Checking %s (%d train, %d test)...", split_id, len(train_df), len(test_df))

        # Look up embeddings
        train_indices = np.array([all_texts_set[t] for t in train_df["text"]])
        test_indices = np.array([all_texts_set[t] for t in test_df["text"]])
        train_emb = all_embeddings[train_indices]
        test_emb = all_embeddings[test_indices]

        result = check_split_pair(
            train_df,
            test_df,
            split_id,
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            jaccard_threshold=jaccard_threshold,
            cosine_threshold=cosine_threshold,
            k=k,
        )
        report_entries.append(result["report"])
        all_flags.extend(result["flags"])

    flags_df = (
        pd.DataFrame(all_flags)
        if all_flags
        else pd.DataFrame(
            columns=[
                "split_id",
                "test_idx",
                "sample_id",
                "check_type",
                "score",
                "nearest_neighbor_idx",
                "nearest_neighbor_id",
            ]
        )
    )

    return report_entries, flags_df
