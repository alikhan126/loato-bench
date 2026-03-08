"""Split generation: standard CV, LOATO, direct/indirect, cross-lingual.

Generates JSON-serialisable index splits for all four experiment types.
Each split references rows in the unified DataFrame by integer index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from loato_bench.utils.config import DATA_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard CV
# ---------------------------------------------------------------------------


def generate_standard_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate stratified K-fold CV splits.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset.
    n_folds : int
        Number of folds.
    stratify_by : list[str] | None
        Columns to use for stratification.  ``None`` values in any column
        are replaced with ``"_unknown"`` so sklearn does not error.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``{experiment, n_folds, seed, folds: [{fold, train_indices, test_indices}]}``

    Raises
    ------
    ValueError
        If the DataFrame is empty or has fewer than *n_folds* samples.
    """
    if df.empty:
        raise ValueError("Cannot generate CV splits from an empty DataFrame")

    if len(df) < n_folds:
        raise ValueError(f"Need at least {n_folds} samples for {n_folds}-fold CV, got {len(df)}")

    if stratify_by is None:
        stratify_by = ["label"]

    # Build composite stratification key (handle nulls)
    strat_col = df[stratify_by].fillna("_unknown").astype(str).agg("_".join, axis=1)

    # Check we have at least 2 classes
    if strat_col.nunique() < 2:
        raise ValueError(
            "Stratified CV requires at least 2 distinct classes in stratify_by columns"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, strat_col)):
        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
            }
        )

    return {
        "experiment": "standard_cv",
        "n_folds": n_folds,
        "seed": seed,
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# LOATO (Leave-One-Attack-Type-Out)
# ---------------------------------------------------------------------------


def generate_loato_splits(
    df: pd.DataFrame,
    benign_test_fraction: float = 0.2,
    min_samples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate LOATO splits — one fold per viable attack category.

    For each category *K* with ``>= min_samples`` injection samples:

    - **Train**: all injection samples from K-1 categories + (1 - benign_test_fraction) benign
    - **Test**: all injection samples from category K + benign_test_fraction of benign

    Each fold uses a different seeded benign split to avoid test-set homogeneity.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with ``label`` and ``attack_category`` columns.
    benign_test_fraction : float
        Fraction of benign samples allocated to each test fold.
    min_samples : int
        Minimum injection samples required in a category to create a fold.
    seed : int
        Base random seed.

    Returns
    -------
    dict[str, Any]
        ``{experiment, seed, folds: [{fold_name, held_out_category, ...}]}``

    Raises
    ------
    ValueError
        If no category meets the *min_samples* threshold.
    """
    if df.empty:
        raise ValueError("Cannot generate LOATO splits from an empty DataFrame")

    injection_df = df[df["label"] == 1]
    benign_indices = df[df["label"] == 0].index.tolist()

    # Find viable categories
    cat_counts = injection_df["attack_category"].value_counts()
    viable: list[str] = [str(cat) for cat, count in cat_counts.items() if count >= min_samples]

    if not viable:
        raise ValueError(
            f"No attack category has >= {min_samples} samples. Counts: {cat_counts.to_dict()}"
        )

    folds = []
    for fold_idx, held_out in enumerate(sorted(viable)):
        rng = np.random.RandomState(seed + fold_idx)

        # Test injection: all samples from held-out category
        test_injection_idx = injection_df[
            injection_df["attack_category"] == held_out
        ].index.tolist()

        # Train injection: all other categories
        train_injection_idx = injection_df[
            injection_df["attack_category"] != held_out
        ].index.tolist()

        # Benign split
        n_benign_test = max(1, int(len(benign_indices) * benign_test_fraction))
        shuffled_benign = rng.permutation(benign_indices).tolist()
        benign_test_idx = shuffled_benign[:n_benign_test]
        benign_train_idx = shuffled_benign[n_benign_test:]

        train_indices = sorted(train_injection_idx + benign_train_idx)
        test_indices = sorted(test_injection_idx + benign_test_idx)

        train_categories = sorted(c for c in cat_counts.index if c != held_out and c in viable)

        folds.append(
            {
                "fold_name": f"held_out_{held_out}",
                "held_out_category": held_out,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_categories": train_categories,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
            }
        )

    return {
        "experiment": "loato",
        "seed": seed,
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# Direct → Indirect transfer
# ---------------------------------------------------------------------------


def generate_direct_indirect_split(
    df: pd.DataFrame,
    benign_test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train on direct injections + benign, test on indirect + benign sample.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with ``label`` and ``is_indirect`` columns.
    benign_test_fraction : float
        Fraction of benign samples in test set.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``{experiment, seed, train_indices, test_indices}``

    Raises
    ------
    ValueError
        If there are no indirect injection samples.
    """
    if df.empty:
        raise ValueError("Cannot generate direct/indirect split from an empty DataFrame")

    indirect_injection = df[(df["label"] == 1) & (df["is_indirect"] == True)]  # noqa: E712
    if indirect_injection.empty:
        raise ValueError("No indirect injection samples found in dataset")

    direct_injection = df[(df["label"] == 1) & (df["is_indirect"] != True)]  # noqa: E712
    benign_indices = df[df["label"] == 0].index.tolist()

    rng = np.random.RandomState(seed)
    n_benign_test = max(1, int(len(benign_indices) * benign_test_fraction))
    shuffled_benign = rng.permutation(benign_indices).tolist()
    benign_test_idx = shuffled_benign[:n_benign_test]
    benign_train_idx = shuffled_benign[n_benign_test:]

    train_indices = sorted(direct_injection.index.tolist() + benign_train_idx)
    test_indices = sorted(indirect_injection.index.tolist() + benign_test_idx)

    return {
        "experiment": "direct_indirect",
        "seed": seed,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


# ---------------------------------------------------------------------------
# Cross-lingual transfer
# ---------------------------------------------------------------------------


def generate_crosslingual_split(
    df: pd.DataFrame,
    benign_test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train on English samples, test on non-English + benign sample.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with ``language`` column.
    benign_test_fraction : float
        Fraction of benign samples in test set.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``{experiment, seed, train_indices, test_indices, test_languages}``

    Raises
    ------
    ValueError
        If there are no non-English samples.
    """
    if df.empty:
        raise ValueError("Cannot generate crosslingual split from an empty DataFrame")

    non_english = df[df["language"] != "en"]
    if non_english.empty:
        raise ValueError("No non-English samples found in dataset")

    english = df[df["language"] == "en"]

    # English benign split
    english_benign = english[english["label"] == 0].index.tolist()
    rng = np.random.RandomState(seed)
    n_benign_test = max(1, int(len(english_benign) * benign_test_fraction))
    shuffled_benign = rng.permutation(english_benign).tolist()
    benign_test_idx = shuffled_benign[:n_benign_test]
    benign_train_idx = shuffled_benign[n_benign_test:]

    # Train: all English samples except benign test portion
    english_injection = english[english["label"] == 1].index.tolist()
    train_indices = sorted(english_injection + benign_train_idx)

    # Test: all non-English + benign test sample
    test_indices = sorted(non_english.index.tolist() + benign_test_idx)

    test_languages = sorted(non_english["language"].unique().tolist())

    return {
        "experiment": "crosslingual",
        "seed": seed,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "test_languages": test_languages,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_splits(splits: dict[str, Any], path: Path) -> None:
    """Save splits to JSON file (indent=2 for readability).

    Parameters
    ----------
    splits : dict[str, Any]
        Split definition dict.
    path : Path
        Destination file path (parent dirs created automatically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path) -> dict[str, Any]:
    """Load splits from JSON file.

    Parameters
    ----------
    path : Path
        Path to the splits JSON.

    Returns
    -------
    dict[str, Any]
        Loaded split definition.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Generate all splits
# ---------------------------------------------------------------------------


def generate_all_splits(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    min_loato_samples: int = 200,
) -> dict[str, Path]:
    """Generate all 4 split types and save to *output_dir*.

    Loads experiment configs from ``configs/experiments/*.yaml``.

    Parameters
    ----------
    df : pd.DataFrame
        Finalized unified dataset (filtered, taxonomy-mapped, merged).
    output_dir : Path | None
        Directory for JSON split files.  Defaults to ``data/splits/``.

    Returns
    -------
    dict[str, Path]
        Mapping of experiment name → saved file path.
    """
    from loato_bench.utils.config import load_experiment_config

    if output_dir is None:
        output_dir = DATA_DIR / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # 1. Standard CV
    cv_cfg = load_experiment_config("standard_cv")
    cv_splits = generate_standard_cv_splits(
        df,
        n_folds=cv_cfg.n_folds or 5,
        stratify_by=cv_cfg.stratify_by or ["label"],
        seed=cv_cfg.seed,
    )
    cv_path = output_dir / "standard_cv_folds.json"
    save_splits(cv_splits, cv_path)
    saved["standard_cv"] = cv_path
    logger.info("Saved standard CV splits to %s", cv_path)

    # 2. LOATO
    loato_cfg = load_experiment_config("loato")
    try:
        loato_splits = generate_loato_splits(
            df,
            benign_test_fraction=loato_cfg.benign_test_fraction or 0.2,
            min_samples=min_loato_samples,
            seed=loato_cfg.seed,
        )
        loato_path = output_dir / "loato_splits.json"
        save_splits(loato_splits, loato_path)
        saved["loato"] = loato_path
        logger.info("Saved LOATO splits to %s", loato_path)
    except ValueError as e:
        logger.warning("Skipping LOATO splits: %s", e)

    # 3. Direct → Indirect
    di_cfg = load_experiment_config("direct_indirect")
    try:
        di_splits = generate_direct_indirect_split(
            df,
            benign_test_fraction=di_cfg.benign_test_fraction or 0.2,
            seed=di_cfg.seed,
        )
        di_path = output_dir / "direct_indirect_split.json"
        save_splits(di_splits, di_path)
        saved["direct_indirect"] = di_path
        logger.info("Saved direct/indirect split to %s", di_path)
    except ValueError as e:
        logger.warning("Skipping direct/indirect split: %s", e)

    # 4. Cross-lingual
    cl_cfg = load_experiment_config("crosslingual")
    try:
        cl_splits = generate_crosslingual_split(
            df,
            benign_test_fraction=cl_cfg.benign_test_fraction or 0.2,
            seed=cl_cfg.seed,
        )
        cl_path = output_dir / "crosslingual_split.json"
        save_splits(cl_splits, cl_path)
        saved["crosslingual"] = cl_path
        logger.info("Saved crosslingual split to %s", cl_path)
    except ValueError as e:
        logger.warning("Skipping crosslingual split: %s", e)

    return saved
