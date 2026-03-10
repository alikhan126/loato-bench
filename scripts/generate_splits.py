"""LOATO-2A-06: Generate & lock all evaluation splits.

Loads labeled_v1.parquet, filters per final_categories.json, and generates:

1. Standard CV — 5-fold stratified on (label, attack_category), C7 excluded
2. LOATO — 5 folds (C1–C5 held out), C6 train-only, C7 excluded

Outputs parquet train/test pairs and a SHA-256 manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loato_bench.data.splits import (  # noqa: E402
    generate_loato_splits,
    generate_standard_cv_splits,
    save_split_parquets,
    save_splits,
    write_split_manifest,
)
from loato_bench.data.taxonomy_spec import SLUG_TO_CATEGORY_ID  # noqa: E402
from loato_bench.utils.config import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def load_final_categories() -> dict[str, object]:
    """Load final_categories.json from configs/."""
    path = PROJECT_ROOT / "configs" / "final_categories.json"
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def main() -> None:
    # 1. Load source data
    parquet_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d samples from %s", len(df), parquet_path.name)

    # 2. Load feasibility config
    config = load_final_categories()
    loato_categories: list[str] = config["loato_categories"]  # type: ignore[assignment]
    logger.info("LOATO-eligible categories: %s", loato_categories)

    # Determine excluded and train-only categories
    all_slugs = {c["slug"] for c in config["categories"]}  # type: ignore[union-attr]
    excluded = {"other"}  # C7 — excluded from all splits
    train_only = all_slugs - set(loato_categories) - excluded  # C6
    logger.info("Excluded (C7): %s", excluded)
    logger.info("Train-only: %s", train_only)

    # 3. Filter C7 from DataFrame
    c7_mask = df["attack_category"] == "other"
    n_c7 = c7_mask.sum()
    df_filtered = df[~c7_mask].reset_index(drop=True)
    logger.info("Filtered %d C7 samples → %d remaining", n_c7, len(df_filtered))

    output_dir = DATA_DIR / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_parquet_files: list[Path] = []
    splits_meta: list[dict[str, object]] = []

    # -----------------------------------------------------------------------
    # 4. Standard CV — 5-fold stratified
    # -----------------------------------------------------------------------
    logger.info("\n--- Standard CV (5-fold) ---")
    cv_splits = generate_standard_cv_splits(
        df_filtered,
        n_folds=5,
        stratify_by=["label"],
        seed=SEED,
    )

    # Save JSON (backward compat)
    save_splits(cv_splits, output_dir / "standard_cv_folds.json")

    # Save parquets
    cv_dir = output_dir / "standard_cv"
    cv_parquets = save_split_parquets(
        df_filtered,
        cv_splits,
        cv_dir,
        fold_name_fn=lambda f: f"fold_{f['fold']}",
    )
    all_parquet_files.extend(cv_parquets)

    for fold in cv_splits["folds"]:
        n_train = len(fold["train_indices"])
        n_test = len(fold["test_indices"])
        train_df = df_filtered.iloc[fold["train_indices"]]
        test_df = df_filtered.iloc[fold["test_indices"]]
        logger.info(
            "  fold_%d: train=%d (inj=%d), test=%d (inj=%d)",
            fold["fold"],
            n_train,
            (train_df["label"] == 1).sum(),
            n_test,
            (test_df["label"] == 1).sum(),
        )
        splits_meta.append(
            {
                "split_type": "standard_cv",
                "fold_id": fold["fold"],
                "n_train": n_train,
                "n_test": n_test,
                "train_injection": int((train_df["label"] == 1).sum()),
                "train_benign": int((train_df["label"] == 0).sum()),
                "test_injection": int((test_df["label"] == 1).sum()),
                "test_benign": int((test_df["label"] == 0).sum()),
            }
        )

    # -----------------------------------------------------------------------
    # 5. LOATO — 5 folds (C1–C5)
    # -----------------------------------------------------------------------
    logger.info("\n--- LOATO (5 folds) ---")
    loato_splits = generate_loato_splits(
        df_filtered,
        benign_test_fraction=0.2,
        min_samples=200,
        seed=SEED,
        train_only_categories=train_only,
    )

    # Save JSON (backward compat)
    save_splits(loato_splits, output_dir / "loato_splits.json")

    # Save parquets with C-ID naming
    def loato_fold_name(fold: dict[str, object]) -> str:
        slug = str(fold["held_out_category"])
        cid = SLUG_TO_CATEGORY_ID.get(slug, slug)
        return f"fold_{cid}"

    loato_dir = output_dir / "loato"
    loato_parquets = save_split_parquets(
        df_filtered,
        loato_splits,
        loato_dir,
        fold_name_fn=loato_fold_name,
    )
    all_parquet_files.extend(loato_parquets)

    for fold in loato_splits["folds"]:
        slug = fold["held_out_category"]
        cid = SLUG_TO_CATEGORY_ID.get(str(slug), str(slug))
        train_df = df_filtered.iloc[fold["train_indices"]]
        test_df = df_filtered.iloc[fold["test_indices"]]

        # Verify C6 not in test
        test_cats = test_df[test_df["label"] == 1]["attack_category"].unique()
        assert "context_manipulation" not in test_cats, f"C6 in test for fold {cid}!"

        # Verify no C7
        all_cats = set(train_df["attack_category"].dropna()) | set(
            test_df["attack_category"].dropna()
        )
        assert "other" not in all_cats, f"C7 found in fold {cid}!"

        # Verify no overlap
        train_set = set(fold["train_indices"])
        test_set = set(fold["test_indices"])
        assert train_set.isdisjoint(test_set), f"Overlap in fold {cid}!"

        logger.info(
            "  %s (%s): train=%d, test=%d, held_out_inj=%d",
            cid,
            slug,
            fold["n_train"],
            fold["n_test"],
            (test_df["label"] == 1).sum(),
        )

        splits_meta.append(
            {
                "split_type": "loato",
                "fold_id": cid,
                "held_out_category": slug,
                "n_train": fold["n_train"],
                "n_test": fold["n_test"],
                "train_injection": int((train_df["label"] == 1).sum()),
                "train_benign": int((train_df["label"] == 0).sum()),
                "test_injection": int((test_df["label"] == 1).sum()),
                "test_benign": int((test_df["label"] == 0).sum()),
                "train_categories": fold["train_categories"],
            }
        )

    # -----------------------------------------------------------------------
    # 6. Write manifest
    # -----------------------------------------------------------------------
    manifest_path = write_split_manifest(
        output_dir=output_dir,
        source_path=parquet_path,
        split_files=all_parquet_files,
        splits_meta=splits_meta,
        seed=SEED,
    )
    logger.info("\nWrote manifest: %s", manifest_path)

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Standard CV: 5 folds × train/test parquet")
    logger.info("LOATO: %d folds × train/test parquet", len(loato_splits["folds"]))
    logger.info("Total parquet files: %d", len(all_parquet_files))
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
