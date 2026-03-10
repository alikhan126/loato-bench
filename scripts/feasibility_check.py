"""LOATO-2A-04: Feasibility check — count viable LOATO categories.

Loads the final labeled dataset, applies manual overrides from 2A-03,
filters GenTel samples (injection_score < 0.4), runs within-category
near-dedup (Jaccard 0.90, word 5-grams), and counts samples per category.

Output: prints category counts and viable LOATO categories (≥200 samples).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loato_bench.analysis.quality import compute_injection_confidence_scores  # noqa: E402
from loato_bench.data.base import UnifiedSample  # noqa: E402
from loato_bench.data.harmonize import near_dedup  # noqa: E402
from loato_bench.data.review import apply_manual_overrides, load_manual_overrides  # noqa: E402
from loato_bench.data.taxonomy_spec import TAXONOMY_V1  # noqa: E402
from loato_bench.utils.config import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MIN_SAMPLES = 200


def df_rows_to_samples(df: pd.DataFrame) -> list[UnifiedSample]:
    """Convert DataFrame rows back to UnifiedSample for near_dedup."""
    samples = []
    for _, row in df.iterrows():
        samples.append(
            UnifiedSample(
                text=str(row["text"]),
                label=int(row["label"]),
                source=str(row["source"]),
                attack_category=row.get("attack_category"),
                original_category=row.get("original_category"),
                language=str(row.get("language", "en")),
                is_indirect=bool(row.get("is_indirect", False)),
            )
        )
    return samples


def main() -> None:
    # 1. Load labeled dataset
    parquet_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d samples from %s", len(df), parquet_path)

    # 2. Apply manual overrides from 2A-03
    overrides_path = DATA_DIR / "review" / "manual_overrides.csv"
    if overrides_path.exists():
        overrides = load_manual_overrides(overrides_path)
        df = apply_manual_overrides(df, overrides)
        logger.info("Applied %d manual overrides", len(overrides))
    else:
        logger.warning("No manual_overrides.csv found — skipping")

    # Show label_source breakdown after overrides
    logger.info("Label source breakdown:\n%s", df["label_source"].value_counts().to_string())

    # 3. Filter to injection samples only
    injection_df = df[df["label"] == 1].copy()
    logger.info("Injection samples: %d", len(injection_df))

    # 4. Apply GenTel filter: drop GenTel samples with injection_score < 0.4
    gentel_mask = injection_df["source"].str.lower().str.contains("gentel", na=False)
    gentel_count = gentel_mask.sum()
    logger.info("GenTel injection samples: %d", gentel_count)

    if gentel_count > 0:
        gentel_df = injection_df[gentel_mask]
        scores = compute_injection_confidence_scores(gentel_df)
        low_score_mask = scores < 0.4
        drop_count = low_score_mask.sum()
        # Drop low-confidence GenTel samples
        drop_indices = gentel_df[low_score_mask].index
        injection_df = injection_df.drop(drop_indices)
        logger.info(
            "GenTel filter: dropped %d / %d (score < 0.4), kept %d",
            drop_count,
            gentel_count,
            gentel_count - drop_count,
        )

        # Cap GenTel at 5K if needed
        remaining_gentel = injection_df[
            injection_df["source"].str.lower().str.contains("gentel", na=False)
        ]
        if len(remaining_gentel) > 5000:
            gentel_scores = compute_injection_confidence_scores(remaining_gentel)
            keep_idx = gentel_scores.nlargest(5000).index
            drop_idx = remaining_gentel.index.difference(keep_idx)
            injection_df = injection_df.drop(drop_idx)
            logger.info("GenTel cap: dropped %d to keep top 5K", len(drop_idx))

    logger.info("After GenTel filter: %d injection samples", len(injection_df))

    # 5. Within-category near-dedup (Jaccard 0.90, word 5-grams)
    category_counts_pre_dedup: dict[str, int] = {}
    category_counts_post_dedup: dict[str, int] = {}
    deduped_frames: list[pd.DataFrame] = []

    categories = [
        c for c in injection_df["attack_category"].unique() if isinstance(c, str) and c != "other"
    ]
    categories.sort()

    for cat in categories:
        cat_df = injection_df[injection_df["attack_category"] == cat]
        category_counts_pre_dedup[cat] = len(cat_df)

        # Convert to UnifiedSample for near_dedup
        samples = df_rows_to_samples(cat_df)
        deduped = near_dedup(samples, threshold=0.90)

        category_counts_post_dedup[cat] = len(deduped)

        # Keep the deduped indices (by reconstructing from text match)
        deduped_texts = {s.text for s in deduped}
        kept_mask = cat_df["text"].isin(deduped_texts)
        deduped_frames.append(cat_df[kept_mask])

    # Also handle "other" and benign (not for LOATO, but for reference)
    other_df = injection_df[injection_df["attack_category"] == "other"]
    category_counts_pre_dedup["other"] = len(other_df)
    category_counts_post_dedup["other"] = len(other_df)  # skip dedup for other

    # 6. Report
    logger.info("\n" + "=" * 70)
    logger.info("LOATO FEASIBILITY CHECK — CATEGORY COUNTS")
    logger.info("=" * 70)
    logger.info(
        "%-25s %10s %10s %10s",
        "Category",
        "Pre-dedup",
        "Post-dedup",
        "Viable?",
    )
    logger.info("-" * 70)

    viable_categories: list[str] = []
    for cat in sorted(category_counts_post_dedup.keys()):
        pre = category_counts_pre_dedup[cat]
        post = category_counts_post_dedup[cat]
        viable = post >= MIN_SAMPLES and cat != "other"
        marker = "YES" if viable else ("n/a" if cat == "other" else "NO")
        logger.info("%-25s %10d %10d %10s", cat, pre, post, marker)
        if viable:
            viable_categories.append(cat)

    logger.info("-" * 70)
    logger.info("Viable LOATO categories (≥%d): %d", MIN_SAMPLES, len(viable_categories))
    logger.info("Categories: %s", ", ".join(viable_categories))

    # Determine outcome
    n_viable = len(viable_categories)
    if n_viable >= 5:
        outcome = "A"
        framing = "Strong benchmark — full plan stands"
    elif n_viable >= 3:
        outcome = "B"
        framing = "Solid benchmark — note coverage will grow as new attack types emerge"
    else:
        outcome = "C"
        framing = "Reframe thesis to emphasize protocol contribution over benchmark breadth"

    logger.info("\nOutcome: %s — %s", outcome, framing)

    # 7. Build updated final_categories.json
    updated_categories = []
    for cat_spec in TAXONOMY_V1.values():
        entry = {
            "id": cat_spec.id,
            "name": cat_spec.name,
            "slug": cat_spec.slug,
            "mechanism": cat_spec.mechanism,
            "loato_eligible": cat_spec.slug in viable_categories,
        }
        if cat_spec.slug in category_counts_post_dedup:
            entry["sample_count"] = category_counts_post_dedup[cat_spec.slug]
        updated_categories.append(entry)

    output = {
        "taxonomy_version": "1.0",
        "n_categories": 7,
        "loato_eligible_count": n_viable,
        "loato_categories": viable_categories,
        "feasibility_outcome": outcome,
        "feasibility_note": framing,
        "min_samples_threshold": MIN_SAMPLES,
        "filters_applied": [
            "manual_overrides (2A-03)",
            "gentel_injection_score >= 0.4",
            "gentel_cap_5000",
            "within_category_near_dedup (jaccard=0.90, word_5grams)",
        ],
        "categories": updated_categories,
    }

    out_path = PROJECT_ROOT / "configs" / "final_categories.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    logger.info("\nWrote updated %s", out_path)


if __name__ == "__main__":
    main()
