"""Re-label the 1,065 uncertain samples using GPT-4o-mini.

Reuses the existing labeling pipeline (same system prompt, same taxonomy,
same confidence threshold of 0.6).

- Confident (≥0.6) → apply that label
- Still uncertain (<0.6) → assign C7 (other)

Output: data/review/manual_overrides.csv
"""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path
import sys

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loato_bench.data.llm_labeler import (  # noqa: E402
    _call_openai_async,
    build_labeling_system_prompt,
    parse_llm_response,
)
from loato_bench.data.taxonomy import _text_hash  # noqa: E402
from loato_bench.utils.config import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
CONCURRENCY = 10


async def main() -> None:
    # Load labeled dataset to get full text
    parquet_path = DATA_DIR / "processed" / "labeled_v1.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d samples from %s", len(df), parquet_path)

    # Find uncertain samples
    uncertain_mask = df["label_source"] == "uncertain"
    uncertain = df[uncertain_mask].copy()
    logger.info("Found %d uncertain samples", len(uncertain))

    if uncertain.empty:
        logger.info("Nothing to do.")
        return

    # Compute hashes for matching
    uncertain["_hash"] = uncertain["text"].apply(_text_hash)

    # Build work items
    system_prompt = build_labeling_system_prompt()
    import openai

    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    results: list[dict[str, str]] = []
    total = len(uncertain)
    completed = 0

    async def process_one(row: pd.Series) -> dict[str, str]:
        nonlocal completed
        text = str(row["text"])
        sample_hash = str(row["_hash"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        async with sem:
            try:
                raw = await _call_openai_async(client, messages, MODEL, TEMPERATURE)
            except Exception:
                logger.exception("API call failed for %s", sample_hash)
                # Fallback to C7
                completed += 1
                return {"sample_hash": sample_hash, "correct_category": "other"}

        slug, confidence = parse_llm_response(raw)

        if slug and confidence is not None and confidence >= CONFIDENCE_THRESHOLD:
            category = slug
        else:
            category = "other"

        completed += 1
        if completed % 100 == 0:
            logger.info("Progress: %d / %d", completed, total)

        return {"sample_hash": sample_hash, "correct_category": category}

    # Run all concurrently
    tasks = [process_one(row) for _, row in uncertain.iterrows()]
    results = await asyncio.gather(*tasks)

    await client.close()

    # Write manual_overrides.csv
    out_dir = DATA_DIR / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manual_overrides.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_hash", "correct_category"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    categories = pd.Series([r["correct_category"] for r in results])
    logger.info("Wrote %d overrides to %s", len(results), out_path)
    logger.info("Distribution:\n%s", categories.value_counts().to_string())

    confident_count = sum(1 for r in results if r["correct_category"] != "other")
    other_count = sum(1 for r in results if r["correct_category"] == "other")
    logger.info("Confident: %d, Fallback to C7 (other): %d", confident_count, other_count)


if __name__ == "__main__":
    asyncio.run(main())
