"""Databricks Dolly 15K dataset loader.

Loads the ``databricks/databricks-dolly-15k`` dataset from HuggingFace.
All samples are benign (label=0) — human-written instruction-following prompts
by ~5,000 Databricks employees.

Reference: https://huggingface.co/datasets/databricks/databricks-dolly-15k
Schema: instruction (str), context (str), response (str), category (str)
Splits: train (15,011 rows)
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "databricks/databricks-dolly-15k"
DEFAULT_MAX_SAMPLES = 15_000


class DollyLoader(DatasetLoader):
    """Load the Databricks Dolly 15K dataset.

    Uses the ``instruction`` field as prompt text. Skips empty instructions,
    deduplicates, and caps at ``max_samples``. All samples are benign (label=0).

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to return after dedup. Defaults to 10,000.
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download, deduplicate, and transform the Dolly dataset."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH)

        # Collect all rows across splits
        all_rows: list[dict] = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info("Processing split '%s' (%d rows)", split_name, len(split_data))
            for row in split_data:
                all_rows.append(row)
        logger.info("Total raw rows: %d", len(all_rows))

        # Filter empty instructions and deduplicate
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in all_rows:
            text = row["instruction"].strip()
            if not text:
                continue
            if text not in seen:
                seen.add(text)
                deduped.append(row)
        logger.info("After empty-filter + dedup: %d rows", len(deduped))

        # Cap at max_samples
        if len(deduped) > self.max_samples:
            deduped = deduped[: self.max_samples]
            logger.info("Capped to %d samples", self.max_samples)

        # Transform to UnifiedSample
        samples: list[UnifiedSample] = []
        for row in deduped:
            samples.append(
                UnifiedSample(
                    text=row["instruction"].strip(),
                    label=0,
                    source="dolly",
                    attack_category=None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "category": row.get("category"),
                        "has_context": bool(row.get("context", "").strip()),
                    },
                )
            )

        logger.info("Loaded %d benign samples from Dolly", len(samples))
        return samples
