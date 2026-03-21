"""Stanford Alpaca (cleaned) dataset loader.

Loads the ``yahma/alpaca-cleaned`` dataset from HuggingFace. Uses the
``instruction`` field as prompt text, appending ``input`` when non-empty.
All samples are benign (label=0).

Reference: https://huggingface.co/datasets/yahma/alpaca-cleaned
Schema: instruction (str), input (str), output (str)
Splits: train (~51,760 rows)
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "yahma/alpaca-cleaned"
DEFAULT_MAX_SAMPLES = 8_000


class AlpacaLoader(DatasetLoader):
    """Load the Stanford Alpaca (cleaned) dataset.

    Composes prompt text from ``instruction`` + ``input`` (when non-empty),
    deduplicates, and caps at ``max_samples``. All samples are benign (label=0).

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to return after dedup. Defaults to 3,000.
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download, deduplicate, and transform the Alpaca dataset."""
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

        # Compose text, filter empty, deduplicate
        seen: set[str] = set()
        deduped: list[tuple[dict, str]] = []
        for row in all_rows:
            instruction = row["instruction"].strip()
            if not instruction:
                continue
            input_text = row.get("input", "").strip()
            text = f"{instruction}\n{input_text}" if input_text else instruction
            if text not in seen:
                seen.add(text)
                deduped.append((row, text))
        logger.info("After empty-filter + dedup: %d rows", len(deduped))

        # Cap at max_samples
        if len(deduped) > self.max_samples:
            deduped = deduped[: self.max_samples]
            logger.info("Capped to %d samples", self.max_samples)

        # Transform to UnifiedSample
        samples: list[UnifiedSample] = []
        for row, text in deduped:
            samples.append(
                UnifiedSample(
                    text=text,
                    label=0,
                    source="alpaca",
                    attack_category=None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "has_input": bool(row.get("input", "").strip()),
                    },
                )
            )

        logger.info("Loaded %d benign samples from Alpaca", len(samples))
        return samples
