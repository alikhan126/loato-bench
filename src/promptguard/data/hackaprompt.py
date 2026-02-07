"""HackAPrompt dataset loader.

Loads the ``hackaprompt/hackaprompt-dataset`` from HuggingFace, filters for
successful injection attempts (``correct=True``, ``error=False``),
deduplicates on ``user_input``, and caps at ``max_samples``.

Reference: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
Schema: user_input, prompt, completion, model, correct, level, error, ...
Note: Injection-only dataset — all samples are label=1.
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from promptguard.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "hackaprompt/hackaprompt-dataset"
DEFAULT_MAX_SAMPLES = 5000


class HackaPromptLoader(DatasetLoader):
    """Load the HackAPrompt dataset.

    Filters for successful (``correct=True``) and non-error rows,
    deduplicates on ``user_input``, and caps at ``max_samples``.
    All samples are injection (label=1) — this dataset has no benign samples.

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to return after filtering and dedup.
        Defaults to 5000.
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download, filter, dedup, and transform the HackAPrompt dataset."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH)

        # Collect all rows across splits
        all_rows: list[dict] = []
        for split_name in dataset.keys():
            for row in dataset[split_name]:
                all_rows.append(row)
        logger.info("Total raw rows: %d", len(all_rows))

        # Filter: keep only correct, non-error rows
        filtered = [r for r in all_rows if r.get("correct") and not r.get("error")]
        logger.info("After correct=True, error=False filter: %d rows", len(filtered))

        # Deduplicate on user_input (keep first occurrence)
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in filtered:
            text = row["user_input"]
            if text not in seen:
                seen.add(text)
                deduped.append(row)
        logger.info("After dedup on user_input: %d rows", len(deduped))

        # Cap at max_samples
        if len(deduped) > self.max_samples:
            deduped = deduped[: self.max_samples]
            logger.info("Capped to %d samples", self.max_samples)

        # Transform to UnifiedSample
        samples: list[UnifiedSample] = []
        for row in deduped:
            samples.append(
                UnifiedSample(
                    text=row["user_input"],
                    label=1,  # All HackAPrompt samples are injections
                    source="hackaprompt",
                    attack_category=None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "level": row.get("level"),
                        "model": row.get("model"),
                        "score": row.get("score"),
                        "dataset_subset": row.get("dataset"),
                    },
                )
            )

        logger.info("Loaded %d samples from HackAPrompt", len(samples))
        return samples
