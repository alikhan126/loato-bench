"""OpenAssistant OASST1 dataset loader.

Loads the ``OpenAssistant/oasst1`` dataset from HuggingFace, filters for
English prompter (user) messages, deduplicates, and caps at ``max_samples``.
All samples are benign (label=0).

Reference: https://huggingface.co/datasets/OpenAssistant/oasst1
Schema: message_id, parent_id, text, role (prompter|assistant), lang, ...
Splits: train (~84K messages), validation (~4.4K messages)
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "OpenAssistant/oasst1"
DEFAULT_MAX_SAMPLES = 8_000


class OASSTLoader(DatasetLoader):
    """Load English prompter messages from OASST1.

    Filters for ``role == "prompter"`` and ``lang == "en"``, deduplicates on
    text, and caps at ``max_samples``. All samples are benign (label=0).

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to return after filtering and dedup.
        Defaults to 8,000.
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download, filter, deduplicate, and transform OASST1 prompter messages."""
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

        # Filter: prompter role only
        prompter_rows = [r for r in all_rows if r.get("role") == "prompter"]
        logger.info("After role=prompter filter: %d rows", len(prompter_rows))

        # Filter: English only
        english_rows = [r for r in prompter_rows if r.get("lang") == "en"]
        logger.info("After lang=en filter: %d rows", len(english_rows))

        # Filter empty text and deduplicate
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in english_rows:
            text = row["text"].strip()
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
                    text=row["text"].strip(),
                    label=0,
                    source="oasst",
                    attack_category=None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "message_id": row.get("message_id"),
                        "parent_id": row.get("parent_id"),
                    },
                )
            )

        logger.info("Loaded %d benign samples from OASST1", len(samples))
        return samples
