"""WildChat nontoxic dataset loader.

Loads the ``allenai/WildChat-nontoxic`` dataset from HuggingFace, extracts the
first user turn from each conversation, filters for English, deduplicates, and
caps at ``max_samples``. All samples are benign (label=0).

Reference: https://huggingface.co/datasets/allenai/WildChat-nontoxic
Schema: conversation (list[dict]), language (str), model (str), ...
Note: ``language`` uses full names (e.g., "English") not ISO codes.
"""

from __future__ import annotations

import logging

from datasets import load_dataset
from huggingface_hub import get_token

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "allenai/WildChat-nontoxic"
DEFAULT_MAX_SAMPLES = 8_000


class WildChatLoader(DatasetLoader):
    """Load first user turns from WildChat nontoxic conversations.

    Extracts the first ``role == "user"`` message from each conversation,
    filters for English, deduplicates, and caps at ``max_samples``.
    All samples are benign (label=0).

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to return after filtering and dedup.
        Defaults to 8,000.
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download, extract first user turns, filter, deduplicate, and transform."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH, token=get_token())

        # Collect all rows across splits
        all_rows: list[dict] = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info("Processing split '%s' (%d rows)", split_name, len(split_data))
            for row in split_data:
                all_rows.append(row)
        logger.info("Total raw rows: %d", len(all_rows))

        # Filter: English only
        english_rows = [r for r in all_rows if r.get("language") == "English"]
        logger.info("After language=English filter: %d rows", len(english_rows))

        # Extract first user turn, filter empty, deduplicate
        seen: set[str] = set()
        deduped: list[tuple[dict, str]] = []
        skipped_no_user = 0
        for row in english_rows:
            conversation = row.get("conversation", [])
            first_user_text = ""
            for turn in conversation:
                if turn.get("role") == "user":
                    first_user_text = turn.get("content", "").strip()
                    break

            if not first_user_text:
                skipped_no_user += 1
                continue
            if first_user_text not in seen:
                seen.add(first_user_text)
                deduped.append((row, first_user_text))

        logger.info(
            "After first-user-turn extraction + dedup: %d rows (skipped %d with no user turn)",
            len(deduped),
            skipped_no_user,
        )

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
                    source="wildchat",
                    attack_category=None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "model": row.get("model"),
                    },
                )
            )

        logger.info("Loaded %d benign samples from WildChat", len(samples))
        return samples
