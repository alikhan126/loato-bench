"""Deepset prompt-injections dataset loader.

Loads the ``deepset/prompt-injections`` dataset from HuggingFace, merges
the train and test splits, and returns a list of :class:`UnifiedSample`.

Reference: https://huggingface.co/datasets/deepset/prompt-injections
Schema: text (str), label (int: 0=benign, 1=injection)
Splits: train (546 rows), test (116 rows) — 662 total
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "deepset/prompt-injections"


class DeepsetLoader(DatasetLoader):
    """Load the Deepset prompt-injections dataset.

    Merges train + test splits into a single list of :class:`UnifiedSample`.
    All samples are marked as direct (``is_indirect=False``) and English.
    Attack categories are left as ``None`` — taxonomy mapping is a later step.
    """

    def load(self) -> list[UnifiedSample]:
        """Download and transform the Deepset dataset into unified format."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH)

        samples: list[UnifiedSample] = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info("Processing split '%s' (%d rows)", split_name, len(split_data))
            for row in split_data:
                samples.append(
                    UnifiedSample(
                        text=row["text"],
                        label=int(row["label"]),
                        source="deepset",
                        attack_category=None,
                        language="en",
                        is_indirect=False,
                        metadata={"split": split_name},
                    )
                )

        logger.info("Loaded %d samples from Deepset", len(samples))
        return samples
