"""PINT / Gandalf dataset loader.

Uses the ``lakera/gandalf_ignore_instructions`` HuggingFace dataset as the
publicly available fallback for the PINT Benchmark.

Reference: https://huggingface.co/datasets/lakera/gandalf_ignore_instructions
Schema: text (str), similarity (float64)
Splits: train (777), validation (111), test (112) — 1000 total
Note: All samples are injection attempts (label=1). English only.
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from promptguard.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "lakera/gandalf_ignore_instructions"


class PINTLoader(DatasetLoader):
    """Load the PINT / Gandalf Ignore Instructions dataset.

    Merges all splits (train, validation, test) into a single list.
    All samples are injection attempts (label=1). The ``similarity`` score
    (cosine similarity to "Ignore all previous instructions") is stored
    in metadata.
    """

    def load(self) -> list[UnifiedSample]:
        """Download and transform the Gandalf dataset."""
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
                        label=1,  # All Gandalf samples are injections
                        source="pint",
                        attack_category=None,
                        language="en",
                        is_indirect=False,
                        metadata={
                            "similarity": float(row["similarity"]),
                            "split": split_name,
                        },
                    )
                )

        logger.info("Loaded %d samples from PINT/Gandalf", len(samples))
        return samples
