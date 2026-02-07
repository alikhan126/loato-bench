"""GenTel-Bench dataset loader.

Loads the ``GenTelLab/gentelbench-v1`` dataset from HuggingFace.

Reference: https://huggingface.co/datasets/GenTelLab/gentelbench-v1
Schema: id (int), text (str), label (int: 0/1), domain (str), subdomain (str)
Note: Categories are *content harm* types, not injection techniques.
      A quality gate (EDA) is needed to identify genuine injection samples.
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from promptguard.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "GenTelLab/gentelbench-v1"


class GenTelLoader(DatasetLoader):
    """Load the GenTel-Bench dataset.

    Loads all rows and preserves domain/subdomain metadata for later
    quality-gate filtering during EDA. Attack categories are NOT mapped
    here — that happens in the taxonomy step.

    Parameters
    ----------
    max_samples : int or None
        Maximum number of samples to return. ``None`` means no cap.
        Useful because the full dataset is ~177K rows.
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download and transform the GenTel-Bench dataset."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH)

        all_rows: list[dict] = []
        for split_name in dataset.keys():
            for row in dataset[split_name]:
                all_rows.append(row)
        logger.info("Total raw rows: %d", len(all_rows))

        # Cap if requested
        if self.max_samples is not None and len(all_rows) > self.max_samples:
            all_rows = all_rows[: self.max_samples]
            logger.info("Capped to %d samples", self.max_samples)

        samples: list[UnifiedSample] = []
        for row in all_rows:
            label = int(row["label"])
            samples.append(
                UnifiedSample(
                    text=row["text"],
                    label=label,
                    source="gentelbench",
                    attack_category=None,
                    original_category=row.get("subdomain") if label == 1 else None,
                    language="en",
                    is_indirect=False,
                    metadata={
                        "id": row.get("id"),
                        "domain": row.get("domain"),
                        "subdomain": row.get("subdomain"),
                    },
                )
            )

        logger.info("Loaded %d samples from GenTel-Bench", len(samples))
        return samples
