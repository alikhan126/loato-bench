"""Open-Prompt-Injection dataset loader.

Loads the ``guychuk/open-prompt-injection`` dataset from HuggingFace. Each row
contains paired ``normal_input`` (benign) and ``attack_input`` (injection) text.

Reference: https://huggingface.co/datasets/guychuk/open-prompt-injection
Paper: "Formalizing and Benchmarking Prompt Injection Attacks and Defenses" (USENIX Security 2024)

Schema: instruction, normal_input, attack_input, task_type, attack_type,
        injected_task, sample_id
Total: ~33,600 rows, single train split.

Each row produces:
  - 1 injection sample from ``attack_input`` (label=1, is_indirect=True)
  - 1 benign sample from ``normal_input`` (label=0, deduplicated)
"""

from __future__ import annotations

import logging

from datasets import load_dataset

from loato_bench.data.base import DatasetLoader, UnifiedSample

logger = logging.getLogger(__name__)

HF_PATH = "guychuk/open-prompt-injection"


class OpenPromptLoader(DatasetLoader):
    """Load the Open-Prompt-Injection dataset.

    Produces both benign and injection samples. Benign samples are
    deduplicated on ``normal_input`` since the same clean text appears
    across multiple attack types. Injection samples are marked as
    indirect (``is_indirect=True``) because the attack is embedded in
    the data field, not the instruction.

    Parameters
    ----------
    max_samples : int | None
        Maximum number of injection samples to include. Benign samples
        are capped proportionally. ``None`` means no cap (default).
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self.max_samples = max_samples

    def load(self) -> list[UnifiedSample]:
        """Download and transform Open-Prompt-Injection into unified format."""
        logger.info("Loading dataset from %s", HF_PATH)
        dataset = load_dataset(HF_PATH)

        # Collect all rows across splits
        all_rows: list[dict] = []
        for split_name in dataset.keys():
            for row in dataset[split_name]:
                all_rows.append(row)
        logger.info("Total raw rows: %d", len(all_rows))

        # Cap rows before processing if max_samples is set
        if self.max_samples is not None and len(all_rows) > self.max_samples:
            all_rows = all_rows[: self.max_samples]
            logger.info("Capped to %d rows", self.max_samples)

        # Build injection samples (one per row)
        injection_samples: list[UnifiedSample] = []
        for row in all_rows:
            injection_samples.append(
                UnifiedSample(
                    text=row["attack_input"],
                    label=1,
                    source="open_prompt_injection",
                    attack_category=None,
                    original_category=row.get("attack_type"),
                    language="en",
                    is_indirect=True,
                    metadata={
                        "task_type": row.get("task_type"),
                        "injected_task": row.get("injected_task"),
                        "instruction": row.get("instruction"),
                        "sample_id": row.get("sample_id"),
                    },
                )
            )

        # Build benign samples (deduplicated on normal_input)
        seen_benign: set[str] = set()
        benign_samples: list[UnifiedSample] = []
        for row in all_rows:
            text = row["normal_input"]
            if text not in seen_benign:
                seen_benign.add(text)
                benign_samples.append(
                    UnifiedSample(
                        text=text,
                        label=0,
                        source="open_prompt_injection",
                        attack_category=None,
                        original_category=None,
                        language="en",
                        is_indirect=False,
                        metadata={
                            "task_type": row.get("task_type"),
                        },
                    )
                )

        samples = injection_samples + benign_samples
        logger.info(
            "Loaded %d samples (%d injection, %d benign) from Open-Prompt-Injection",
            len(samples),
            len(injection_samples),
            len(benign_samples),
        )
        return samples
