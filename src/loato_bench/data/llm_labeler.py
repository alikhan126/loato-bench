"""GPT-4o-mini batch categorization of unlabeled injection samples.

Uses the OpenAI structured JSON output to classify injection samples into
taxonomy v1.0 categories (C1--C7).  Results are logged to JSONL for
checkpoint/resume and auditing.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from loato_bench.data.taxonomy import _text_hash
from loato_bench.data.taxonomy_spec import (
    CATEGORY_ID_TO_SLUG,
    TAXONOMY_V1,
)
from loato_bench.utils.config import load_llm_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_labeling_system_prompt() -> str:
    """Build the system prompt from ``TAXONOMY_V1`` category specs.

    Dynamically generates category descriptions, boundary rules, signal
    phrases, and examples.  Never hardcoded.

    Returns
    -------
    str
        System prompt for the GPT-4o-mini labeling task.
    """
    lines: list[str] = [
        "You are a prompt-injection attack classifier.",
        "Classify the user's prompt into exactly ONE of the following categories.",
        "",
        "## Categories",
        "",
    ]

    for cid, spec in TAXONOMY_V1.items():
        lines.append(f"### {cid}: {spec.name} (`{spec.slug}`)")
        lines.append(f"Mechanism: {spec.mechanism}")
        if spec.signal_phrases:
            phrases = ", ".join(f'"{p}"' for p in spec.signal_phrases)
            lines.append(f"Signal phrases: {phrases}")
        if spec.exclusions:
            for exc in spec.exclusions:
                lines.append(f"- Exclusion: {exc}")
        if spec.examples_positive:
            lines.append("Positive examples:")
            for ex in spec.examples_positive:
                lines.append(f'  - "{ex}"')
        if spec.examples_negative:
            lines.append("Negative examples:")
            for ex in spec.examples_negative:
                lines.append(f'  - "{ex}"')
        lines.append("")

    lines.extend(
        [
            "## Boundary Rules",
            "- If one strategy clearly dominates, assign to that category.",
            "- If the prompt adopts a persona/roleplay AND overrides instructions, "
            "the primary mechanism decides: persona = C2, override = C1.",
            "- Encoding/obfuscation tricks are C3 even if wrapped in a persona.",
            "- Use C7 only if no single category fits or multiple strategies "
            "are equally prominent.",
            "",
            "## Response Format",
            "Respond with ONLY a JSON object (no markdown, no explanation):",
            '{"category": "<C-ID>", "confidence": <float 0.0-1.0>}',
            "",
            'Example: {"category": "C1", "confidence": 0.92}',
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_llm_response(raw: str) -> tuple[str | None, float | None]:
    """Parse the LLM JSON response into ``(slug, confidence)``.

    Parameters
    ----------
    raw : str
        Raw response string from the LLM.

    Returns
    -------
    tuple[str | None, float | None]
        ``(slug, confidence)`` on success, ``(None, None)`` on failure.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON: %s", str(raw)[:200])
        return None, None

    category_id = data.get("category")
    confidence = data.get("confidence")

    if category_id not in CATEGORY_ID_TO_SLUG:
        logger.warning("Unknown category ID: %s", category_id)
        return None, None

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        logger.warning("Invalid confidence value: %s", confidence)
        return None, None

    if not 0.0 <= confidence <= 1.0:
        logger.warning("Confidence out of range: %s", confidence)
        return None, None

    return CATEGORY_ID_TO_SLUG[category_id], confidence


# ---------------------------------------------------------------------------
# Checkpoint / logging
# ---------------------------------------------------------------------------


def load_checkpoint(log_path: Path) -> set[str]:
    """Load already-processed sample hashes from a JSONL log file.

    Parameters
    ----------
    log_path : Path
        Path to the JSONL log file.

    Returns
    -------
    set[str]
        Set of sample hashes that have already been processed.
    """
    if not log_path.exists():
        return set()

    done: set[str] = set()
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sample_hash = record.get("sample_hash")
                if sample_hash:
                    done.add(sample_hash)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt log line: %s", line[:100])
    return done


def append_log(log_path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to the JSONL log file.

    Creates parent directories and the file if they don't exist.

    Parameters
    ----------
    log_path : Path
        Path to the JSONL log file.
    record : dict
        Record to append.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# OpenAI API call (tenacity-wrapped)
# ---------------------------------------------------------------------------


@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def _call_openai(
    client: openai.OpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Call the OpenAI Chat Completions API with structured JSON output.

    Parameters
    ----------
    client : openai.OpenAI
        OpenAI client instance.
    messages : list[dict[str, str]]
        Chat messages.
    model : str
        Model name (e.g. ``"gpt-4o-mini"``).
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        Raw response content.
    """
    response = client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=64,
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------


def label_samples(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.6,
    max_calls: int | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Label unlabeled injection samples using GPT-4o-mini.

    Only processes rows where ``label == 1`` and ``attack_category`` is null.
    Writes every API response to a JSONL log before applying labels.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``text``, ``label``, ``attack_category`` columns.
    confidence_threshold : float
        Minimum confidence to apply a label.  Below this, ``label_source``
        is set to ``"uncertain"``.
    max_calls : int or None
        Maximum number of API calls.  ``None`` means process all.
    output_dir : Path or None
        Directory for labeling artifacts.  Defaults to ``data/labeling/``.
    dry_run : bool
        If True, skip API calls and return the DataFrame unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``category_id``, ``category_slug``, ``confidence``,
        and ``label_source`` columns added/updated.
    """
    from loato_bench.utils.config import DATA_DIR

    if output_dir is None:
        output_dir = DATA_DIR / "labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "llm_labels_raw.jsonl"

    df = df.copy()

    # Ensure columns exist
    if "label_source" not in df.columns:
        df["label_source"] = pd.Series(dtype="object")
    if "confidence" not in df.columns:
        df["confidence"] = pd.Series(dtype="float64")

    # Mark existing Tier 1/2 labels
    has_category = df["attack_category"].notna() & (df["label"] == 1)
    df.loc[has_category & df["label_source"].isna(), "label_source"] = "tier1_2"

    # Find unlabeled injection samples
    unlabeled_mask = (df["label"] == 1) & df["attack_category"].isna()
    unlabeled_indices = df.index[unlabeled_mask].tolist()

    if not unlabeled_indices:
        logger.info("No unlabeled injection samples to process.")
        return df

    logger.info("Found %d unlabeled injection samples.", len(unlabeled_indices))

    if dry_run:
        logger.info("Dry run — skipping API calls.")
        return df

    # Load checkpoint
    done_hashes = load_checkpoint(log_path)
    logger.info("Checkpoint: %d samples already processed.", len(done_hashes))

    # Build prompt
    system_prompt = build_labeling_system_prompt()

    # Load config
    try:
        llm_config = load_llm_config()
        model = llm_config.model
        temperature = llm_config.temperature
    except Exception:
        model = "gpt-4o-mini"
        temperature = 0.0

    client = openai.OpenAI()

    calls_made = 0
    for idx in unlabeled_indices:
        text = df.at[idx, "text"]
        sample_hash = _text_hash(str(text))

        # Skip if already in checkpoint
        if sample_hash in done_hashes:
            continue

        if max_calls is not None and calls_made >= max_calls:
            logger.info("Reached max_calls limit (%d).", max_calls)
            break

        # Call API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(text)},
        ]
        try:
            raw_response = _call_openai(client, messages, model, temperature)
        except Exception:
            logger.exception("API call failed for sample %s", sample_hash)
            continue

        calls_made += 1

        slug, confidence = parse_llm_response(raw_response)

        # Log BEFORE applying label
        record = {
            "sample_hash": sample_hash,
            "text_preview": str(text)[:100],
            "raw_response": raw_response,
            "category_id": next((cid for cid, s in CATEGORY_ID_TO_SLUG.items() if s == slug), None),
            "category_slug": slug,
            "confidence": confidence,
            "model": model,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }
        append_log(log_path, record)
        done_hashes.add(sample_hash)

        # Apply label
        if slug is not None and confidence is not None:
            if confidence >= confidence_threshold:
                df.at[idx, "attack_category"] = slug
                df.at[idx, "label_source"] = "gpt4o_mini"
                df.at[idx, "confidence"] = confidence
            else:
                df.at[idx, "label_source"] = "uncertain"
                df.at[idx, "confidence"] = confidence

    # Save processed labels
    labeled_mask = df["label_source"].isin({"gpt4o_mini", "uncertain"})
    if labeled_mask.any():
        cols = ["attack_category", "confidence", "label_source"]
        processed_df = df.loc[labeled_mask, cols].copy()
        processed_df["sample_hash"] = [_text_hash(str(t)) for t in df.loc[labeled_mask, "text"]]
        processed_df["applied"] = df.loc[labeled_mask, "label_source"] == "gpt4o_mini"
        processed_path = output_dir / "llm_labels_processed.parquet"
        processed_df.to_parquet(processed_path, index=False)
        logger.info("Saved processed labels to %s", processed_path)

    logger.info(
        "Labeling complete: %d API calls, %d labels applied.",
        calls_made,
        (df["label_source"] == "gpt4o_mini").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Distribution validation
# ---------------------------------------------------------------------------


def validate_distribution(results_df: pd.DataFrame) -> dict[str, Any]:
    """Sanity-check the distribution of LLM-assigned labels.

    Flags if any single category has > 60% of LLM labels.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with ``attack_category`` and ``label_source`` columns.

    Returns
    -------
    dict[str, Any]
        Report with category counts, percentages, and warnings.
    """
    llm_labeled = results_df[results_df["label_source"] == "gpt4o_mini"]
    total = len(llm_labeled)

    if total == 0:
        return {"total_llm_labeled": 0, "category_counts": {}, "warnings": []}

    counts = llm_labeled["attack_category"].value_counts().to_dict()
    percentages = {k: v / total for k, v in counts.items()}

    warnings: list[str] = []
    for cat, pct in percentages.items():
        if pct > 0.6:
            warnings.append(
                f"Category '{cat}' has {pct:.1%} of LLM labels ({counts[cat]}/{total}). "
                f"Expected < 60%."
            )

    return {
        "total_llm_labeled": total,
        "category_counts": counts,
        "category_percentages": {k: round(v, 4) for k, v in percentages.items()},
        "warnings": warnings,
    }
