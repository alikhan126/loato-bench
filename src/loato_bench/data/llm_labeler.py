"""GPT-4o-mini / GPT-4o batch categorization of unlabeled injection samples.

Uses the OpenAI structured JSON output to classify injection samples into
taxonomy v1.0 categories (C1--C7).  Results are logged to JSONL for
checkpoint/resume and auditing.

Supports:
- Async concurrent requests for real-time labeling
- OpenAI Batch API for large runs (50% cheaper, no rate-limit issues)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from pathlib import Path
import time
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

DEFAULT_CONCURRENCY = 8


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
# OpenAI API calls (sync + async, tenacity-wrapped)
# ---------------------------------------------------------------------------


@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def _call_openai(  # pragma: no cover
    client: openai.OpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Call the OpenAI Chat Completions API with structured JSON output."""
    response = client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=64,
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or "").strip()


@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
)
async def _call_openai_async(
    client: openai.AsyncOpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Async version of _call_openai for concurrent batching."""
    response = await client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=64,
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Main labeling function (async core + sync wrapper)
# ---------------------------------------------------------------------------


async def _label_samples_async(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.6,
    max_calls: int | None = None,
    output_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    model_override: str | None = None,
) -> pd.DataFrame:
    """Async core: label unlabeled injection samples with concurrent API calls."""
    log_path = output_dir / "llm_labels_raw.jsonl"

    # Load checkpoint and apply existing results to df
    done_hashes = load_checkpoint(log_path)
    logger.info("Checkpoint: %d samples already processed.", len(done_hashes))

    if done_hashes and log_path.exists():
        checkpoint_data: dict[str, tuple[str | None, float | None]] = {}
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    checkpoint_data[rec["sample_hash"]] = (
                        rec.get("category_slug"),
                        rec.get("confidence"),
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

        # Apply checkpoint labels to unlabeled rows
        unlabeled_mask_ckpt = (df["label"] == 1) & df["attack_category"].isna()
        applied_from_ckpt = 0
        for idx in df.index[unlabeled_mask_ckpt]:
            text = str(df.at[idx, "text"])
            sh = _text_hash(text)
            if sh in checkpoint_data:
                slug, conf = checkpoint_data[sh]
                if slug and conf is not None and conf >= confidence_threshold:
                    df.at[idx, "attack_category"] = slug
                    df.at[idx, "label_source"] = "llm"
                    df.at[idx, "confidence"] = conf
                    applied_from_ckpt += 1
                elif conf is not None:
                    df.at[idx, "label_source"] = "uncertain"
                    df.at[idx, "confidence"] = conf
        logger.info("Applied %d labels from checkpoint.", applied_from_ckpt)

    # Build work items: (df_index, text, sample_hash)
    unlabeled_mask = (df["label"] == 1) & df["attack_category"].isna()
    work_items: list[tuple[int, str, str]] = []
    for idx in df.index[unlabeled_mask]:
        text = str(df.at[idx, "text"])
        sample_hash = _text_hash(text)
        if sample_hash not in done_hashes:
            work_items.append((idx, text, sample_hash))

    if max_calls is not None:
        work_items = work_items[:max_calls]

    if not work_items:
        logger.info("No new samples to process (all checkpointed).")
        return df

    logger.info(
        "Processing %d samples with concurrency=%d...",
        len(work_items),
        concurrency,
    )

    # Load config
    try:
        llm_config = load_llm_config()
        model = model_override or llm_config.model
        temperature = llm_config.temperature
    except Exception:
        model = model_override or "gpt-4o-mini"
        temperature = 0.0

    # Derive label_source tag from model name
    label_source_tag = model.replace("-", "_").replace(".", "_")

    system_prompt = build_labeling_system_prompt()
    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    calls_made = 0
    log_lock = asyncio.Lock()

    async def _process_one(
        idx: int, text: str, sample_hash: str
    ) -> tuple[int, str | None, float | None]:
        nonlocal calls_made
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        async with sem:
            try:
                raw_response = await _call_openai_async(client, messages, model, temperature)
            except Exception:
                logger.exception("API call failed for %s", sample_hash)
                return idx, None, None

        slug, confidence = parse_llm_response(raw_response)

        record = {
            "sample_hash": sample_hash,
            "text_preview": text[:100],
            "raw_response": raw_response,
            "category_id": next(
                (cid for cid, s in CATEGORY_ID_TO_SLUG.items() if s == slug),
                None,
            ),
            "category_slug": slug,
            "confidence": confidence,
            "model": model,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }
        async with log_lock:
            append_log(log_path, record)
            calls_made += 1
            if calls_made % 500 == 0:
                logger.info("Progress: %d / %d calls", calls_made, len(work_items))

        return idx, slug, confidence

    tasks = [_process_one(idx, text, sh) for idx, text, sh in work_items]
    results = await asyncio.gather(*tasks)

    # Apply labels
    for idx, slug, confidence in results:
        if slug is not None and confidence is not None:
            if confidence >= confidence_threshold:
                df.at[idx, "attack_category"] = slug
                df.at[idx, "label_source"] = label_source_tag
                df.at[idx, "confidence"] = confidence
            else:
                df.at[idx, "label_source"] = "uncertain"
                df.at[idx, "confidence"] = confidence

    await client.close()

    logger.info(
        "Labeling complete: %d API calls, %d labels applied.",
        calls_made,
        df["label_source"].isin({label_source_tag}).sum(),
    )
    return df


def label_samples(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.6,
    max_calls: int | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    model_override: str | None = None,
) -> pd.DataFrame:
    """Label unlabeled injection samples using GPT-4o-mini.

    Uses async concurrent requests (default 50 in-flight) for throughput.
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
    concurrency : int
        Max concurrent API requests (default 50).

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
    unlabeled_count = unlabeled_mask.sum()

    if unlabeled_count == 0:
        logger.info("No unlabeled injection samples to process.")
        return df

    logger.info("Found %d unlabeled injection samples.", unlabeled_count)

    if dry_run:
        logger.info("Dry run — skipping API calls.")
        return df

    df = asyncio.run(
        _label_samples_async(
            df,
            confidence_threshold=confidence_threshold,
            max_calls=max_calls,
            output_dir=output_dir,
            concurrency=concurrency,
            model_override=model_override,
        )
    )

    # Save processed labels
    non_llm = {"tier1_2", "uncertain"}
    llm_sources = {s for s in df["label_source"].dropna().unique() if s not in non_llm}
    labeled_mask = df["label_source"].isin(llm_sources | {"uncertain"})
    if labeled_mask.any():
        cols = ["attack_category", "confidence", "label_source"]
        processed_df = df.loc[labeled_mask, cols].copy()
        processed_df["sample_hash"] = [_text_hash(str(t)) for t in df.loc[labeled_mask, "text"]]
        processed_df["applied"] = df.loc[labeled_mask, "label_source"] != "uncertain"
        processed_path = output_dir / "llm_labels_processed.parquet"
        processed_df.to_parquet(processed_path, index=False)
        logger.info("Saved processed labels to %s", processed_path)

    return df


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

BATCH_MODEL = "gpt-4o"


def create_batch_request_file(  # pragma: no cover
    df: pd.DataFrame,
    output_dir: Path,
    *,
    model: str = BATCH_MODEL,
) -> tuple[Path, dict[str, int]]:
    """Create a JSONL request file for the OpenAI Batch API.

    Skips samples already in the checkpoint log.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with unlabeled injection samples.
    output_dir : Path
        Directory for batch artifacts.
    model : str
        Model to use (default ``gpt-4o``).

    Returns
    -------
    tuple[Path, dict[str, int]]
        Path to the request JSONL file and mapping of custom_id -> df index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "llm_labels_raw.jsonl"
    done_hashes = load_checkpoint(log_path)

    system_prompt = build_labeling_system_prompt()
    request_path = output_dir / "batch_requests.jsonl"
    id_to_idx: dict[str, int] = {}

    unlabeled_mask = (df["label"] == 1) & df["attack_category"].isna()

    with open(request_path, "w") as f:
        for idx in df.index[unlabeled_mask]:
            text = str(df.at[idx, "text"])
            sample_hash = _text_hash(text)
            if sample_hash in done_hashes:
                continue

            custom_id = sample_hash
            id_to_idx[custom_id] = idx
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request) + "\n")

    # Save id-to-idx mapping
    mapping_path = output_dir / "batch_id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_to_idx, f)

    logger.info("Created batch request file: %s (%d requests)", request_path, len(id_to_idx))
    return request_path, id_to_idx


def submit_batch(  # pragma: no cover
    request_path: Path,
) -> str:
    """Upload request file and submit an OpenAI batch job.

    Parameters
    ----------
    request_path : Path
        Path to the JSONL request file.

    Returns
    -------
    str
        Batch job ID.
    """
    client = openai.OpenAI()

    with open(request_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    logger.info("Uploaded file: %s", uploaded.id)

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info("Submitted batch: %s (status: %s)", batch.id, batch.status)
    return batch.id


def poll_batch(  # pragma: no cover
    batch_id: str,
    *,
    poll_interval: int = 30,
) -> str:
    """Poll a batch job until completion.

    Parameters
    ----------
    batch_id : str
        Batch job ID.
    poll_interval : int
        Seconds between polls.

    Returns
    -------
    str
        Output file ID.
    """
    client = openai.OpenAI()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0

        logger.info(
            "Batch %s: status=%s, completed=%d/%d, failed=%d",
            batch_id,
            status,
            completed,
            total,
            failed,
        )

        if status == "completed":
            if batch.output_file_id is None:
                raise RuntimeError(f"Batch {batch_id} completed but no output file.")
            return batch.output_file_id

        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

        time.sleep(poll_interval)


def download_and_apply_batch_results(  # pragma: no cover
    output_file_id: str,
    df: pd.DataFrame,
    output_dir: Path,
    *,
    confidence_threshold: float = 0.6,
    model: str = BATCH_MODEL,
) -> pd.DataFrame:
    """Download batch results and apply labels to the DataFrame.

    Parameters
    ----------
    output_file_id : str
        OpenAI file ID for the batch output.
    df : pd.DataFrame
        Dataset to apply labels to.
    output_dir : Path
        Directory for labeling artifacts.
    confidence_threshold : float
        Min confidence to apply label.
    model : str
        Model name for logging.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with labels applied.
    """
    client = openai.OpenAI()
    content = client.files.content(output_file_id)

    # Load id-to-idx mapping
    mapping_path = output_dir / "batch_id_mapping.json"
    with open(mapping_path) as f:
        id_to_idx: dict[str, int] = {k: int(v) for k, v in json.load(f).items()}

    log_path = output_dir / "llm_labels_raw.jsonl"
    results_path = output_dir / "batch_results_raw.jsonl"

    # Save raw results
    results_path.write_bytes(content.read())

    applied = 0
    uncertain = 0
    errors = 0

    for line in results_path.read_text().strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id", "")
        idx = id_to_idx.get(custom_id)
        if idx is None:
            continue

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            errors += 1
            continue

        raw_response = (choices[0].get("message", {}).get("content") or "").strip()
        slug, confidence = parse_llm_response(raw_response)

        text = str(df.at[idx, "text"])

        # Log to JSONL (same format as real-time)
        record = {
            "sample_hash": custom_id,
            "text_preview": text[:100],
            "raw_response": raw_response,
            "category_id": next(
                (cid for cid, s in CATEGORY_ID_TO_SLUG.items() if s == slug),
                None,
            ),
            "category_slug": slug,
            "confidence": confidence,
            "model": model,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "source": "batch_api",
        }
        append_log(log_path, record)

        if slug is not None and confidence is not None:
            if confidence >= confidence_threshold:
                df.at[idx, "attack_category"] = slug
                df.at[idx, "label_source"] = "gpt4o"
                df.at[idx, "confidence"] = confidence
                applied += 1
            else:
                df.at[idx, "label_source"] = "uncertain"
                df.at[idx, "confidence"] = confidence
                uncertain += 1

    logger.info(
        "Batch results: %d applied, %d uncertain, %d errors",
        applied,
        uncertain,
        errors,
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
    # Any label_source that isn't tier1_2 or uncertain is an LLM label
    llm_labeled = results_df[
        results_df["label_source"].notna()
        & ~results_df["label_source"].isin({"tier1_2", "uncertain"})
    ]
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
