"""Tests for loato_bench.data.llm_labeler — GPT-4o-mini batch categorization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from loato_bench.data.llm_labeler import (
    append_log,
    build_labeling_system_prompt,
    label_samples,
    load_checkpoint,
    parse_llm_response,
    validate_distribution,
)
from loato_bench.data.taxonomy_spec import CATEGORY_ID_TO_SLUG, TAXONOMY_V1

# ---------------------------------------------------------------------------
# TestBuildLabelingSystemPrompt
# ---------------------------------------------------------------------------


class TestBuildLabelingSystemPrompt:
    """Tests for build_labeling_system_prompt."""

    def test_contains_all_category_ids(self) -> None:
        prompt = build_labeling_system_prompt()
        for cid in TAXONOMY_V1:
            assert cid in prompt, f"Missing category ID {cid}"

    def test_contains_all_slugs(self) -> None:
        prompt = build_labeling_system_prompt()
        for spec in TAXONOMY_V1.values():
            assert spec.slug in prompt

    def test_contains_json_format_instruction(self) -> None:
        prompt = build_labeling_system_prompt()
        assert '"category"' in prompt
        assert '"confidence"' in prompt
        assert "JSON" in prompt

    def test_contains_boundary_rules(self) -> None:
        prompt = build_labeling_system_prompt()
        assert "Boundary Rules" in prompt

    def test_built_from_taxonomy_v1(self) -> None:
        """Verify descriptions come from TAXONOMY_V1, not hardcoded."""
        prompt = build_labeling_system_prompt()
        for spec in TAXONOMY_V1.values():
            assert spec.mechanism in prompt

    def test_contains_signal_phrases(self) -> None:
        prompt = build_labeling_system_prompt()
        # C1 has signal phrases
        for phrase in TAXONOMY_V1["C1"].signal_phrases:
            assert phrase in prompt


# ---------------------------------------------------------------------------
# TestParseLlmResponse
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for parse_llm_response."""

    def test_valid_response(self) -> None:
        raw = '{"category": "C1", "confidence": 0.87}'
        slug, confidence = parse_llm_response(raw)
        assert slug == "instruction_override"
        assert confidence == 0.87

    def test_all_valid_categories(self) -> None:
        for cid, expected_slug in CATEGORY_ID_TO_SLUG.items():
            raw = json.dumps({"category": cid, "confidence": 0.5})
            slug, confidence = parse_llm_response(raw)
            assert slug == expected_slug
            assert confidence == 0.5

    def test_invalid_category(self) -> None:
        raw = '{"category": "C99", "confidence": 0.5}'
        slug, confidence = parse_llm_response(raw)
        assert slug is None
        assert confidence is None

    def test_malformed_json(self) -> None:
        slug, confidence = parse_llm_response("not json at all")
        assert slug is None
        assert confidence is None

    def test_missing_category_key(self) -> None:
        raw = '{"confidence": 0.5}'
        slug, confidence = parse_llm_response(raw)
        assert slug is None
        assert confidence is None

    def test_missing_confidence_key(self) -> None:
        raw = '{"category": "C1"}'
        slug, confidence = parse_llm_response(raw)
        assert slug is None
        assert confidence is None

    def test_confidence_out_of_range_high(self) -> None:
        raw = '{"category": "C1", "confidence": 1.5}'
        slug, confidence = parse_llm_response(raw)
        assert slug is None
        assert confidence is None

    def test_confidence_out_of_range_low(self) -> None:
        raw = '{"category": "C1", "confidence": -0.1}'
        slug, confidence = parse_llm_response(raw)
        assert slug is None
        assert confidence is None

    def test_confidence_as_string(self) -> None:
        raw = '{"category": "C1", "confidence": "0.9"}'
        slug, confidence = parse_llm_response(raw)
        assert slug == "instruction_override"
        assert confidence == 0.9

    def test_empty_string(self) -> None:
        slug, confidence = parse_llm_response("")
        assert slug is None
        assert confidence is None

    def test_none_input(self) -> None:
        slug, confidence = parse_llm_response(None)  # type: ignore[arg-type]
        assert slug is None
        assert confidence is None

    def test_boundary_confidence_zero(self) -> None:
        raw = '{"category": "C1", "confidence": 0.0}'
        slug, confidence = parse_llm_response(raw)
        assert slug == "instruction_override"
        assert confidence == 0.0

    def test_boundary_confidence_one(self) -> None:
        raw = '{"category": "C1", "confidence": 1.0}'
        slug, confidence = parse_llm_response(raw)
        assert slug == "instruction_override"
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# TestLoadCheckpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    """Tests for load_checkpoint."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_checkpoint(tmp_path / "nonexistent.jsonl")
        assert result == set()

    def test_file_with_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        records = [
            {"sample_hash": "abc123", "raw_response": "..."},
            {"sample_hash": "def456", "raw_response": "..."},
        ]
        log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        result = load_checkpoint(log_path)
        assert result == {"abc123", "def456"}

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        lines = [
            json.dumps({"sample_hash": "good1"}),
            "this is not json",
            json.dumps({"sample_hash": "good2"}),
        ]
        log_path.write_text("\n".join(lines) + "\n")

        result = load_checkpoint(log_path)
        assert result == {"good1", "good2"}

    def test_empty_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        result = load_checkpoint(log_path)
        assert result == set()

    def test_records_without_hash(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text(json.dumps({"other_key": "value"}) + "\n")
        result = load_checkpoint(log_path)
        assert result == set()


# ---------------------------------------------------------------------------
# TestAppendLog
# ---------------------------------------------------------------------------


class TestAppendLog:
    """Tests for append_log."""

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        log_path = tmp_path / "subdir" / "log.jsonl"
        record = {"sample_hash": "abc", "data": 1}
        append_log(log_path, record)

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == record

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        append_log(log_path, {"sample_hash": "first"})
        append_log(log_path, {"sample_hash": "second"})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["sample_hash"] == "first"
        assert json.loads(lines[1])["sample_hash"] == "second"


# ---------------------------------------------------------------------------
# TestLabelSamples
# ---------------------------------------------------------------------------


def _make_df(
    n_labeled: int = 3,
    n_unlabeled: int = 5,
    n_benign: int = 2,
) -> pd.DataFrame:
    """Create a test DataFrame with labeled, unlabeled injection, and benign samples."""
    rows = []
    for i in range(n_labeled):
        rows.append(
            {
                "text": f"labeled injection {i}",
                "label": 1,
                "source": "test",
                "attack_category": "instruction_override",
            }
        )
    for i in range(n_unlabeled):
        rows.append(
            {
                "text": f"unlabeled injection {i}",
                "label": 1,
                "source": "test",
                "attack_category": None,
            }
        )
    for i in range(n_benign):
        rows.append(
            {
                "text": f"benign sample {i}",
                "label": 0,
                "source": "test",
                "attack_category": None,
            }
        )
    return pd.DataFrame(rows)


def _mock_openai_response(category: str = "C2", confidence: float = 0.85) -> MagicMock:
    """Create a mock OpenAI response."""
    choice = MagicMock()
    choice.message.content = json.dumps({"category": category, "confidence": confidence})
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLabelSamples:
    """Tests for label_samples."""

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_only_unlabeled_injections_sent(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Only injection samples with null attack_category get API calls."""
        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response()

        df = _make_df(n_labeled=2, n_unlabeled=3, n_benign=2)
        result = label_samples(df, output_dir=tmp_path)

        # Should have made 3 API calls (one per unlabeled injection)
        assert mock_client.chat.completions.create.call_count == 3
        # Benign samples should not have label_source set to gpt4o_mini
        benign = result[result["label"] == 0]
        assert (benign["label_source"] != "gpt4o_mini").all()

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_checkpoint_resume_skips_done(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Samples already in checkpoint log are skipped."""
        from loato_bench.data.taxonomy import _text_hash

        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response()

        df = _make_df(n_labeled=0, n_unlabeled=3, n_benign=0)

        # Pre-populate checkpoint with hash of first sample
        log_path = tmp_path / "llm_labels_raw.jsonl"
        first_hash = _text_hash(df.iloc[0]["text"])
        append_log(log_path, {"sample_hash": first_hash})

        label_samples(df, output_dir=tmp_path)

        # Should skip the first sample, process 2
        assert mock_client.chat.completions.create.call_count == 2

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_confidence_threshold_filtering(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Labels below threshold get 'uncertain' label_source."""
        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Return low-confidence response
        mock_client.chat.completions.create.return_value = _mock_openai_response(confidence=0.3)

        df = _make_df(n_labeled=0, n_unlabeled=2, n_benign=0)
        result = label_samples(df, confidence_threshold=0.6, output_dir=tmp_path)

        # All should be "uncertain"
        unlabeled = result[result["label"] == 1]
        assert (unlabeled["label_source"] == "uncertain").all()
        # attack_category should remain None
        assert unlabeled["attack_category"].isna().all()

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_max_calls_respected(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Stops after max_calls API calls."""
        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response()

        df = _make_df(n_labeled=0, n_unlabeled=10, n_benign=0)
        label_samples(df, max_calls=3, output_dir=tmp_path)

        assert mock_client.chat.completions.create.call_count == 3

    def test_dry_run_no_api_calls(self, tmp_path: Path) -> None:
        """Dry run makes no API calls."""
        df = _make_df(n_labeled=0, n_unlabeled=5, n_benign=0)

        with patch("loato_bench.data.llm_labeler.openai.OpenAI") as mock_cls:
            result = label_samples(df, dry_run=True, output_dir=tmp_path)
            mock_cls.assert_not_called()

        # DataFrame returned unchanged (no label_source set to gpt4o_mini)
        assert "label_source" in result.columns
        assert (result["label_source"] != "gpt4o_mini").all()

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_jsonl_written_before_labels(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """JSONL log is written for every API call."""
        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response()

        df = _make_df(n_labeled=0, n_unlabeled=3, n_benign=0)
        label_samples(df, output_dir=tmp_path)

        log_path = tmp_path / "llm_labels_raw.jsonl"
        assert log_path.exists()
        lines = [line for line in log_path.read_text().strip().split("\n") if line]
        assert len(lines) == 3

        # Each line should be valid JSON with required fields
        for line in lines:
            record = json.loads(line)
            assert "sample_hash" in record
            assert "raw_response" in record
            assert "timestamp" in record
            assert "model" in record

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_no_unlabeled_samples(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """No API calls when all injections are already labeled."""
        df = _make_df(n_labeled=5, n_unlabeled=0, n_benign=2)
        result = label_samples(df, output_dir=tmp_path)

        mock_openai_cls.assert_not_called()
        assert len(result) == 7

    @patch("loato_bench.data.llm_labeler.openai.OpenAI")
    @patch("loato_bench.data.llm_labeler.load_llm_config")
    def test_labels_applied_above_threshold(
        self, mock_config: MagicMock, mock_openai_cls: MagicMock, tmp_path: Path
    ) -> None:
        """High-confidence labels are applied to attack_category."""
        mock_config.return_value = MagicMock(model="gpt-4o-mini", temperature=0.0)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            category="C3", confidence=0.95
        )

        df = _make_df(n_labeled=0, n_unlabeled=2, n_benign=0)
        result = label_samples(df, confidence_threshold=0.6, output_dir=tmp_path)

        labeled = result[result["label_source"] == "gpt4o_mini"]
        assert len(labeled) == 2
        assert (labeled["attack_category"] == "obfuscation_encoding").all()


# ---------------------------------------------------------------------------
# TestValidateDistribution
# ---------------------------------------------------------------------------


class TestValidateDistribution:
    """Tests for validate_distribution."""

    def test_balanced_distribution_no_warnings(self) -> None:
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override"] * 30
                + ["jailbreak_roleplay"] * 30
                + ["obfuscation_encoding"] * 40,
                "label_source": ["gpt4o_mini"] * 100,
            }
        )
        report = validate_distribution(df)
        assert report["total_llm_labeled"] == 100
        assert report["warnings"] == []

    def test_single_category_above_60_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override"] * 80 + ["jailbreak_roleplay"] * 20,
                "label_source": ["gpt4o_mini"] * 100,
            }
        )
        report = validate_distribution(df)
        assert len(report["warnings"]) == 1
        assert "instruction_override" in report["warnings"][0]

    def test_no_llm_labels(self) -> None:
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override"] * 10,
                "label_source": ["tier1_2"] * 10,
            }
        )
        report = validate_distribution(df)
        assert report["total_llm_labeled"] == 0
        assert report["warnings"] == []

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"attack_category": [], "label_source": []})
        report = validate_distribution(df)
        assert report["total_llm_labeled"] == 0

    def test_category_counts_correct(self) -> None:
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override"] * 10 + ["jailbreak_roleplay"] * 5,
                "label_source": ["gpt4o_mini"] * 15,
            }
        )
        report = validate_distribution(df)
        assert report["category_counts"]["instruction_override"] == 10
        assert report["category_counts"]["jailbreak_roleplay"] == 5

    def test_only_counts_gpt4o_mini_labels(self) -> None:
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override"] * 50 + ["instruction_override"] * 50,
                "label_source": ["gpt4o_mini"] * 50 + ["tier1_2"] * 50,
            }
        )
        report = validate_distribution(df)
        # Only the 50 gpt4o_mini labels should be counted
        assert report["total_llm_labeled"] == 50
