"""Tests for manual review & spot-check module (LOATO-2A-03)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from loato_bench.data.review import (
    EXPORT_COLUMNS,
    LLM_LABEL_SOURCES,
    apply_manual_overrides,
    compute_error_rates,
    export_spot_check_samples,
    export_uncertain_pool,
    generate_coverage_report_v2,
    load_manual_overrides,
)
from loato_bench.data.taxonomy import _text_hash

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def review_df() -> pd.DataFrame:
    """Synthetic dataset with controlled mix of label sources and categories."""
    rows: list[dict[str, object]] = []

    # ~10 tier1_2 rows (various categories)
    for i in range(10):
        cats = ["instruction_override", "jailbreak_roleplay", "obfuscation_encoding"]
        rows.append(
            {
                "text": f"tier1_2 injection sample {i}",
                "label": 1,
                "source": "hackaprompt",
                "attack_category": cats[i % 3],
                "label_source": "tier1_2",
                "confidence": 1.0,
            }
        )

    # ~15 confident LLM rows across 3 categories
    for i in range(15):
        cats = ["instruction_override", "jailbreak_roleplay", "social_engineering"]
        rows.append(
            {
                "text": f"llm labeled injection sample {i}",
                "label": 1,
                "source": "deepset",
                "attack_category": cats[i % 3],
                "label_source": "llm",
                "confidence": 0.8,
            }
        )

    # ~5 uncertain rows
    for i in range(5):
        rows.append(
            {
                "text": f"uncertain injection sample {i}",
                "label": 1,
                "source": "gentel",
                "attack_category": "other",
                "label_source": "uncertain",
                "confidence": 0.4,
            }
        )

    # ~10 benign rows
    for i in range(10):
        rows.append(
            {
                "text": f"benign text sample {i}",
                "label": 0,
                "source": "deepset",
                "attack_category": None,
                "label_source": None,
                "confidence": None,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestExportSpotCheckSamples
# ---------------------------------------------------------------------------


class TestExportSpotCheckSamples:
    """Tests for export_spot_check_samples."""

    def test_returns_dataframe_with_export_columns(self, review_df: pd.DataFrame) -> None:
        """Exported frame has all expected columns."""
        result = export_spot_check_samples(review_df)
        assert list(result.columns) == EXPORT_COLUMNS

    def test_only_includes_llm_labeled_rows(self, review_df: pd.DataFrame) -> None:
        """Only rows with LLM label sources appear in the export."""
        result = export_spot_check_samples(review_df)
        # All rows should have been drawn from the LLM pool
        assert all(result["label_source"].isin(LLM_LABEL_SOURCES))

    def test_respects_n_per_category(self, review_df: pd.DataFrame) -> None:
        """No category exceeds n_per_category samples."""
        n = 3
        result = export_spot_check_samples(review_df, n_per_category=n)
        for _, group in result.groupby("attack_category"):
            assert len(group) <= n

    def test_text_truncated_to_500(self, review_df: pd.DataFrame) -> None:
        """Text column is capped at 500 characters."""
        # Add a row with very long text
        long_row = review_df.iloc[10:11].copy()
        long_row["text"] = "x" * 1000
        df = pd.concat([review_df, long_row], ignore_index=True)

        result = export_spot_check_samples(df)
        assert all(result["text"].str.len() <= 500)

    def test_sample_hash_present(self, review_df: pd.DataFrame) -> None:
        """Each row has a non-empty sample_hash."""
        result = export_spot_check_samples(review_df)
        assert result["sample_hash"].notna().all()
        assert (result["sample_hash"] != "").all()

    def test_correct_category_column_blank(self, review_df: pd.DataFrame) -> None:
        """The correct_category column is empty (for human filling)."""
        result = export_spot_check_samples(review_df)
        assert (result["correct_category"] == "").all()

    def test_empty_llm_pool_returns_empty(self) -> None:
        """Returns empty DataFrame when no LLM-labeled rows exist."""
        df = pd.DataFrame(
            {
                "text": ["hello"],
                "label": [1],
                "source": ["test"],
                "attack_category": ["other"],
                "label_source": ["tier1_2"],
                "confidence": [1.0],
            }
        )
        result = export_spot_check_samples(df)
        assert len(result) == 0
        assert list(result.columns) == EXPORT_COLUMNS

    def test_deterministic_with_seed(self, review_df: pd.DataFrame) -> None:
        """Same seed produces same output."""
        r1 = export_spot_check_samples(review_df, seed=123)
        r2 = export_spot_check_samples(review_df, seed=123)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestExportUncertainPool
# ---------------------------------------------------------------------------


class TestExportUncertainPool:
    """Tests for export_uncertain_pool."""

    def test_returns_only_uncertain_rows(self, review_df: pd.DataFrame) -> None:
        """Only rows with label_source='uncertain' are exported."""
        result = export_uncertain_pool(review_df)
        assert (result["label_source"] == "uncertain").all()

    def test_correct_count(self, review_df: pd.DataFrame) -> None:
        """Count matches number of uncertain rows in source data."""
        expected = (review_df["label_source"] == "uncertain").sum()
        result = export_uncertain_pool(review_df)
        assert len(result) == expected

    def test_has_export_columns(self, review_df: pd.DataFrame) -> None:
        """Exported frame has all expected columns."""
        result = export_uncertain_pool(review_df)
        assert list(result.columns) == EXPORT_COLUMNS

    def test_empty_when_no_uncertain(self) -> None:
        """Returns empty DataFrame when no uncertain rows exist."""
        df = pd.DataFrame(
            {
                "text": ["text"],
                "label": [1],
                "source": ["test"],
                "attack_category": ["other"],
                "label_source": ["llm"],
                "confidence": [0.9],
            }
        )
        result = export_uncertain_pool(df)
        assert len(result) == 0

    def test_sample_hash_matches_text(self, review_df: pd.DataFrame) -> None:
        """Sample hashes correspond to the truncated text's original."""
        result = export_uncertain_pool(review_df)
        for _, row in result.iterrows():
            # Hash is computed from original text (not truncated)
            original = review_df[review_df["text"].apply(_text_hash) == row["sample_hash"]]
            assert len(original) >= 1


# ---------------------------------------------------------------------------
# TestLoadManualOverrides
# ---------------------------------------------------------------------------


class TestLoadManualOverrides:
    """Tests for load_manual_overrides."""

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        """Valid CSV with slug categories loads correctly."""
        csv_path = tmp_path / "overrides.csv"
        csv_path.write_text("sample_hash,correct_category\nabc123,instruction_override\n")
        result = load_manual_overrides(csv_path)
        assert len(result) == 1
        assert result.at[0, "correct_category"] == "instruction_override"

    def test_normalizes_c_ids_to_slugs(self, tmp_path: Path) -> None:
        """Category IDs like 'C1' are converted to slugs."""
        csv_path = tmp_path / "overrides.csv"
        csv_path.write_text("sample_hash,correct_category\nabc123,C1\ndef456,C2\n")
        result = load_manual_overrides(csv_path)
        assert result.at[0, "correct_category"] == "instruction_override"
        assert result.at[1, "correct_category"] == "jailbreak_roleplay"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_manual_overrides(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path: Path) -> None:
        """ValueError when required columns are missing."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("sample_hash,wrong_col\nabc123,val\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_manual_overrides(csv_path)

    def test_raises_on_unknown_category(self, tmp_path: Path) -> None:
        """ValueError when an unknown category slug is found."""
        csv_path = tmp_path / "overrides.csv"
        csv_path.write_text("sample_hash,correct_category\nabc123,totally_fake_category\n")
        with pytest.raises(ValueError, match="Unknown categories"):
            load_manual_overrides(csv_path)

    def test_skips_blank_correct_category(self, tmp_path: Path) -> None:
        """Rows with empty correct_category are filtered out."""
        csv_path = tmp_path / "overrides.csv"
        csv_path.write_text("sample_hash,correct_category\nabc123,instruction_override\ndef456,\n")
        result = load_manual_overrides(csv_path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestApplyManualOverrides
# ---------------------------------------------------------------------------


class TestApplyManualOverrides:
    """Tests for apply_manual_overrides."""

    def test_updates_matched_rows(self, review_df: pd.DataFrame) -> None:
        """Matched rows get new category, label_source='manual', confidence=1.0."""
        # Pick a sample from the LLM pool
        text = "llm labeled injection sample 0"
        h = _text_hash(text)
        overrides = pd.DataFrame({"sample_hash": [h], "correct_category": ["social_engineering"]})
        result = apply_manual_overrides(review_df, overrides)

        matched = result[result["text"] == text]
        assert len(matched) == 1
        assert matched.iloc[0]["attack_category"] == "social_engineering"
        assert matched.iloc[0]["label_source"] == "manual"
        assert matched.iloc[0]["confidence"] == 1.0

    def test_does_not_modify_original(self, review_df: pd.DataFrame) -> None:
        """Original DataFrame is not changed."""
        original = review_df.copy()
        overrides = pd.DataFrame({"sample_hash": ["xxx"], "correct_category": ["other"]})
        apply_manual_overrides(review_df, overrides)
        pd.testing.assert_frame_equal(review_df, original)

    def test_unmatched_rows_unchanged(self, review_df: pd.DataFrame) -> None:
        """Rows that don't match any override stay the same."""
        overrides = pd.DataFrame(
            {"sample_hash": ["nonexistent_hash"], "correct_category": ["other"]}
        )
        result = apply_manual_overrides(review_df, overrides)
        # No manual label_source should appear
        assert (result["label_source"] != "manual").all() | result["label_source"].isna().all()

    def test_no_hash_column_leaks(self, review_df: pd.DataFrame) -> None:
        """Internal _hash column is not present in the output."""
        overrides = pd.DataFrame({"sample_hash": ["xxx"], "correct_category": ["other"]})
        result = apply_manual_overrides(review_df, overrides)
        assert "_hash" not in result.columns


# ---------------------------------------------------------------------------
# TestComputeErrorRates
# ---------------------------------------------------------------------------


class TestComputeErrorRates:
    """Tests for compute_error_rates."""

    def test_zero_errors(self) -> None:
        """No errors when all correct_category match attack_category."""
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override", "jailbreak_roleplay"],
                "correct_category": ["instruction_override", "jailbreak_roleplay"],
            }
        )
        result = compute_error_rates(df)
        assert result["overall_error_rate"] == 0.0
        assert result["high_error_categories"] == []

    def test_all_errors(self) -> None:
        """100% error rate when none match."""
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override", "instruction_override"],
                "correct_category": ["jailbreak_roleplay", "social_engineering"],
            }
        )
        result = compute_error_rates(df)
        assert result["overall_error_rate"] == 1.0

    def test_skips_blank_rows(self) -> None:
        """Rows with blank correct_category are ignored."""
        df = pd.DataFrame(
            {
                "attack_category": ["instruction_override", "jailbreak_roleplay"],
                "correct_category": ["instruction_override", ""],
            }
        )
        result = compute_error_rates(df)
        assert result["per_category"]["instruction_override"]["total"] == 1
        assert "jailbreak_roleplay" not in result["per_category"]

    def test_flags_high_error_categories(self) -> None:
        """Categories with >20% error rate are flagged."""
        df = pd.DataFrame(
            {
                "attack_category": ["cat_a"] * 10,
                "correct_category": ["cat_a"] * 7 + ["cat_b"] * 3,
            }
        )
        result = compute_error_rates(df)
        assert "cat_a" in result["high_error_categories"]

    def test_empty_dataframe(self) -> None:
        """Empty input returns zero rates."""
        df = pd.DataFrame(columns=["attack_category", "correct_category"])
        result = compute_error_rates(df)
        assert result["overall_error_rate"] == 0.0
        assert result["per_category"] == {}


# ---------------------------------------------------------------------------
# TestGenerateCoverageReportV2
# ---------------------------------------------------------------------------


class TestGenerateCoverageReportV2:
    """Tests for generate_coverage_report_v2."""

    def test_counts_injection_samples(self, review_df: pd.DataFrame) -> None:
        """Total injection count matches label=1 rows."""
        result = generate_coverage_report_v2(review_df)
        expected = (review_df["label"] == 1).sum()
        assert result["total_injection"] == expected

    def test_per_category_present(self, review_df: pd.DataFrame) -> None:
        """Per-category dict is non-empty for datasets with categories."""
        result = generate_coverage_report_v2(review_df)
        assert len(result["per_category"]) > 0

    def test_per_label_source_present(self, review_df: pd.DataFrame) -> None:
        """Per-label-source counts are included."""
        result = generate_coverage_report_v2(review_df)
        assert "tier1_2" in result["per_label_source"]
        assert "llm" in result["per_label_source"]

    def test_coverage_percentage(self, review_df: pd.DataFrame) -> None:
        """Coverage percentage is computed correctly."""
        result = generate_coverage_report_v2(review_df)
        total = (review_df["label"] == 1).sum()
        mapped = review_df[(review_df["label"] == 1) & review_df["attack_category"].notna()]
        expected_pct = round(len(mapped) / total * 100, 2)
        assert result["coverage_pct"] == expected_pct

    def test_meets_threshold_flag(self) -> None:
        """Boolean flag correctly reflects the 90% threshold."""
        df = pd.DataFrame(
            {
                "text": [f"t{i}" for i in range(10)],
                "label": [1] * 10,
                "attack_category": ["instruction_override"] * 9 + [None],
                "label_source": ["llm"] * 10,
            }
        )
        result = generate_coverage_report_v2(df)
        assert result["coverage_pct"] == 90.0
        assert result["meets_90_pct_threshold"] is True

    def test_empty_dataframe(self) -> None:
        """Empty dataset returns zeros."""
        df = pd.DataFrame(columns=["text", "label", "attack_category", "label_source"])
        result = generate_coverage_report_v2(df)
        assert result["total_injection"] == 0
        assert result["coverage_pct"] == 0.0
