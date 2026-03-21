"""Tests for analysis/report.py — results consolidation, tables, figures, statistics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loato_bench.analysis.report import (
    ResultRow,
    build_master_table,
    build_per_fold_table,
    generate_narrative_summary,
    generate_report,
    load_all_results,
    load_result_json,
    plot_cv_vs_loato_comparison,
    plot_delta_f1_heatmap,
    plot_per_fold_f1,
    run_significance_tests,
    table_to_latex,
    table_to_markdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result_json(
    experiment: str,
    embedding: str,
    classifier: str,
    folds: list[dict],
) -> dict:
    """Build a minimal result JSON dict matching ExperimentResult.to_dict() format."""
    f1s = [f["metrics"]["f1"]["value"] for f in folds]
    return {
        "experiment": experiment,
        "embedding": embedding,
        "classifier": classifier,
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "folds": folds,
    }


def _make_fold(
    fold_id: int,
    f1: float,
    held_out: str | None = None,
    precision: float = 0.9,
    recall: float = 0.85,
    accuracy: float = 0.9,
    auc_roc: float = 0.95,
) -> dict:
    """Build a minimal fold dict."""
    return {
        "fold_id": fold_id,
        "held_out_category": held_out,
        "train_size": 1000,
        "test_size": 200,
        "metrics": {
            "accuracy": {"value": accuracy, "ci_lower": None, "ci_upper": None},
            "precision": {"value": precision, "ci_lower": None, "ci_upper": None},
            "recall": {"value": recall, "ci_lower": None, "ci_upper": None},
            "f1": {"value": f1, "ci_lower": None, "ci_upper": None},
            "auc_roc": {"value": auc_roc, "ci_lower": None, "ci_upper": None},
            "auc_pr": {"value": 0.90, "ci_lower": None, "ci_upper": None},
        },
    }


@pytest.fixture()
def cv_result_json() -> dict:
    """Standard CV result JSON with 5 folds."""
    folds = [_make_fold(i, 0.95 + i * 0.005) for i in range(5)]
    return _make_result_json("standard_cv", "minilm", "logreg", folds)


@pytest.fixture()
def loato_result_json() -> dict:
    """LOATO result JSON with 6 folds (held-out categories)."""
    categories = [
        "instruction_override",
        "jailbreak_roleplay",
        "obfuscation_encoding",
        "information_extraction",
        "social_engineering",
        "other",
    ]
    folds = [_make_fold(i, 0.80 + i * 0.02, held_out=cat) for i, cat in enumerate(categories)]
    return _make_result_json("loato", "minilm", "logreg", folds)


@pytest.fixture()
def results_dir(tmp_path: Path, cv_result_json: dict, loato_result_json: dict) -> Path:
    """Directory with standard_cv and loato result JSONs."""
    cv_path = tmp_path / "standard_cv_minilm_logreg.json"
    cv_path.write_text(json.dumps(cv_result_json))

    loato_path = tmp_path / "loato_minilm_logreg.json"
    loato_path.write_text(json.dumps(loato_result_json))

    return tmp_path


@pytest.fixture()
def sample_results() -> list[ResultRow]:
    """Two ResultRows: one standard_cv, one loato."""
    cv = ResultRow(
        experiment="standard_cv",
        embedding="minilm",
        classifier="logreg",
        mean_f1=0.96,
        std_f1=0.005,
        fold_f1s=[0.95, 0.955, 0.960, 0.965, 0.970],
        fold_categories=[None, None, None, None, None],
        precision=0.95,
        recall=0.93,
        accuracy=0.94,
        auc_roc=0.98,
    )
    loato = ResultRow(
        experiment="loato",
        embedding="minilm",
        classifier="logreg",
        mean_f1=0.85,
        std_f1=0.04,
        fold_f1s=[0.80, 0.82, 0.85, 0.88, 0.90, 0.85],
        fold_categories=[
            "instruction_override",
            "jailbreak_roleplay",
            "obfuscation_encoding",
            "information_extraction",
            "social_engineering",
            "other",
        ],
        precision=0.88,
        recall=0.82,
        accuracy=0.85,
        auc_roc=0.92,
    )
    return [cv, loato]


# ---------------------------------------------------------------------------
# TestLoadResultJson
# ---------------------------------------------------------------------------


class TestLoadResultJson:
    """Tests for load_result_json."""

    def test_loads_valid_json(self, tmp_path: Path, cv_result_json: dict) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps(cv_result_json))

        row = load_result_json(path)
        assert row.experiment == "standard_cv"
        assert row.embedding == "minilm"
        assert row.classifier == "logreg"
        assert len(row.fold_f1s) == 5
        assert row.mean_f1 == pytest.approx(np.mean(row.fold_f1s), abs=1e-6)

    def test_loads_loato_with_categories(self, tmp_path: Path, loato_result_json: dict) -> None:
        path = tmp_path / "loato.json"
        path.write_text(json.dumps(loato_result_json))

        row = load_result_json(path)
        assert row.experiment == "loato"
        assert len(row.fold_categories) == 6
        assert all(c is not None for c in row.fold_categories)

    def test_averages_metrics_across_folds(self, tmp_path: Path, cv_result_json: dict) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps(cv_result_json))

        row = load_result_json(path)
        assert row.precision > 0
        assert row.recall > 0
        assert row.accuracy > 0
        assert row.auc_roc > 0


# ---------------------------------------------------------------------------
# TestLoadAllResults
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    """Tests for load_all_results."""

    def test_loads_multiple_jsons(self, results_dir: Path) -> None:
        rows = load_all_results(results_dir)
        assert len(rows) == 2
        experiments = {r.experiment for r in rows}
        assert experiments == {"standard_cv", "loato"}

    def test_skips_all_results_file(self, results_dir: Path) -> None:
        # Write an aggregated file that should be skipped
        agg_path = results_dir / "standard_cv_all_results.json"
        agg_path.write_text("[]")

        rows = load_all_results(results_dir)
        assert len(rows) == 2

    def test_skips_invalid_json(self, results_dir: Path) -> None:
        bad_path = results_dir / "bad_result.json"
        bad_path.write_text("{invalid json")

        rows = load_all_results(results_dir)
        assert len(rows) == 2  # Only the valid ones

    def test_empty_directory(self, tmp_path: Path) -> None:
        rows = load_all_results(tmp_path)
        assert rows == []


# ---------------------------------------------------------------------------
# TestBuildMasterTable
# ---------------------------------------------------------------------------


class TestBuildMasterTable:
    """Tests for build_master_table."""

    def test_has_expected_columns(self, sample_results: list[ResultRow]) -> None:
        df = build_master_table(sample_results)
        assert "Embedding" in df.columns
        assert "Classifier" in df.columns
        assert "CV_F1" in df.columns
        assert "LOATO_F1" in df.columns
        assert "Delta_F1" in df.columns

    def test_computes_delta_f1(self, sample_results: list[ResultRow]) -> None:
        df = build_master_table(sample_results)
        row = df.iloc[0]
        expected_delta = row["CV_F1"] - row["LOATO_F1"]
        assert row["Delta_F1"] == pytest.approx(expected_delta)

    def test_positive_delta_means_cv_better(self, sample_results: list[ResultRow]) -> None:
        df = build_master_table(sample_results)
        # CV (0.96) > LOATO (0.85), so delta should be positive
        assert df.iloc[0]["Delta_F1"] > 0

    def test_empty_results(self) -> None:
        df = build_master_table([])
        assert df.empty


# ---------------------------------------------------------------------------
# TestBuildPerFoldTable
# ---------------------------------------------------------------------------


class TestBuildPerFoldTable:
    """Tests for build_per_fold_table."""

    def test_has_category_columns(self, sample_results: list[ResultRow]) -> None:
        df = build_per_fold_table(sample_results)
        assert not df.empty
        assert "instruction_override" in df.columns
        assert "Mean" in df.columns

    def test_excludes_standard_cv(self, sample_results: list[ResultRow]) -> None:
        # Only LOATO results should appear
        df = build_per_fold_table(sample_results)
        assert len(df) == 1  # Only the loato row

    def test_no_loato_results(self) -> None:
        cv_only = [
            ResultRow(
                experiment="standard_cv",
                embedding="minilm",
                classifier="logreg",
                mean_f1=0.96,
                std_f1=0.005,
            )
        ]
        df = build_per_fold_table(cv_only)
        assert df.empty


# ---------------------------------------------------------------------------
# TestRunSignificanceTests
# ---------------------------------------------------------------------------


class TestRunSignificanceTests:
    """Tests for run_significance_tests."""

    def test_returns_results_for_matched_pairs(self, sample_results: list[ResultRow]) -> None:
        sig = run_significance_tests(sample_results)
        assert len(sig) == 1
        assert sig[0].embedding == "MiniLM (384d)"
        assert sig[0].classifier == "LogReg"

    def test_has_p_value(self, sample_results: list[ResultRow]) -> None:
        sig = run_significance_tests(sample_results)
        assert 0.0 <= sig[0].p_value <= 1.0

    def test_reports_significance_flag(self, sample_results: list[ResultRow]) -> None:
        sig = run_significance_tests(sample_results)
        assert isinstance(sig[0].significant, bool)

    def test_no_results_without_both_experiments(self) -> None:
        cv_only = [
            ResultRow(
                experiment="standard_cv",
                embedding="minilm",
                classifier="logreg",
                mean_f1=0.96,
                std_f1=0.005,
                fold_f1s=[0.95, 0.96, 0.97],
            )
        ]
        sig = run_significance_tests(cv_only)
        assert sig == []


# ---------------------------------------------------------------------------
# TestTableExport
# ---------------------------------------------------------------------------


class TestTableExport:
    """Tests for table_to_markdown and table_to_latex."""

    def test_markdown_contains_header(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        md = table_to_markdown(df, title="Test Table")
        assert "## Test Table" in md
        assert "A" in md
        assert "B" in md

    def test_latex_contains_tabular(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        latex = table_to_latex(df)
        assert "\\begin{tabular}" in latex

    def test_latex_with_caption(self) -> None:
        df = pd.DataFrame({"A": [1.0]})
        latex = table_to_latex(df, caption="My Caption", label="tab:test")
        assert "\\caption{My Caption}" in latex
        assert "\\label{tab:test}" in latex

    def test_markdown_formats_floats(self) -> None:
        df = pd.DataFrame({"F1": [0.123456789]})
        md = table_to_markdown(df)
        assert "0.1235" in md  # Rounded to 4 decimals


# ---------------------------------------------------------------------------
# TestPlots
# ---------------------------------------------------------------------------


class TestPlots:
    """Tests for plot functions (save without error)."""

    def test_delta_f1_heatmap(self, sample_results: list[ResultRow], tmp_path: Path) -> None:
        master = build_master_table(sample_results)
        output = tmp_path / "heatmap.png"
        plot_delta_f1_heatmap(master, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_per_fold_f1(self, sample_results: list[ResultRow], tmp_path: Path) -> None:
        output = tmp_path / "per_fold.png"
        plot_per_fold_f1(sample_results, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_cv_vs_loato_comparison(self, sample_results: list[ResultRow], tmp_path: Path) -> None:
        master = build_master_table(sample_results)
        output = tmp_path / "comparison.png"
        plot_cv_vs_loato_comparison(master, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_per_fold_skips_without_loato(self, tmp_path: Path) -> None:
        cv_only = [
            ResultRow(
                experiment="standard_cv",
                embedding="minilm",
                classifier="logreg",
                mean_f1=0.96,
                std_f1=0.005,
            )
        ]
        output = tmp_path / "per_fold.png"
        plot_per_fold_f1(cv_only, output)
        assert not output.exists()  # Should skip and not create file


# ---------------------------------------------------------------------------
# TestNarrativeSummary
# ---------------------------------------------------------------------------


class TestNarrativeSummary:
    """Tests for generate_narrative_summary."""

    def test_produces_markdown(self, sample_results: list[ResultRow]) -> None:
        master = build_master_table(sample_results)
        per_fold = build_per_fold_table(sample_results)
        sig = run_significance_tests(sample_results)

        summary = generate_narrative_summary(master, per_fold, sig)
        assert "# LOATO-4B-01" in summary
        assert "ΔF1" in summary or "Delta" in summary

    def test_empty_table(self) -> None:
        summary = generate_narrative_summary(pd.DataFrame(), pd.DataFrame(), [])
        assert "No results available" in summary


# ---------------------------------------------------------------------------
# TestGenerateReport (integration)
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Integration test for generate_report."""

    def test_generates_all_outputs(self, results_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "report_output"
        outputs = generate_report(results_dir, output_dir)

        assert len(outputs) > 0
        assert (output_dir / "figures").is_dir()
        assert (output_dir / "tables").is_dir()
        assert (output_dir / "4b_01_summary.md").exists()
        assert (output_dir / "4b_01_raw_data.json").exists()

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        outputs = generate_report(empty_dir, output_dir)
        assert outputs == {}
