"""Tests for split generation module (Sprint 2A)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loato_bench.data import splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n_benign: int = 100,
    categories: dict[str, int] | None = None,
    languages: dict[str, int] | None = None,
    n_indirect: int = 0,
) -> pd.DataFrame:
    """Build a synthetic dataset for split tests.

    Parameters
    ----------
    n_benign : int
        Number of benign samples.
    categories : dict[str, int]
        Mapping of category name → count of injection samples.
    languages : dict[str, int]
        Extra non-English injection samples per language code.
    n_indirect : int
        Number of indirect injection samples (drawn from first category).
    """
    rows: list[dict[str, object]] = []

    # Benign
    for i in range(n_benign):
        rows.append(
            {
                "text": f"benign text {i}",
                "label": 0,
                "source": "deepset",
                "attack_category": None,
                "language": "en",
                "is_indirect": False,
            }
        )

    # Injection categories
    if categories is None:
        categories = {"instruction_override": 300, "jailbreak_roleplay": 250}

    for cat, count in categories.items():
        for j in range(count):
            rows.append(
                {
                    "text": f"{cat} attack text {j}",
                    "label": 1,
                    "source": "hackaprompt",
                    "attack_category": cat,
                    "language": "en",
                    "is_indirect": j < n_indirect and cat == list(categories.keys())[0],
                }
            )

    # Non-English injection
    if languages:
        for lang, count in languages.items():
            for k in range(count):
                rows.append(
                    {
                        "text": f"foreign attack {lang} {k}",
                        "label": 1,
                        "source": "open_prompt",
                        "attack_category": "instruction_override",
                        "language": lang,
                        "is_indirect": False,
                    }
                )

    return pd.DataFrame(rows)


# ===========================================================================
# TestGenerateStandardCVSplits
# ===========================================================================


class TestGenerateStandardCVSplits:
    """Tests for generate_standard_cv_splits."""

    def test_returns_dict_with_correct_structure(self) -> None:
        df = _make_df()
        result = splits.generate_standard_cv_splits(df)
        assert "experiment" in result
        assert result["experiment"] == "standard_cv"
        assert "n_folds" in result
        assert "seed" in result
        assert "folds" in result

    def test_generates_n_folds(self) -> None:
        df = _make_df()
        result = splits.generate_standard_cv_splits(df, n_folds=3)
        assert len(result["folds"]) == 3

    def test_no_index_overlap(self) -> None:
        df = _make_df()
        result = splits.generate_standard_cv_splits(df)
        for fold in result["folds"]:
            train = set(fold["train_indices"])
            test = set(fold["test_indices"])
            assert train.isdisjoint(test), "Train/test overlap in fold"

    def test_all_indices_covered(self) -> None:
        df = _make_df()
        result = splits.generate_standard_cv_splits(df)
        all_test = set()
        for fold in result["folds"]:
            all_test.update(fold["test_indices"])
        assert all_test == set(range(len(df)))

    def test_stratification_preserves_label_ratio(self) -> None:
        df = _make_df(n_benign=200, categories={"cat1": 200})
        result = splits.generate_standard_cv_splits(df, stratify_by=["label"])
        overall_ratio = (df["label"] == 1).mean()
        for fold in result["folds"]:
            fold_labels = df.iloc[fold["test_indices"]]["label"]
            fold_ratio = (fold_labels == 1).mean()
            assert abs(fold_ratio - overall_ratio) < 0.1

    def test_seed_deterministic(self) -> None:
        df = _make_df()
        r1 = splits.generate_standard_cv_splits(df, seed=42)
        r2 = splits.generate_standard_cv_splits(df, seed=42)
        assert r1["folds"][0]["test_indices"] == r2["folds"][0]["test_indices"]

    def test_single_class_raises(self) -> None:
        df = pd.DataFrame(
            {
                "text": [f"t{i}" for i in range(10)],
                "label": [1] * 10,
                "attack_category": ["cat"] * 10,
            }
        )
        with pytest.raises(ValueError, match="at least 2"):
            splits.generate_standard_cv_splits(df, stratify_by=["label"])

    def test_empty_raises(self) -> None:
        df = pd.DataFrame(columns=["text", "label", "attack_category"])
        with pytest.raises(ValueError, match="empty"):
            splits.generate_standard_cv_splits(df)

    def test_excluded_categories_removes_from_splits(self) -> None:
        """Excluded categories should be absent from all fold indices."""
        df = _make_df(categories={"cat1": 300, "cat2": 250, "other": 100})
        result = splits.generate_standard_cv_splits(df, excluded_categories={"other"})
        # Reconstruct the filtered df to check
        benign_mask = df["label"] == 0
        kept_mask = benign_mask | ~df["attack_category"].isin({"other"})
        expected_n = kept_mask.sum()
        all_indices = set()
        for fold in result["folds"]:
            all_indices.update(fold["test_indices"])
        assert len(all_indices) == expected_n

    def test_excluded_categories_preserves_benign(self) -> None:
        """Benign samples should be kept even when categories are excluded."""
        df = _make_df(n_benign=50, categories={"cat1": 100, "other": 80})
        result = splits.generate_standard_cv_splits(df, excluded_categories={"other"})
        total = sum(len(f["train_indices"]) + len(f["test_indices"]) for f in result["folds"])
        # 5 folds: each sample appears once in test, 4 times in train → total = 5 * n_samples
        n_kept = 50 + 100  # benign + cat1
        assert total == 5 * n_kept


# ===========================================================================
# TestGenerateLoatoSplits
# ===========================================================================


class TestGenerateLoatoSplits:
    """Tests for generate_loato_splits."""

    def test_returns_dict_with_correct_structure(self) -> None:
        df = _make_df()
        result = splits.generate_loato_splits(df, min_samples=50)
        assert result["experiment"] == "loato"
        assert "seed" in result
        assert "folds" in result
        assert len(result["folds"]) > 0

    def test_creates_one_fold_per_viable_category(self) -> None:
        df = _make_df(categories={"cat1": 250, "cat2": 250, "cat3": 250})
        result = splits.generate_loato_splits(df, min_samples=200)
        assert len(result["folds"]) == 3

    def test_skips_categories_below_min_samples(self) -> None:
        df = _make_df(categories={"big": 300, "small": 50})
        result = splits.generate_loato_splits(df, min_samples=200)
        assert len(result["folds"]) == 1
        assert result["folds"][0]["held_out_category"] == "big"

    def test_train_contains_other_categories(self) -> None:
        df = _make_df(categories={"cat1": 300, "cat2": 300})
        result = splits.generate_loato_splits(df, min_samples=200)
        for fold in result["folds"]:
            held = fold["held_out_category"]
            train_cats = df.iloc[fold["train_indices"]]
            injection_train = train_cats[train_cats["label"] == 1]
            assert held not in injection_train["attack_category"].values

    def test_test_contains_held_out_and_benign(self) -> None:
        df = _make_df(categories={"cat1": 300, "cat2": 300})
        result = splits.generate_loato_splits(df, min_samples=200)
        for fold in result["folds"]:
            test_df = df.iloc[fold["test_indices"]]
            # Must have both benign and held-out injection
            assert (test_df["label"] == 0).any()
            assert (test_df["label"] == 1).any()

    def test_no_index_overlap(self) -> None:
        df = _make_df()
        result = splits.generate_loato_splits(df, min_samples=200)
        for fold in result["folds"]:
            train = set(fold["train_indices"])
            test = set(fold["test_indices"])
            assert train.isdisjoint(test)

    def test_held_out_absent_from_train(self) -> None:
        df = _make_df(categories={"cat1": 300, "cat2": 300})
        result = splits.generate_loato_splits(df, min_samples=200)
        for fold in result["folds"]:
            train_df = df.iloc[fold["train_indices"]]
            injection_train = train_df[train_df["label"] == 1]
            assert fold["held_out_category"] not in injection_train["attack_category"].values

    def test_seed_deterministic(self) -> None:
        df = _make_df()
        r1 = splits.generate_loato_splits(df, min_samples=200, seed=42)
        r2 = splits.generate_loato_splits(df, min_samples=200, seed=42)
        assert r1["folds"][0]["test_indices"] == r2["folds"][0]["test_indices"]

    def test_fold_name_pattern(self) -> None:
        df = _make_df(categories={"instruction_override": 300})
        result = splits.generate_loato_splits(df, min_samples=200)
        assert result["folds"][0]["fold_name"] == "held_out_instruction_override"

    def test_insufficient_data_raises(self) -> None:
        df = _make_df(n_benign=5, categories={"cat1": 10})
        with pytest.raises(ValueError):
            splits.generate_loato_splits(df, min_samples=200)

    def test_train_only_categories_in_train_not_held_out(self) -> None:
        """Train-only categories appear in training but never as held-out folds."""
        df = _make_df(categories={"cat1": 300, "cat2": 300, "small_cat": 50})
        result = splits.generate_loato_splits(
            df,
            min_samples=200,
            train_only_categories={"small_cat"},
        )
        # Only cat1 and cat2 should be held out
        held_out = {f["held_out_category"] for f in result["folds"]}
        assert held_out == {"cat1", "cat2"}

        # small_cat samples should be in training for every fold
        for fold in result["folds"]:
            train_df = df.iloc[fold["train_indices"]]
            train_cats = set(train_df[train_df["label"] == 1]["attack_category"])
            assert "small_cat" in train_cats

    def test_train_only_categories_absent_from_test(self) -> None:
        """Train-only categories should never appear in test injection samples."""
        df = _make_df(categories={"cat1": 300, "train_only": 100})
        result = splits.generate_loato_splits(
            df,
            min_samples=50,
            train_only_categories={"train_only"},
        )
        for fold in result["folds"]:
            test_df = df.iloc[fold["test_indices"]]
            test_injection = test_df[test_df["label"] == 1]
            assert "train_only" not in test_injection["attack_category"].values


# ===========================================================================
# TestGenerateDirectIndirectSplit
# ===========================================================================


class TestGenerateDirectIndirectSplit:
    """Tests for generate_direct_indirect_split."""

    def test_returns_dict_with_correct_structure(self) -> None:
        df = _make_df(categories={"cat1": 200}, n_indirect=50)
        result = splits.generate_direct_indirect_split(df)
        assert result["experiment"] == "direct_indirect"
        assert "train_indices" in result
        assert "test_indices" in result

    def test_train_has_direct_and_benign(self) -> None:
        df = _make_df(categories={"cat1": 200}, n_indirect=50)
        result = splits.generate_direct_indirect_split(df)
        train_df = df.iloc[result["train_indices"]]
        # Should have benign
        assert (train_df["label"] == 0).any()
        # Should have direct injection (is_indirect=False, label=1)
        direct = train_df[(train_df["label"] == 1) & (train_df["is_indirect"] == False)]  # noqa: E712
        assert len(direct) > 0

    def test_test_has_indirect_and_benign(self) -> None:
        df = _make_df(categories={"cat1": 200}, n_indirect=50)
        result = splits.generate_direct_indirect_split(df)
        test_df = df.iloc[result["test_indices"]]
        assert (test_df["label"] == 0).any()
        indirect = test_df[(test_df["label"] == 1)]
        assert (indirect["is_indirect"] == True).all()  # noqa: E712

    def test_no_index_overlap(self) -> None:
        df = _make_df(categories={"cat1": 200}, n_indirect=50)
        result = splits.generate_direct_indirect_split(df)
        assert set(result["train_indices"]).isdisjoint(set(result["test_indices"]))

    def test_no_indirect_raises(self) -> None:
        df = _make_df(categories={"cat1": 100}, n_indirect=0)
        with pytest.raises(ValueError, match="indirect"):
            splits.generate_direct_indirect_split(df)

    def test_seed_deterministic(self) -> None:
        df = _make_df(categories={"cat1": 200}, n_indirect=50)
        r1 = splits.generate_direct_indirect_split(df, seed=42)
        r2 = splits.generate_direct_indirect_split(df, seed=42)
        assert r1["train_indices"] == r2["train_indices"]


# ===========================================================================
# TestGenerateCrosslingualSplit
# ===========================================================================


class TestGenerateCrosslingualSplit:
    """Tests for generate_crosslingual_split."""

    def test_returns_dict_with_correct_structure(self) -> None:
        df = _make_df(languages={"fr": 50, "de": 30})
        result = splits.generate_crosslingual_split(df)
        assert result["experiment"] == "crosslingual"
        assert "test_languages" in result

    def test_train_is_english_only(self) -> None:
        df = _make_df(languages={"fr": 50})
        result = splits.generate_crosslingual_split(df)
        train_df = df.iloc[result["train_indices"]]
        assert (train_df["language"] == "en").all()

    def test_test_has_non_english_and_benign(self) -> None:
        df = _make_df(languages={"fr": 50})
        result = splits.generate_crosslingual_split(df)
        test_df = df.iloc[result["test_indices"]]
        assert (test_df["language"] != "en").any()
        assert (test_df["label"] == 0).any()

    def test_no_index_overlap(self) -> None:
        df = _make_df(languages={"fr": 50})
        result = splits.generate_crosslingual_split(df)
        assert set(result["train_indices"]).isdisjoint(set(result["test_indices"]))

    def test_reports_test_languages(self) -> None:
        df = _make_df(languages={"fr": 50, "de": 30})
        result = splits.generate_crosslingual_split(df)
        assert "de" in result["test_languages"]
        assert "fr" in result["test_languages"]

    def test_all_english_raises(self) -> None:
        df = _make_df()  # all English
        with pytest.raises(ValueError, match="non-English"):
            splits.generate_crosslingual_split(df)


# ===========================================================================
# TestSaveLoadSplits
# ===========================================================================


class TestSaveLoadSplits:
    """Tests for save_splits / load_splits."""

    def test_round_trip(self, tmp_path: Path) -> None:
        data = {"experiment": "test", "folds": [{"train_indices": [0, 1], "test_indices": [2]}]}
        path = tmp_path / "splits.json"
        splits.save_splits(data, path)
        loaded = splits.load_splits(path)
        assert loaded == data

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "splits.json"
        splits.save_splits({"a": 1}, path)
        assert path.exists()

    def test_load_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            splits.load_splits(tmp_path / "nope.json")

    def test_handles_large_index_arrays(self, tmp_path: Path) -> None:
        data = {"indices": list(range(100_000))}
        path = tmp_path / "big.json"
        splits.save_splits(data, path)
        loaded = splits.load_splits(path)
        assert len(loaded["indices"]) == 100_000

    def test_json_is_readable(self, tmp_path: Path) -> None:
        path = tmp_path / "readable.json"
        splits.save_splits({"key": "value"}, path)
        text = path.read_text()
        parsed = json.loads(text)
        assert parsed["key"] == "value"
        assert "\n" in text  # indent=2 produces newlines


# ===========================================================================
# TestSaveSplitParquets
# ===========================================================================


class TestSaveSplitParquets:
    """Tests for save_split_parquets and compute_file_sha256."""

    def test_creates_train_test_parquets(self, tmp_path: Path) -> None:
        """Parquet files are created for each fold."""
        df = _make_df(n_benign=20, categories={"cat1": 30})
        splits_dict = {
            "experiment": "test",
            "folds": [
                {"fold": 0, "train_indices": list(range(40)), "test_indices": list(range(40, 50))},
            ],
        }
        written = splits.save_split_parquets(df, splits_dict, tmp_path)
        assert len(written) == 2
        assert (tmp_path / "fold_0" / "train.parquet").exists()
        assert (tmp_path / "fold_0" / "test.parquet").exists()

    def test_parquet_row_counts_match(self, tmp_path: Path) -> None:
        """Parquet files have correct number of rows."""
        df = _make_df(n_benign=20, categories={"cat1": 30})
        splits_dict = {
            "experiment": "test",
            "folds": [
                {"fold": 0, "train_indices": list(range(40)), "test_indices": list(range(40, 50))},
            ],
        }
        splits.save_split_parquets(df, splits_dict, tmp_path)
        train_df = pd.read_parquet(tmp_path / "fold_0" / "train.parquet")
        test_df = pd.read_parquet(tmp_path / "fold_0" / "test.parquet")
        assert len(train_df) == 40
        assert len(test_df) == 10

    def test_custom_fold_name_fn(self, tmp_path: Path) -> None:
        """Custom fold naming function is used."""
        df = _make_df(n_benign=20, categories={"cat1": 30})
        splits_dict = {
            "experiment": "test",
            "folds": [
                {
                    "fold": 0,
                    "held_out": "cat1",
                    "train_indices": list(range(20)),
                    "test_indices": list(range(20, 50)),
                },
            ],
        }
        splits.save_split_parquets(
            df, splits_dict, tmp_path, fold_name_fn=lambda f: f"held_{f['held_out']}"
        )
        assert (tmp_path / "held_cat1" / "train.parquet").exists()

    def test_multiple_folds(self, tmp_path: Path) -> None:
        """Multiple folds produce separate directories."""
        df = _make_df(n_benign=20, categories={"cat1": 30})
        splits_dict = {
            "experiment": "test",
            "folds": [
                {"fold": 0, "train_indices": list(range(25)), "test_indices": list(range(25, 50))},
                {"fold": 1, "train_indices": list(range(25, 50)), "test_indices": list(range(25))},
            ],
        }
        written = splits.save_split_parquets(df, splits_dict, tmp_path)
        assert len(written) == 4
        assert (tmp_path / "fold_0" / "train.parquet").exists()
        assert (tmp_path / "fold_1" / "test.parquet").exists()

    def test_compute_file_sha256_deterministic(self, tmp_path: Path) -> None:
        """SHA-256 is deterministic for the same content."""
        path = tmp_path / "test.txt"
        path.write_text("hello world")
        h1 = splits.compute_file_sha256(path)
        h2 = splits.compute_file_sha256(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length


# ===========================================================================
# TestWriteSplitManifest
# ===========================================================================


class TestWriteSplitManifest:
    """Tests for write_split_manifest."""

    def test_creates_manifest_file(self, tmp_path: Path) -> None:
        """Manifest JSON is written to output_dir."""
        source = tmp_path / "source.parquet"
        source.write_bytes(b"fake parquet data")
        manifest_path = splits.write_split_manifest(
            output_dir=tmp_path,
            source_path=source,
            split_files=[],
            splits_meta=[],
        )
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["version"] == "2.0"

    def test_manifest_contains_source_hash(self, tmp_path: Path) -> None:
        """Manifest includes SHA-256 of source file."""
        source = tmp_path / "source.parquet"
        source.write_bytes(b"fake parquet data")
        manifest_path = splits.write_split_manifest(
            output_dir=tmp_path,
            source_path=source,
            split_files=[],
            splits_meta=[],
        )
        data = json.loads(manifest_path.read_text())
        assert "sha256" in data["source_data"]["source.parquet"]

    def test_manifest_includes_file_hashes(self, tmp_path: Path) -> None:
        """Manifest includes SHA-256 of each split file."""
        source = tmp_path / "source.parquet"
        source.write_bytes(b"fake parquet data")
        split_file = tmp_path / "fold_0" / "train.parquet"
        split_file.parent.mkdir()
        split_file.write_bytes(b"fake train data")

        manifest_path = splits.write_split_manifest(
            output_dir=tmp_path,
            source_path=source,
            split_files=[split_file],
            splits_meta=[{"split_type": "cv", "fold_id": 0}],
        )
        data = json.loads(manifest_path.read_text())
        assert "fold_0/train.parquet" in data["file_hashes"]

    def test_manifest_includes_seed(self, tmp_path: Path) -> None:
        """Manifest records random_state."""
        source = tmp_path / "source.parquet"
        source.write_bytes(b"fake parquet data")
        manifest_path = splits.write_split_manifest(
            output_dir=tmp_path,
            source_path=source,
            split_files=[],
            splits_meta=[],
            seed=123,
        )
        data = json.loads(manifest_path.read_text())
        assert data["random_state"] == 123


# ===========================================================================
# TestGenerateAllSplits
# ===========================================================================


class TestGenerateAllSplits:
    """Tests for generate_all_splits."""

    def test_generates_standard_cv(self, tmp_path: Path) -> None:
        df = _make_df(
            n_benign=100,
            categories={"cat1": 300, "cat2": 250},
            languages={"fr": 50},
            n_indirect=30,
        )
        result = splits.generate_all_splits(df, output_dir=tmp_path)
        assert "standard_cv" in result
        assert result["standard_cv"].exists()

    def test_generates_loato_when_viable(self, tmp_path: Path) -> None:
        df = _make_df(
            n_benign=100,
            categories={"cat1": 300, "cat2": 250},
        )
        result = splits.generate_all_splits(df, output_dir=tmp_path)
        assert "loato" in result

    def test_skips_loato_when_no_viable_categories(self, tmp_path: Path) -> None:
        df = _make_df(n_benign=100, categories={"cat1": 50})
        result = splits.generate_all_splits(df, output_dir=tmp_path)
        assert "loato" not in result

    def test_returns_path_mapping(self, tmp_path: Path) -> None:
        df = _make_df(
            n_benign=100,
            categories={"cat1": 300},
            languages={"fr": 50},
            n_indirect=30,
        )
        result = splits.generate_all_splits(df, output_dir=tmp_path)
        for name, path in result.items():
            assert isinstance(path, Path)
            assert path.exists()

    def test_uses_tmp_path(self, tmp_path: Path) -> None:
        df = _make_df(n_benign=100, categories={"cat1": 300})
        result = splits.generate_all_splits(df, output_dir=tmp_path)
        for path in result.values():
            assert str(path).startswith(str(tmp_path))
