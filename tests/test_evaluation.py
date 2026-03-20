"""Tests for evaluation metrics and LOATO protocol."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from loato_bench.classifiers.logreg import LogRegClassifier
from loato_bench.evaluation.loato import (
    ExperimentResult,
    compute_generalization_gap,
    run_experiment,
)
from loato_bench.evaluation.metrics import (
    EvalMetrics,
    MetricResult,
    bootstrap_ci,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_predictions():
    """Perfect binary predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_prob = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        dtype=np.float64,
    )
    return y_true, y_pred, y_prob


@pytest.fixture()
def imperfect_predictions():
    """Imperfect binary predictions (1 error)."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)  # 1 FP
    y_prob = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        dtype=np.float64,
    )
    return y_true, y_pred, y_prob


@pytest.fixture()
def toy_split_dir():
    """Create a temporary directory with a fake 2-fold split JSON.

    Indices are interleaved so each fold has both classes in train and test.
    Labels: [0,1,0,1,0,1,0,1,0,1] — even=benign, odd=injection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        split = {
            "experiment": "test",
            "seed": 42,
            "folds": [
                {
                    "train_indices": [0, 1, 2, 3, 4, 5],
                    "test_indices": [6, 7, 8, 9],
                    "held_out_category": "cat_a",
                },
                {
                    "train_indices": [4, 5, 6, 7, 8, 9],
                    "test_indices": [0, 1, 2, 3],
                    "held_out_category": "cat_b",
                },
            ],
        }
        split_path = Path(tmpdir) / "test_split.json"
        with open(split_path, "w") as f:
            json.dump(split, f)
        yield split_path


@pytest.fixture()
def toy_embeddings():
    """Linearly separable embeddings for 10 samples (alternating labels)."""
    rng = np.random.RandomState(42)
    # Interleave benign and injection so any subset has both classes
    X = np.empty((10, 8), dtype=np.float32)
    y = np.empty(10, dtype=np.int64)
    for i in range(10):
        if i % 2 == 0:  # benign
            X[i] = rng.randn(8).astype(np.float32) - 3.0
            y[i] = 0
        else:  # injection
            X[i] = rng.randn(8).astype(np.float32) + 3.0
            y[i] = 1
    return X, y


# ---------------------------------------------------------------------------
# MetricResult tests
# ---------------------------------------------------------------------------


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_to_dict_without_ci(self):
        m = MetricResult(value=0.95)
        d = m.to_dict()
        assert d["value"] == 0.95
        assert d["ci_lower"] is None
        assert d["ci_upper"] is None

    def test_to_dict_with_ci(self):
        m = MetricResult(value=0.95, ci_lower=0.90, ci_upper=0.98)
        d = m.to_dict()
        assert d["ci_lower"] == 0.90
        assert d["ci_upper"] == 0.98


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_pred, y_prob)
        assert isinstance(m, EvalMetrics)
        assert m.accuracy.value == 1.0
        assert m.f1.value == 1.0
        assert m.precision.value == 1.0
        assert m.recall.value == 1.0

    def test_imperfect_predictions(self, imperfect_predictions):
        y_true, y_pred, y_prob = imperfect_predictions
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.accuracy.value < 1.0
        assert m.accuracy.value > 0.0

    def test_auc_computed_with_proba(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.auc_roc.value > 0.0
        assert m.auc_pr.value > 0.0

    def test_auc_zero_without_proba(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        m = compute_metrics(y_true, y_pred, None)
        assert m.auc_roc.value == 0.0
        assert m.auc_pr.value == 0.0

    def test_returns_eval_metrics(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_pred, y_prob)
        d = m.to_dict()
        assert set(d.keys()) == {"accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"}

    def test_summary_is_flat(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_pred, y_prob)
        s = m.summary()
        assert all(isinstance(v, float) for v in s.values())


# ---------------------------------------------------------------------------
# bootstrap_ci tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_returns_ci_bounds(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        m = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=100)
        assert m.f1.ci_lower is not None
        assert m.f1.ci_upper is not None

    def test_ci_lower_le_upper(self, imperfect_predictions):
        y_true, y_pred, y_prob = imperfect_predictions
        m = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=100)
        for name in ["accuracy", "precision", "recall", "f1"]:
            metric = getattr(m, name)
            assert metric.ci_lower is not None
            assert metric.ci_upper is not None
            assert metric.ci_lower <= metric.ci_upper

    def test_point_estimate_matches(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        point = compute_metrics(y_true, y_pred, y_prob)
        boot = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=100)
        assert boot.f1.value == point.f1.value


# ---------------------------------------------------------------------------
# run_experiment tests
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """Test the LOATO/CV experiment runner."""

    def test_returns_experiment_result(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        assert isinstance(result, ExperimentResult)

    def test_correct_number_of_folds(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        assert len(result.folds) == 2

    def test_fold_has_held_out_category(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        assert result.folds[0].held_out_category == "cat_a"
        assert result.folds[1].held_out_category == "cat_b"

    def test_mean_f1_computed(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        assert 0.0 <= result.mean_f1 <= 1.0
        assert result.std_f1 >= 0.0

    def test_metadata_correct(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        assert result.experiment == "test"
        assert result.embedding == "test_emb"
        assert result.classifier == "logreg"

    def test_to_dict_serializable(self, toy_split_dir, toy_embeddings):
        X, y = toy_embeddings
        clf = LogRegClassifier()
        result = run_experiment(
            experiment="test",
            split_path=toy_split_dir,
            embeddings=X,
            labels=y,
            classifier=clf,
            embedding_name="test_emb",
        )
        d = result.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Generalization gap tests
# ---------------------------------------------------------------------------


class TestGeneralizationGap:
    """Test ΔF1 computation."""

    def test_positive_gap(self):
        cv = ExperimentResult("cv", "emb", "clf", [], mean_f1=0.95, std_f1=0.02)
        loato = ExperimentResult("loato", "emb", "clf", [], mean_f1=0.85, std_f1=0.05)
        gap = compute_generalization_gap(cv, loato)
        assert abs(gap - 0.10) < 1e-6

    def test_zero_gap(self):
        cv = ExperimentResult("cv", "emb", "clf", [], mean_f1=0.90, std_f1=0.02)
        loato = ExperimentResult("loato", "emb", "clf", [], mean_f1=0.90, std_f1=0.02)
        gap = compute_generalization_gap(cv, loato)
        assert abs(gap) < 1e-6

    def test_negative_gap(self):
        cv = ExperimentResult("cv", "emb", "clf", [], mean_f1=0.80, std_f1=0.02)
        loato = ExperimentResult("loato", "emb", "clf", [], mean_f1=0.85, std_f1=0.02)
        gap = compute_generalization_gap(cv, loato)
        assert gap < 0.0
