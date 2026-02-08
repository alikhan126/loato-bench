"""Tests for W&B tracking utilities — Sprint 1B."""

from unittest.mock import MagicMock, patch

import numpy as np

from loato_bench.tracking.wandb_utils import (
    finish_run,
    init_run,
    log_confusion_matrix,
    log_metrics,
)


class TestInitRun:
    """Tests for init_run."""

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_calls_wandb_init_with_correct_name(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock()
        init_run("loato", "minilm", "xgboost", 3)
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "loato_minilm_xgboost_3"

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_group_follows_convention(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock()
        init_run("standard_cv", "bge_large", "svm", 0)
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["group"] == "standard_cv_bge_large_svm"

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_project_is_loato_bench(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock()
        init_run("loato", "minilm", "logreg", 1)
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "loato-bench"

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_tags_include_experiment_embedding_classifier(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock()
        init_run("loato", "minilm", "mlp", 2)
        call_kwargs = mock_wandb.init.call_args[1]
        assert set(call_kwargs["tags"]) == {"loato", "minilm", "mlp"}

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_passes_config_dict(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock()
        cfg = {"lr": 0.01, "epochs": 10}
        init_run("loato", "minilm", "logreg", 0, config=cfg)
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["config"] == cfg

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_returns_wandb_run(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        result = init_run("loato", "minilm", "logreg", 0)
        assert result is mock_run


class TestLogMetrics:
    """Tests for log_metrics."""

    def test_logs_metrics_without_prefix(self):
        mock_run = MagicMock()
        metrics = {"f1": 0.92, "accuracy": 0.95}
        log_metrics(mock_run, metrics)
        mock_run.log.assert_called_once_with({"f1": 0.92, "accuracy": 0.95})

    def test_logs_metrics_with_prefix(self):
        mock_run = MagicMock()
        metrics = {"f1": 0.88}
        log_metrics(mock_run, metrics, prefix="val")
        mock_run.log.assert_called_once_with({"val/f1": 0.88})


class TestLogConfusionMatrix:
    """Tests for log_confusion_matrix."""

    @patch("loato_bench.tracking.wandb_utils.wandb")
    def test_logs_confusion_matrix(self, mock_wandb):
        mock_run = MagicMock()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        log_confusion_matrix(mock_run, y_true, y_pred, ["benign", "injection"])
        mock_wandb.plot.confusion_matrix.assert_called_once()
        mock_run.log.assert_called_once()


class TestFinishRun:
    """Tests for finish_run."""

    def test_calls_run_finish(self):
        mock_run = MagicMock()
        finish_run(mock_run)
        mock_run.finish.assert_called_once()
