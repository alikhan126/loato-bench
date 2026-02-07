"""Tests for reproducibility seed utility."""

import random
from unittest.mock import patch

import numpy as np
import torch

from promptguard.utils.reproducibility import seed_everything


class TestSeedEverything:
    """Tests for the seed_everything function."""

    def test_python_random_is_seeded(self):
        seed_everything(42)
        a = random.random()
        seed_everything(42)
        b = random.random()
        assert a == b

    def test_numpy_is_seeded(self):
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_is_seeded(self):
        seed_everything(42)
        a = torch.rand(5)
        seed_everything(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_results(self):
        seed_everything(42)
        a = random.random()
        seed_everything(99)
        b = random.random()
        assert a != b

    def test_default_seed_is_42(self):
        seed_everything()
        a = random.random()
        seed_everything(42)
        b = random.random()
        assert a == b

    @patch("promptguard.utils.reproducibility.torch.manual_seed")
    @patch("promptguard.utils.reproducibility.torch.backends.mps.is_available", return_value=False)
    @patch("promptguard.utils.reproducibility.torch.cuda.is_available", return_value=True)
    @patch("promptguard.utils.reproducibility.torch.cuda.manual_seed_all")
    def test_cuda_seeded_when_available(self, mock_seed_all, mock_cuda, mock_mps, mock_manual):
        seed_everything(42)
        mock_seed_all.assert_called_once_with(42)

    @patch("promptguard.utils.reproducibility.torch.manual_seed")
    @patch("promptguard.utils.reproducibility.torch.cuda.is_available", return_value=False)
    @patch("promptguard.utils.reproducibility.torch.backends.mps.is_available", return_value=True)
    @patch("promptguard.utils.reproducibility.torch.mps.manual_seed")
    def test_mps_seeded_when_available(self, mock_mps_seed, mock_mps, mock_cuda, mock_manual):
        seed_everything(42)
        mock_mps_seed.assert_called_once_with(42)

    @patch("promptguard.utils.reproducibility.torch.manual_seed")
    @patch("promptguard.utils.reproducibility.torch.backends.mps.is_available", return_value=False)
    @patch("promptguard.utils.reproducibility.torch.cuda.is_available", return_value=False)
    @patch("promptguard.utils.reproducibility.torch.cuda.manual_seed_all")
    def test_cuda_not_seeded_when_unavailable(
        self, mock_seed_all, mock_cuda, mock_mps, mock_manual
    ):
        seed_everything(42)
        mock_seed_all.assert_not_called()
