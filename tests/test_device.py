"""Tests for device selection utility."""

from unittest.mock import patch

import torch

from loato_bench.utils.device import get_device


class TestGetDeviceAuto:
    """Tests for auto device selection."""

    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=False)
    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=True)
    def test_auto_prefers_mps_when_available(self, mock_mps, mock_cuda):
        device = get_device("auto")
        assert device == torch.device("mps")

    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=False)
    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=True)
    def test_auto_falls_back_to_cuda(self, mock_cuda, mock_mps):
        device = get_device("auto")
        assert device == torch.device("cuda")

    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=False)
    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, mock_cuda, mock_mps):
        device = get_device("auto")
        assert device == torch.device("cpu")

    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=False)
    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=False)
    def test_default_is_auto(self, mock_mps, mock_cuda):
        """Calling with no args defaults to auto."""
        device = get_device()
        assert device == torch.device("cpu")


class TestGetDeviceExplicit:
    """Tests for explicit device selection."""

    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=True)
    def test_mps_when_available(self, mock_mps):
        device = get_device("mps")
        assert device == torch.device("mps")

    @patch("loato_bench.utils.device.torch.backends.mps.is_available", return_value=False)
    def test_mps_falls_back_to_cpu(self, mock_mps):
        device = get_device("mps")
        assert device == torch.device("cpu")

    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=True)
    def test_cuda_when_available(self, mock_cuda):
        device = get_device("cuda")
        assert device == torch.device("cuda")

    @patch("loato_bench.utils.device.torch.cuda.is_available", return_value=False)
    def test_cuda_falls_back_to_cpu(self, mock_cuda):
        device = get_device("cuda")
        assert device == torch.device("cpu")

    def test_cpu_always_returns_cpu(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_unknown_preferred_returns_cpu(self):
        """Unrecognized preferred string falls back to CPU."""
        device = get_device("tpu")
        assert device == torch.device("cpu")
