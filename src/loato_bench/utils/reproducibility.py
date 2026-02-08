"""Seed everything for reproducibility."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
