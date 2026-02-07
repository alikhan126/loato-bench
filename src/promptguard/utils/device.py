"""Device selection for PyTorch (CPU / MPS / CUDA)."""

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Select the best available device for computation.

    Args:
        preferred: One of 'auto', 'cpu', 'mps', 'cuda'.
            'auto' will pick MPS on Apple Silicon, CUDA if available, else CPU.

    Returns:
        A torch.device instance.
    """
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
