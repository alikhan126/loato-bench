"""Embedding cache manager (NumPy .npz format)."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def compute_text_hash(texts: list[str]) -> str:
    """Compute a deterministic SHA-256 hash of a list of texts.

    Sorts texts before hashing so the result is order-independent.
    """
    combined = "\n".join(sorted(texts))
    return hashlib.sha256(combined.encode()).hexdigest()


class EmbeddingCache:
    """Manages cached embeddings as .npz files with .meta.json sidecars.

    Storage layout::

        {base_dir}/{model_name}/
            embeddings.npz   — contains 'embeddings' (float32 N×D) and 'sample_ids' (str)
            meta.json         — model_name, model_version, text_hash, n_samples, dim, created_at
    """

    def __init__(self, model_name: str, base_dir: Path | None = None) -> None:
        from loato_bench.utils.config import DATA_DIR

        if base_dir is None:
            base_dir = DATA_DIR / "embeddings"
        self.model_name = model_name
        self.cache_dir = base_dir / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _npz_path(self) -> Path:
        return self.cache_dir / "embeddings.npz"

    @property
    def _meta_path(self) -> Path:
        return self.cache_dir / "meta.json"

    def save(
        self,
        embeddings: NDArray[np.float32],
        sample_ids: list[str],
        model_version: str,
        text_hash: str,
    ) -> None:
        """Save embeddings and metadata to cache."""
        np.savez(self._npz_path, embeddings=embeddings, sample_ids=np.array(sample_ids))
        meta = {
            "model_name": self.model_name,
            "model_version": model_version,
            "text_hash": text_hash,
            "n_samples": embeddings.shape[0],
            "dim": embeddings.shape[1],
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def load(self) -> tuple[NDArray[np.float32], list[str]] | None:
        """Load cached embeddings. Returns None if no cache exists."""
        if not self._npz_path.exists():
            return None
        data = np.load(self._npz_path, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        sample_ids = data["sample_ids"].tolist()
        return embeddings, sample_ids

    def is_valid(self, model_version: str, text_hash: str) -> bool:
        """Check if the cache matches the expected model version and text hash."""
        if not self._meta_path.exists():
            return False
        meta = json.loads(self._meta_path.read_text())
        return bool(meta["model_version"] == model_version and meta["text_hash"] == text_hash)

    def clear(self) -> None:
        """Remove cached files."""
        if self._npz_path.exists():
            self._npz_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
