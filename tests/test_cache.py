"""Tests for embedding cache manager — Sprint 1B."""

import json

import numpy as np

from promptguard.embeddings.cache import EmbeddingCache, compute_text_hash


class TestComputeTextHash:
    """Tests for the text hashing utility."""

    def test_deterministic(self):
        """Same input always produces the same hash."""
        texts = ["hello", "world"]
        assert compute_text_hash(texts) == compute_text_hash(texts)

    def test_order_independent(self):
        """Hash is the same regardless of input order."""
        assert compute_text_hash(["a", "b", "c"]) == compute_text_hash(["c", "a", "b"])

    def test_different_texts_produce_different_hash(self):
        """Different inputs produce different hashes."""
        assert compute_text_hash(["hello"]) != compute_text_hash(["world"])

    def test_empty_list(self):
        """Empty list produces a valid hash."""
        h = compute_text_hash([])
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length


class TestEmbeddingCacheInit:
    """Tests for cache initialization."""

    def test_creates_directory(self, tmp_path):
        """Cache directory is created on init."""
        cache = EmbeddingCache("test_model", base_dir=tmp_path / "embeddings")
        assert cache.cache_dir.exists()

    def test_default_model_name(self, tmp_path):
        """Cache stores the model name."""
        cache = EmbeddingCache("minilm", base_dir=tmp_path)
        assert cache.model_name == "minilm"


class TestEmbeddingCacheSaveLoad:
    """Tests for save/load round-trips."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save then load returns identical data."""
        cache = EmbeddingCache("test", base_dir=tmp_path)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        ids = ["s0", "s1", "s2", "s3", "s4"]

        cache.save(embeddings, ids, model_version="v1", text_hash="abc123")
        result = cache.load()

        assert result is not None
        loaded_emb, loaded_ids = result
        np.testing.assert_array_equal(loaded_emb, embeddings)
        assert loaded_ids == ids

    def test_load_returns_none_when_empty(self, tmp_path):
        """Load returns None if nothing was saved."""
        cache = EmbeddingCache("empty", base_dir=tmp_path)
        assert cache.load() is None

    def test_large_array_roundtrip(self, tmp_path):
        """Round-trip works for large arrays (1000x384)."""
        cache = EmbeddingCache("large", base_dir=tmp_path)
        embeddings = np.random.randn(1000, 384).astype(np.float32)
        ids = [f"sample_{i}" for i in range(1000)]

        cache.save(embeddings, ids, model_version="v1", text_hash="hash")
        loaded_emb, loaded_ids = cache.load()

        assert loaded_emb.shape == (1000, 384)
        assert loaded_emb.dtype == np.float32
        np.testing.assert_array_equal(loaded_emb, embeddings)
        assert loaded_ids == ids

    def test_save_creates_meta_json(self, tmp_path):
        """Save creates a .meta.json sidecar file."""
        cache = EmbeddingCache("meta_test", base_dir=tmp_path)
        embeddings = np.random.randn(3, 8).astype(np.float32)

        cache.save(embeddings, ["a", "b", "c"], model_version="v2", text_hash="xyz")

        meta_path = cache.cache_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["model_name"] == "meta_test"
        assert meta["model_version"] == "v2"
        assert meta["text_hash"] == "xyz"
        assert meta["n_samples"] == 3
        assert meta["dim"] == 8
        assert "created_at" in meta


class TestEmbeddingCacheIsValid:
    """Tests for cache validation."""

    def test_valid_when_hash_and_version_match(self, tmp_path):
        """Returns True when both hash and version match."""
        cache = EmbeddingCache("valid", base_dir=tmp_path)
        embeddings = np.random.randn(2, 4).astype(np.float32)
        cache.save(embeddings, ["a", "b"], model_version="v1", text_hash="hash1")

        assert cache.is_valid(model_version="v1", text_hash="hash1") is True

    def test_invalid_when_text_hash_changes(self, tmp_path):
        """Returns False when text hash doesn't match."""
        cache = EmbeddingCache("hash_change", base_dir=tmp_path)
        embeddings = np.random.randn(2, 4).astype(np.float32)
        cache.save(embeddings, ["a", "b"], model_version="v1", text_hash="hash1")

        assert cache.is_valid(model_version="v1", text_hash="different") is False

    def test_invalid_when_model_version_changes(self, tmp_path):
        """Returns False when model version doesn't match."""
        cache = EmbeddingCache("ver_change", base_dir=tmp_path)
        embeddings = np.random.randn(2, 4).astype(np.float32)
        cache.save(embeddings, ["a", "b"], model_version="v1", text_hash="hash1")

        assert cache.is_valid(model_version="v2", text_hash="hash1") is False

    def test_invalid_when_no_cache_exists(self, tmp_path):
        """Returns False when no cache files exist."""
        cache = EmbeddingCache("no_cache", base_dir=tmp_path)
        assert cache.is_valid(model_version="v1", text_hash="h") is False


class TestEmbeddingCacheClear:
    """Tests for cache clearing."""

    def test_clear_removes_files(self, tmp_path):
        """Clear removes npz and meta files."""
        cache = EmbeddingCache("clear_test", base_dir=tmp_path)
        embeddings = np.random.randn(2, 4).astype(np.float32)
        cache.save(embeddings, ["a", "b"], model_version="v1", text_hash="h")

        assert (cache.cache_dir / "embeddings.npz").exists()
        assert (cache.cache_dir / "meta.json").exists()

        cache.clear()

        assert not (cache.cache_dir / "embeddings.npz").exists()
        assert not (cache.cache_dir / "meta.json").exists()

    def test_clear_on_empty_cache_is_safe(self, tmp_path):
        """Clear doesn't raise when no files exist."""
        cache = EmbeddingCache("empty_clear", base_dir=tmp_path)
        cache.clear()  # Should not raise
