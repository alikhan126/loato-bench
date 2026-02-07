"""Embedding model implementations and caching."""

from collections.abc import Callable

from promptguard.embeddings.base import EmbeddingModel
from promptguard.embeddings.cache import EmbeddingCache, compute_text_hash
from promptguard.embeddings.instructor import InstructorEmbeddingModel
from promptguard.embeddings.openai_embed import OpenAIEmbedding
from promptguard.embeddings.sentence_tf import SentenceTransformerEmbedding
from promptguard.utils.config import EmbeddingConfig, load_embedding_config

# E5MistralEmbedding is imported lazily since it requires llama-cpp-python

_MODEL_CLASS_MAP: dict[str, Callable[[EmbeddingConfig], EmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerEmbedding,
    "InstructorEmbedding": InstructorEmbeddingModel,
    "openai": OpenAIEmbedding,
}


def get_embedding_model(name: str) -> EmbeddingModel:
    """Factory: load config YAML and return the right embedding model instance.

    Args:
        name: Model key (e.g., 'minilm', 'bge_large', 'instructor', 'openai_small', 'e5_mistral')
    """
    config = load_embedding_config(name)

    if config.library == "llama-cpp-python":
        from promptguard.embeddings.e5_mistral import E5MistralEmbedding

        return E5MistralEmbedding(config)

    cls = _MODEL_CLASS_MAP.get(config.library)
    if cls is None:
        msg = f"Unknown embedding library: {config.library}"
        raise ValueError(msg)
    return cls(config)


__all__ = [
    "EmbeddingCache",
    "EmbeddingModel",
    "InstructorEmbeddingModel",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "compute_text_hash",
    "get_embedding_model",
]
