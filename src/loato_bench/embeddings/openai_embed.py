"""OpenAI text-embedding-3-small model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from loato_bench.embeddings.base import EmbeddingModel
from loato_bench.utils.config import EmbeddingConfig


class OpenAIEmbedding(EmbeddingModel):
    """Embedding model using the OpenAI embeddings API.

    Uses tenacity for automatic retries on rate-limit errors.
    Reads OPENAI_API_KEY from the environment.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._client = openai.OpenAI()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dim(self) -> int:
        return self._config.dim

    def encode(self, texts: list[str], batch_size: int = 256) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._call_api(batch)
            # Sort by index to guarantee order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(item.embedding for item in sorted_data)

        return np.array(all_embeddings, dtype=np.float32)

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((openai.RateLimitError, openai.InternalServerError)),
    )
    def _call_api(self, texts: list[str]) -> openai.types.CreateEmbeddingResponse:
        """Call the OpenAI embeddings API with retry logic."""
        assert self._config.model_id is not None, "model_id is required for OpenAI embeddings"
        return self._client.embeddings.create(
            model=self._config.model_id,
            input=texts,
        )
