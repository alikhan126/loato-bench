"""E5-Mistral-7B GGUF embedding model via llama-cpp-python."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from loato_bench.embeddings.base import EmbeddingModel
from loato_bench.utils.config import EmbeddingConfig

if TYPE_CHECKING:
    from llama_cpp import Llama


class E5MistralEmbedding(EmbeddingModel):
    """Embedding model using E5-Mistral-7B via llama-cpp-python GGUF.

    Downloads the GGUF file from HuggingFace Hub on first use.
    Uses Metal acceleration on Apple Silicon when available.

    Requires ``llama-cpp-python`` and ``huggingface-hub`` to be installed.
    Install with: ``CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python``
    """

    _model: Llama

    def __init__(self, config: EmbeddingConfig) -> None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        self._config = config
        assert config.gguf_repo is not None, "gguf_repo is required for E5-Mistral"
        assert config.gguf_file is not None, "gguf_file is required for E5-Mistral"
        model_path = hf_hub_download(
            config.gguf_repo,
            filename=config.gguf_file,
        )
        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_gpu_layers=-1,
            verbose=False,
        )

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dim(self) -> int:
        return self._config.dim

    def encode(self, texts: list[str], batch_size: int = 1) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        template = self._config.prompt_template or "{text}"
        embeddings = []
        for text in texts:
            formatted = template.format(text=text)
            result = self._model.embed(formatted)
            # llama-cpp returns list of lists; take the first (and only) embedding
            embeddings.append(result[0])

        return np.array(embeddings, dtype=np.float32)
