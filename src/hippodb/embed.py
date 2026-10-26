"""Text embeddings, run locally so iterating costs nothing.

Uses fastembed (ONNX runtime) rather than sentence-transformers: same
bge-small-en-v1.5 model, but no PyTorch install, so it sets up quickly on
Windows/WSL, Mac and Linux alike.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised row per text, shape (len(texts), dim)."""
        ...


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return normalise(vectors)


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
