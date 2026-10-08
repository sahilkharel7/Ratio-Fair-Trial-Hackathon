"""Sentence embeddings from the local MiniLM model in models/ (CPU only, never downloaded here)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ratio.paths import MINILM_DIR

EMBEDDING_DIM = 384


class EmbeddingModelMissing(RuntimeError):
    """The embedding model has not been downloaded yet."""


class MiniLMEmbedder:
    def __init__(self, model_dir: Path = MINILM_DIR) -> None:
        model_dir = Path(model_dir).resolve()
        if not (model_dir / "modules.json").exists():
            raise EmbeddingModelMissing(
                f"Embedding model not found at {model_dir}. "
                "Run `python scripts/fetch_models.py` once while online."
            )
        from sentence_transformers import SentenceTransformer  # heavy import, deferred until needed

        # CPU keeps similarity scores reproducible across machines. An absolute path plus
        # local_files_only means sentence-transformers never looks the model up on the Hub.
        self._model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of two L2-normalised matrices."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)
