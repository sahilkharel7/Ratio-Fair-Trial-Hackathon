"""Local MiniLM embedder (loaded from models/, never downloaded) and the fake used in unit tests."""

import numpy as np
import pytest

from ratio.embeddings import EmbeddingModelMissing, MiniLMEmbedder, cosine_matrix
from ratio.paths import MINILM_DIR
from ratio.testing import FakeEmbedder

needs_model = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")


def test_missing_model_directory_gives_setup_instructions(tmp_path):
    with pytest.raises(EmbeddingModelMissing, match="fetch_models.py"):
        MiniLMEmbedder(tmp_path / "absent")


def test_fake_embedder_is_deterministic_and_normalised():
    vectors = FakeEmbedder().encode(["The defendant was present.", "", "The defendant was present."])
    assert vectors.shape == (3, 512)
    assert np.allclose(vectors[0], vectors[2])
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert np.allclose(vectors[1], 0.0)


def test_cosine_matrix_of_normalised_vectors():
    vectors = FakeEmbedder().encode(["interpreter present", "interpreter present", "sentence reduced"])
    similarity = cosine_matrix(vectors, vectors)
    assert similarity.shape == (3, 3)
    assert similarity[0, 1] > 0.99
    assert similarity[0, 2] < 0.5


@pytest.fixture(scope="module")
def embedder():
    return MiniLMEmbedder()


@pytest.mark.embed
@needs_model
class TestMiniLM:
    def test_shape_dtype_and_norm(self, embedder):
        vectors = embedder.encode(["An interpreter was present.", "The hearing was adjourned."])
        assert vectors.shape == (2, 384)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)

    def test_paraphrase_is_closer_than_unrelated_text(self, embedder):
        a, b, c = embedder.encode(
            [
                "Messages recovered from the accused's mobile telephone show that he coordinated the timing.",
                "Messages retrieved from the telephone of the accused demonstrate that he arranged the timing.",
                "The court adjourned the hearing until the following month.",
            ]
        )
        assert float(a @ b) > float(a @ c) + 0.3

    def test_empty_input_returns_empty_matrix(self, embedder):
        assert embedder.encode([]).shape == (0, 384)
