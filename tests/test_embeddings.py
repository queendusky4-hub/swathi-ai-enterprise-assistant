import numpy as np
import pytest

from swathi_ai.embeddings import HashingEmbeddingProvider


def test_embed_text_returns_expected_shape() -> None:
    provider = HashingEmbeddingProvider(dimension=128)

    embedding = provider.embed_text("Hello Swathi AI")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32


def test_embed_documents_returns_matrix() -> None:
    provider = HashingEmbeddingProvider(dimension=64)

    embeddings = provider.embed_documents(
        [
            "Tamil language assistant",
            "Enterprise artificial intelligence platform",
        ]
    )

    assert embeddings.shape == (2, 64)
    assert embeddings.dtype == np.float32


def test_same_text_produces_same_embedding() -> None:
    provider = HashingEmbeddingProvider()

    first = provider.embed_text("multilingual chatbot")
    second = provider.embed_text("multilingual chatbot")

    assert np.allclose(first, second)


def test_empty_text_raises_error() -> None:
    provider = HashingEmbeddingProvider()

    with pytest.raises(ValueError, match="text cannot be empty"):
        provider.embed_text("   ")


def test_invalid_dimension_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        HashingEmbeddingProvider(dimension=0)