from __future__ import annotations

import numpy as np
import pytest

from swathi_ai.embeddings import EmbeddingService


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del batch_size
        del show_progress_bar
        del convert_to_numpy

        vectors: list[list[float]] = []

        for text in texts:
            normalized_text = text.lower()

            vector = np.array(
                [
                    float("python" in normalized_text),
                    float("database" in normalized_text),
                    float("cloud" in normalized_text),
                ],
                dtype=np.float32,
            )

            if not vector.any():
                vector = np.array(
                    [0.1, 0.1, 0.1],
                    dtype=np.float32,
                )

            if normalize_embeddings:
                norm = np.linalg.norm(vector)

                if norm > 0:
                    vector = vector / norm

            vectors.append(vector.tolist())

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


@pytest.fixture
def embedding_service(
    monkeypatch: pytest.MonkeyPatch,
) -> EmbeddingService:
    monkeypatch.setattr(
        EmbeddingService,
        "_load_model",
        staticmethod(
            lambda model_name, device: (
                FakeSentenceTransformer()
            )
        ),
    )

    return EmbeddingService(
        model_name="fake-model",
        batch_size=4,
    )


def test_embedding_dimension(
    embedding_service: EmbeddingService,
) -> None:
    assert embedding_service.dimension == 3


def test_embed_documents_returns_float32_matrix(
    embedding_service: EmbeddingService,
) -> None:
    embeddings = embedding_service.embed_documents(
        [
            "Python application",
            "Cloud database",
        ]
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert embeddings.flags.c_contiguous


def test_embed_documents_returns_normalized_vectors(
    embedding_service: EmbeddingService,
) -> None:
    embeddings = embedding_service.embed_documents(
        [
            "Python cloud application",
            "Database system",
        ]
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        np.ones(2),
        atol=1e-6,
    )


def test_embed_documents_ignores_blank_text(
    embedding_service: EmbeddingService,
) -> None:
    embeddings = embedding_service.embed_documents(
        [
            "Python",
            "",
            "   ",
            "Database",
        ]
    )

    assert embeddings.shape == (2, 3)


def test_embed_documents_empty_input(
    embedding_service: EmbeddingService,
) -> None:
    embeddings = embedding_service.embed_documents([])

    assert embeddings.shape == (0, 3)
    assert embeddings.dtype == np.float32


def test_embed_documents_rejects_non_string(
    embedding_service: EmbeddingService,
) -> None:
    with pytest.raises(
        TypeError,
        match="Every document must be a string",
    ):
        embedding_service.embed_documents(
            [
                "valid text",
                123,  # type: ignore[list-item]
            ]
        )


def test_embed_query_returns_one_vector(
    embedding_service: EmbeddingService,
) -> None:
    embedding = embedding_service.embed_query(
        "Python cloud"
    )

    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
    assert np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-6,
    )


def test_embed_query_rejects_empty_query(
    embedding_service: EmbeddingService,
) -> None:
    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        embedding_service.embed_query("   ")


def test_empty_model_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EmbeddingService,
        "_load_model",
        staticmethod(
            lambda model_name, device: (
                FakeSentenceTransformer()
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        EmbeddingService(model_name="   ")


def test_invalid_batch_size_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EmbeddingService,
        "_load_model",
        staticmethod(
            lambda model_name, device: (
                FakeSentenceTransformer()
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        EmbeddingService(
            model_name="fake-model",
            batch_size=0,
        )


def test_invalid_embedding_shape_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Expected a two-dimensional "
            "embedding matrix"
        ),
    ):
        EmbeddingService._to_float32_matrix(
            np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            )
        )


def test_non_finite_embedding_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Embedding output contains "
            "invalid values"
        ),
    ):
        EmbeddingService._to_float32_matrix(
            np.array(
                [
                    [1.0, np.nan, 2.0],
                ],
                dtype=np.float32,
            )
        )