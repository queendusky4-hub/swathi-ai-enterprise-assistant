from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from swathi_ai.vector_store import FaissVectorStore


class FakeEmbeddingService:
    dimension = 3

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        vectors = [
            self._create_vector(text)
            for text in texts
        ]

        return np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

    def embed_query(
        self,
        query: str,
    ) -> NDArray[np.float32]:
        return self._create_vector(query)

    @staticmethod
    def _create_vector(
        text: str,
    ) -> NDArray[np.float32]:
        normalized_text = text.lower()

        vector = np.array(
            [
                float(
                    "python" in normalized_text
                    or "programming" in normalized_text
                ),
                float(
                    "database" in normalized_text
                    or "sql" in normalized_text
                ),
                float(
                    "cloud" in normalized_text
                    or "azure" in normalized_text
                ),
            ],
            dtype=np.float32,
        )

        if not vector.any():
            vector = np.array(
                [0.1, 0.1, 0.1],
                dtype=np.float32,
            )

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )


@pytest.fixture
def vector_store() -> FaissVectorStore:
    return FaissVectorStore(
        embedding_service=FakeEmbeddingService(),
    )


def test_new_vector_store_is_empty(
    vector_store: FaissVectorStore,
) -> None:
    assert vector_store.size == 0


def test_add_vectors(
    vector_store: FaissVectorStore,
) -> None:
    added_count = vector_store.add(
        chunk_ids=[
            "chunk-python",
            "chunk-database",
        ],
        texts=[
            "Python programming language",
            "SQL database system",
        ],
    )

    assert added_count == 2
    assert vector_store.size == 2


def test_add_empty_input(
    vector_store: FaissVectorStore,
) -> None:
    added_count = vector_store.add(
        chunk_ids=[],
        texts=[],
    )

    assert added_count == 0
    assert vector_store.size == 0


def test_add_requires_matching_lengths(
    vector_store: FaissVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "chunk_ids and texts must "
            "have equal lengths"
        ),
    ):
        vector_store.add(
            chunk_ids=["chunk-one"],
            texts=[
                "first",
                "second",
            ],
        )


def test_add_rejects_empty_chunk_id(
    vector_store: FaissVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="chunk IDs cannot be empty",
    ):
        vector_store.add(
            chunk_ids=["   "],
            texts=["Python"],
        )


def test_add_rejects_duplicate_chunk_ids(
    vector_store: FaissVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate chunk ID",
    ):
        vector_store.add(
            chunk_ids=[
                "same-chunk",
                "same-chunk",
            ],
            texts=[
                "Python",
                "Database",
            ],
        )


def test_semantic_search_returns_best_match(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=[
            "python-chunk",
            "database-chunk",
            "cloud-chunk",
        ],
        texts=[
            "Python programming guide",
            "SQL database administration",
            "Azure cloud deployment",
        ],
    )

    results = vector_store.search(
        query="Python programming",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "python-chunk"
    assert results[0].score > results[1].score


def test_search_respects_top_k(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=[
            "chunk-one",
            "chunk-two",
            "chunk-three",
        ],
        texts=[
            "Python",
            "Python cloud",
            "Python database",
        ],
    )

    results = vector_store.search(
        query="Python",
        top_k=1,
    )

    assert len(results) == 1


def test_search_filters_allowed_chunk_ids(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=[
            "python-chunk",
            "database-chunk",
            "cloud-chunk",
        ],
        texts=[
            "Python programming",
            "SQL database",
            "Azure cloud",
        ],
    )

    results = vector_store.search(
        query="Python programming",
        top_k=5,
        allowed_chunk_ids={
            "database-chunk",
            "cloud-chunk",
        },
    )

    result_ids = {
        result.chunk_id
        for result in results
    }

    assert "python-chunk" not in result_ids
    assert result_ids == {
        "database-chunk",
        "cloud-chunk",
    }


def test_search_empty_query_returns_empty(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=["chunk-one"],
        texts=["Python"],
    )

    assert vector_store.search("   ") == []


def test_search_non_positive_top_k_returns_empty(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=["chunk-one"],
        texts=["Python"],
    )

    assert vector_store.search(
        query="Python",
        top_k=0,
    ) == []


def test_search_empty_store_returns_empty(
    vector_store: FaissVectorStore,
) -> None:
    assert vector_store.search(
        query="Python",
        top_k=5,
    ) == []


def test_contains_chunk(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=["chunk-one"],
        texts=["Python"],
    )

    assert vector_store.contains("chunk-one")
    assert not vector_store.contains("missing-chunk")


def test_clear_removes_all_vectors(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=[
            "chunk-one",
            "chunk-two",
        ],
        texts=[
            "Python",
            "Database",
        ],
    )

    removed_count = vector_store.clear()

    assert removed_count == 2
    assert vector_store.size == 0
    assert not vector_store.contains(
        "chunk-one"
    )


def test_rebuild_replaces_existing_vectors(
    vector_store: FaissVectorStore,
) -> None:
    vector_store.add(
        chunk_ids=["old-chunk"],
        texts=["Python"],
    )

    rebuilt_count = vector_store.rebuild(
        chunk_ids=[
            "new-database-chunk",
            "new-cloud-chunk",
        ],
        texts=[
            "SQL database",
            "Azure cloud",
        ],
    )

    assert rebuilt_count == 2
    assert vector_store.size == 2
    assert not vector_store.contains(
        "old-chunk"
    )
    assert vector_store.contains(
        "new-database-chunk"
    )
    assert vector_store.contains(
        "new-cloud-chunk"
    )