from __future__ import annotations

import numpy as np

from swathi_ai.document_service import DocumentRAGService
from swathi_ai.rag import RAGContextBuilder
from swathi_ai.vector_store import FaissVectorStore


class FakeEmbeddingService:
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts):
        vectors = []

        for text in texts:
            vector = np.zeros(self._dimension, dtype=np.float32)

            for index, character in enumerate(text.lower()):
                vector[index % self._dimension] += ord(character) % 31

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            vectors.append(vector)

        if not vectors:
            return np.empty(
                (0, self._dimension),
                dtype=np.float32,
            )

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query):
        vector = self.embed_documents([query])[0]
        return vector


def create_rag_builder(
    minimum_score: float = 0.0,
    maximum_context_characters: int = 5000,
) -> RAGContextBuilder:
    embedding_service = FakeEmbeddingService(
        dimension=8
    )

    vector_store = FaissVectorStore(
        embedding_service=embedding_service
    )

    document_service = DocumentRAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return RAGContextBuilder(
        document_service=document_service,
        default_top_k=3,
        minimum_score=minimum_score,
        maximum_context_characters=(
            maximum_context_characters
        ),
    )


def test_build_context_from_document() -> None:
    builder = create_rag_builder()

    upload_result = (
        builder.document_service.index_document(
            filename="ai.txt",
            file_bytes=(
                b"Artificial intelligence allows machines "
                b"to perform tasks that normally require "
                b"human intelligence."
            ),
        )
    )

    result = builder.build_context(
        query="What is artificial intelligence?",
        document_id=upload_result.document_id,
    )

    assert result.has_context is True
    assert result.sources
    assert "Artificial intelligence" in result.context
    assert (
        result.sources[0].document_id
        == upload_result.document_id
    )


def test_build_context_with_document_filter() -> None:
    builder = create_rag_builder()

    tamil_upload = (
        builder.document_service.index_document(
            filename="tamil.txt",
            file_bytes=(
                b"Swathi AI supports Tamil language users."
            ),
        )
    )

    builder.document_service.index_document(
        filename="cloud.txt",
        file_bytes=(
            b"Cloud infrastructure provides scalable servers."
        ),
    )

    result = builder.build_context(
        query="What language users does Swathi AI support?",
        document_id=tamil_upload.document_id,
    )

    assert result.has_context is True
    assert result.sources

    assert all(
        source.document_id == tamil_upload.document_id
        for source in result.sources
    )


def test_build_context_rejects_empty_query() -> None:
    builder = create_rag_builder()

    try:
        builder.build_context(query="   ")
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_build_prompt_without_context() -> None:
    builder = create_rag_builder(
        minimum_score=2.0
    )

    prompt, context = builder.build_prompt(
        query="What is quantum computing?"
    )

    assert context.has_context is False
    assert "No relevant document context was found" in prompt