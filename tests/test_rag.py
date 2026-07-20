from swathi_ai.document_service import (
    DocumentRAGService,
)
from swathi_ai.embeddings import (
    HashingEmbeddingProvider,
)
from swathi_ai.rag import RAGContextBuilder
from swathi_ai.vector_store import VectorStore


def create_rag_builder(
    minimum_score: float = 0.0,
    maximum_context_characters: int = 5000,
) -> RAGContextBuilder:
    provider = HashingEmbeddingProvider(
        dimension=128
    )

    store = VectorStore(dimension=128)

    document_service = DocumentRAGService(
        embedding_provider=provider,
        vector_store=store,
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

    builder.document_service.index_document(
        filename="ai.txt",
        file_bytes=(
            b"Artificial intelligence allows machines "
            b"to perform tasks that normally require "
            b"human intelligence."
        ),
        document_id="ai-document",
    )

    result = builder.build_context(
        query="What is artificial intelligence?"
    )

    assert result.has_context is True
    assert result.sources
    assert "Artificial intelligence" in result.context
    assert (
        result.sources[0].document_id
        == "ai-document"
    )


def test_build_context_with_document_filter() -> None:
    builder = create_rag_builder()

    builder.document_service.index_document(
        filename="tamil.txt",
        file_bytes=(
            b"Swathi AI supports Tamil language users."
        ),
        document_id="tamil-document",
    )

    builder.document_service.index_document(
        filename="cloud.txt",
        file_bytes=(
            b"Cloud infrastructure provides scalable servers."
        ),
        document_id="cloud-document",
    )

    result = builder.build_context(
        query="Tamil language support",
        document_id="tamil-document",
    )

    assert result.sources
    assert all(
        source.document_id == "tamil-document"
        for source in result.sources
    )


def test_build_prompt_contains_question_and_context() -> None:
    builder = create_rag_builder()

    builder.document_service.index_document(
        filename="rag.txt",
        file_bytes=(
            b"Retrieval augmented generation combines "
            b"document retrieval with language generation."
        ),
        document_id="rag-document",
    )

    prompt, context = builder.build_prompt(
        query="What does RAG combine?"
    )

    assert context.has_context is True
    assert "Retrieval augmented generation" in prompt
    assert "What does RAG combine?" in prompt
    assert "Do not invent facts" in prompt


def test_empty_store_creates_no_context_prompt() -> None:
    builder = create_rag_builder()

    prompt, context = builder.build_prompt(
        query="What is in my documents?"
    )

    assert context.has_context is False
    assert context.sources == []
    assert "No relevant document context was found" in prompt


def test_context_respects_character_limit() -> None:
    builder = create_rag_builder(
        maximum_context_characters=100
    )

    builder.document_service.index_document(
        filename="long.txt",
        file_bytes=(
            b"Artificial intelligence and machine learning "
            b"are important technologies. "
            b"This document contains additional information "
            b"about enterprise systems and automation."
        ),
        document_id="long-document",
    )

    result = builder.build_context(
        query="artificial intelligence"
    )

    assert len(result.context) <= 100


def test_empty_query_raises_error() -> None:
    builder = create_rag_builder()

    try:
        builder.build_context("   ")
    except ValueError as error:
        assert "query cannot be empty" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for empty query"
        )


def test_invalid_top_k_raises_error() -> None:
    builder = create_rag_builder()

    try:
        builder.build_context(
            query="test",
            top_k=0,
        )
    except ValueError as error:
        assert (
            "top_k must be greater than zero"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected ValueError for invalid top_k"
        )