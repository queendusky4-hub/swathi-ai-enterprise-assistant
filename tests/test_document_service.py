from swathi_ai.document_service import (
    DocumentRAGService,
    StoredChunk,
)


def make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    chunk_index: int = 0,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.txt",
        text=text,
        page_number=None,
        section_type=None,
        chunk_index=chunk_index,
    )


def test_search_filters_by_document_id() -> None:
    service = DocumentRAGService()

    service._chunks = [
        make_chunk(
            chunk_id="chunk-a",
            document_id="document-a",
            text="Python is commonly used for machine learning.",
        ),
        make_chunk(
            chunk_id="chunk-b",
            document_id="document-b",
            text="Python supports automated data processing.",
        ),
    ]

    results = service.search(
        query="Python",
        top_k=5,
        document_ids=["document-b"],
    )

    assert results
    assert all(
        result.document_id == "document-b"
        for result in results
    )


def test_exact_phrase_match_is_ranked_first() -> None:
    service = DocumentRAGService()

    service._chunks = [
        make_chunk(
            chunk_id="reordered-words",
            document_id="document-a",
            text="Learning platform machine",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="exact-phrase",
            document_id="document-a",
            text="Machine learning platform",
            chunk_index=1,
        ),
    ]

    results = service.search(
        query="machine learning",
        top_k=2,
    )

    assert results
    assert results[0].chunk_id == "exact-phrase"
    assert results[0].score > results[1].score


def test_search_excludes_completely_irrelevant_chunks() -> None:
    service = DocumentRAGService()

    service._chunks = [
        make_chunk(
            chunk_id="cooking",
            document_id="document-a",
            text="This recipe explains how to prepare vegetable soup.",
        ),
        make_chunk(
            chunk_id="gardening",
            document_id="document-b",
            text="Water the garden plants early in the morning.",
        ),
    ]

    results = service.search(
        query="quantum computing",
        top_k=5,
    )

    assert results == []
