from swathi_ai.document_service import DocumentRAGService
from swathi_ai.embeddings import HashingEmbeddingProvider
from swathi_ai.vector_store import VectorStore


def create_service(
    storage_directory=None,
) -> DocumentRAGService:
    provider = HashingEmbeddingProvider(
        dimension=128
    )

    store = VectorStore(dimension=128)

    return DocumentRAGService(
        embedding_provider=provider,
        vector_store=store,
        storage_directory=storage_directory,
    )


def test_index_txt_document() -> None:
    service = create_service()

    result = service.index_document(
        filename="ai-notes.txt",
        file_bytes=(
            b"Artificial intelligence enables machines "
            b"to perform tasks requiring human intelligence."
        ),
        document_id="doc-1",
    )

    assert result.document_id == "doc-1"
    assert result.filename == "ai-notes.txt"
    assert result.chunk_count == 1


def test_search_indexed_document() -> None:
    service = create_service()

    service.index_document(
        filename="machine-learning.txt",
        file_bytes=(
            b"Machine learning allows computer systems "
            b"to learn patterns from data."
        ),
        document_id="ml-document",
    )

    results = service.search(
        query="learning patterns from data",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].document_id == "ml-document"
    assert (
        results[0].filename
        == "machine-learning.txt"
    )


def test_search_with_document_filter() -> None:
    service = create_service()

    service.index_document(
        filename="tamil.txt",
        file_bytes=(
            b"Swathi AI is a multilingual Tamil assistant."
        ),
        document_id="doc-tamil",
    )

    service.index_document(
        filename="cloud.txt",
        file_bytes=(
            b"Cloud systems provide scalable infrastructure."
        ),
        document_id="doc-cloud",
    )

    results = service.search(
        query="Tamil multilingual assistant",
        top_k=5,
        document_id="doc-tamil",
    )

    assert results
    assert all(
        result.document_id == "doc-tamil"
        for result in results
    )


def test_list_documents() -> None:
    service = create_service()

    service.index_document(
        filename="first.txt",
        file_bytes=b"First enterprise document.",
        document_id="first",
    )

    service.index_document(
        filename="second.txt",
        file_bytes=b"Second enterprise document.",
        document_id="second",
    )

    documents = service.list_documents()

    assert len(documents) == 2

    document_ids = {
        document.document_id
        for document in documents
    }

    assert document_ids == {
        "first",
        "second",
    }


def test_delete_document() -> None:
    service = create_service()

    service.index_document(
        filename="delete-me.txt",
        file_bytes=b"This document will be deleted.",
        document_id="delete-me",
    )

    deleted_count = service.delete_document(
        "delete-me"
    )

    assert deleted_count == 1
    assert service.list_documents() == []


def test_duplicate_document_id_raises_error() -> None:
    service = create_service()

    service.index_document(
        filename="first.txt",
        file_bytes=b"First document.",
        document_id="duplicate-id",
    )

    try:
        service.index_document(
            filename="second.txt",
            file_bytes=b"Second document.",
            document_id="duplicate-id",
        )
    except ValueError as error:
        assert "document already exists" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for duplicate document ID"
        )


def test_service_persistence(tmp_path) -> None:
    service = create_service(
        storage_directory=tmp_path
    )

    service.index_document(
        filename="persistent.txt",
        file_bytes=(
            b"Persistent vector stores survive restarts."
        ),
        document_id="persistent-document",
    )

    loaded_service = DocumentRAGService.load(
        storage_directory=tmp_path,
        embedding_provider=(
            HashingEmbeddingProvider(
                dimension=128
            )
        ),
    )

    documents = loaded_service.list_documents()

    assert len(documents) == 1
    assert (
        documents[0].document_id
        == "persistent-document"
    )

    results = loaded_service.search(
        "vector stores survive restarts",
        top_k=1,
    )

    assert results
    assert (
        results[0].document_id
        == "persistent-document"
    )