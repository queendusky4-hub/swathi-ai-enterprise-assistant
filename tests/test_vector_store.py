import numpy as np
import pytest

from swathi_ai.vector_store import VectorStore


def test_add_and_search_vectors() -> None:
    store = VectorStore(dimension=3)

    store.add(
        texts=[
            "Tamil chatbot",
            "Cloud deployment",
        ],
        vectors=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        metadata=[
            {"document_id": "doc-1"},
            {"document_id": "doc-2"},
        ],
        record_ids=[
            "record-1",
            "record-2",
        ],
    )

    results = store.search(
        query_vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].record_id == "record-1"
    assert results[0].text == "Tamil chatbot"
    assert results[0].score == pytest.approx(1.0)


def test_search_with_metadata_filter() -> None:
    store = VectorStore(dimension=2)

    store.add(
        texts=[
            "First document",
            "Second document",
        ],
        vectors=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        metadata=[
            {"document_id": "doc-1"},
            {"document_id": "doc-2"},
        ],
    )

    results = store.search(
        query_vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        metadata_filter={
            "document_id": "doc-2",
        },
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-2"


def test_delete_record() -> None:
    store = VectorStore(dimension=2)

    store.add(
        texts=["First", "Second"],
        vectors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        record_ids=["first", "second"],
    )

    deleted = store.delete(["first"])

    assert deleted == 1
    assert store.size == 1
    assert store.list_records()[0].record_id == "second"


def test_delete_by_metadata() -> None:
    store = VectorStore(dimension=2)

    store.add(
        texts=["First", "Second"],
        vectors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        metadata=[
            {"document_id": "doc-1"},
            {"document_id": "doc-2"},
        ],
    )

    deleted = store.delete_by_metadata(
        {"document_id": "doc-1"}
    )

    assert deleted == 1
    assert store.size == 1
    assert (
        store.list_records()[0].metadata["document_id"]
        == "doc-2"
    )


def test_save_and_load_store(tmp_path) -> None:
    store = VectorStore(dimension=3)

    store.add(
        texts=["Enterprise AI assistant"],
        vectors=np.array(
            [[0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        metadata=[
            {
                "document_id": "doc-1",
                "page": 2,
            }
        ],
        record_ids=["record-1"],
    )

    store.save(tmp_path)

    loaded_store = VectorStore.load(tmp_path)

    assert loaded_store.size == 1
    assert loaded_store.dimension == 3

    results = loaded_store.search(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )
    )

    assert results[0].record_id == "record-1"
    assert results[0].metadata["page"] == 2


def test_incorrect_vector_dimension_raises_error() -> None:
    store = VectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="expected vectors with dimension 3",
    ):
        store.add(
            texts=["Invalid"],
            vectors=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
        )


def test_empty_store_search_returns_empty_list() -> None:
    store = VectorStore(dimension=3)

    results = store.search(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert results == []