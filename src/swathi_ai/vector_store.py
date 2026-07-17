from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class VectorRecord:
    """A single document chunk stored with its vector and metadata."""

    record_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """A semantic-search result."""

    record_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """
    Lightweight persistent vector store using NumPy cosine similarity.

    This implementation is suitable for local development, testing,
    demonstrations and small-to-medium document collections.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self._records: list[VectorRecord] = []
        self._vectors = np.empty(
            shape=(0, dimension),
            dtype=np.float32,
        )

    @property
    def size(self) -> int:
        return len(self._records)

    def add(
        self,
        texts: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        record_ids: list[str] | None = None,
    ) -> list[str]:
        if not texts:
            raise ValueError("texts cannot be empty")

        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")

        if vectors.shape[0] != len(texts):
            raise ValueError(
                "the number of vectors must match the number of texts"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors with dimension {self.dimension}"
            )

        if any(not text.strip() for text in texts):
            raise ValueError("text values cannot be empty")

        if metadata is None:
            metadata = [{} for _ in texts]

        if len(metadata) != len(texts):
            raise ValueError(
                "the number of metadata entries must match the texts"
            )

        if record_ids is None:
            record_ids = [
                str(uuid4())
                for _ in texts
            ]

        if len(record_ids) != len(texts):
            raise ValueError(
                "the number of record IDs must match the texts"
            )

        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record IDs must be unique")

        existing_ids = {
            record.record_id
            for record in self._records
        }

        if any(record_id in existing_ids for record_id in record_ids):
            raise ValueError("a record ID already exists")

        normalised_vectors = self._normalise(vectors)

        new_records = [
            VectorRecord(
                record_id=record_id,
                text=text.strip(),
                metadata=dict(item_metadata),
            )
            for record_id, text, item_metadata in zip(
                record_ids,
                texts,
                metadata,
                strict=True,
            )
        ]

        self._records.extend(new_records)
        self._vectors = np.vstack(
            [
                self._vectors,
                normalised_vectors,
            ]
        ).astype(np.float32)

        return record_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.size == 0:
            return []

        query = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        if query.shape[0] != self.dimension:
            raise ValueError(
                f"expected query dimension {self.dimension}"
            )

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("query vector cannot be all zeros")

        normalised_query = query / query_norm
        scores = self._vectors @ normalised_query

        candidate_indexes = [
            index
            for index, record in enumerate(self._records)
            if self._matches_filter(
                record.metadata,
                metadata_filter,
            )
        ]

        ranked_indexes = sorted(
            candidate_indexes,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                record_id=self._records[index].record_id,
                text=self._records[index].text,
                score=float(scores[index]),
                metadata=dict(self._records[index].metadata),
            )
            for index in ranked_indexes
        ]

    def delete(self, record_ids: list[str]) -> int:
        if not record_ids:
            return 0

        ids_to_delete = set(record_ids)

        kept_indexes = [
            index
            for index, record in enumerate(self._records)
            if record.record_id not in ids_to_delete
        ]

        deleted_count = self.size - len(kept_indexes)

        self._records = [
            self._records[index]
            for index in kept_indexes
        ]

        if kept_indexes:
            self._vectors = self._vectors[kept_indexes]
        else:
            self._vectors = np.empty(
                shape=(0, self.dimension),
                dtype=np.float32,
            )

        return deleted_count

    def delete_by_metadata(
        self,
        metadata_filter: dict[str, Any],
    ) -> int:
        matching_ids = [
            record.record_id
            for record in self._records
            if self._matches_filter(
                record.metadata,
                metadata_filter,
            )
        ]

        return self.delete(matching_ids)

    def list_records(self) -> list[VectorRecord]:
        return list(self._records)

    def save(self, directory: str | Path) -> None:
        output_directory = Path(directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = output_directory / "records.json"
        vectors_path = output_directory / "vectors.npy"

        payload = {
            "dimension": self.dimension,
            "records": [
                asdict(record)
                for record in self._records
            ],
        }

        metadata_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        np.save(
            vectors_path,
            self._vectors,
            allow_pickle=False,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> VectorStore:
        input_directory = Path(directory)
        metadata_path = input_directory / "records.json"
        vectors_path = input_directory / "vectors.npy"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"vector-store metadata not found: {metadata_path}"
            )

        if not vectors_path.exists():
            raise FileNotFoundError(
                f"vector data not found: {vectors_path}"
            )

        payload = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        dimension = int(payload["dimension"])
        store = cls(dimension=dimension)

        store._records = [
            VectorRecord(
                record_id=item["record_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in payload["records"]
        ]

        vectors = np.load(
            vectors_path,
            allow_pickle=False,
        ).astype(np.float32)

        expected_shape = (
            len(store._records),
            dimension,
        )

        if vectors.shape != expected_shape:
            raise ValueError(
                "stored vectors do not match the saved records"
            )

        store._vectors = vectors
        return store

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        float_vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            float_vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError("vectors cannot contain all-zero rows")

        return float_vectors / norms

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        if not metadata_filter:
            return True

        return all(
            metadata.get(key) == value
            for key, value in metadata_filter.items()
        )