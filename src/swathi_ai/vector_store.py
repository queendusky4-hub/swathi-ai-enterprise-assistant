from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from .embeddings import EmbeddingService


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    score: float
    position: int


class FaissVectorStore:
    """
    Thread-safe in-memory FAISS vector store.

    Embeddings are normalized before indexing, so IndexFlatIP returns
    cosine-similarity scores.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ) -> None:
        self.embedding_service = embedding_service
        self.dimension = embedding_service.dimension

        self._index = faiss.IndexFlatIP(
            self.dimension
        )

        self._chunk_ids: list[str] = []
        self._lock = RLock()

    @property
    def size(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def add(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
    ) -> int:
        if len(chunk_ids) != len(texts):
            raise ValueError(
                "chunk_ids and texts must have equal lengths."
            )

        if not chunk_ids:
            return 0

        validated_chunk_ids = (
            self._validate_chunk_ids(chunk_ids)
        )

        embeddings = (
            self.embedding_service.embed_documents(texts)
        )

        if embeddings.shape[0] != len(
            validated_chunk_ids
        ):
            raise RuntimeError(
                "The embedding count does not match "
                "the chunk ID count."
            )

        self._validate_embedding_matrix(embeddings)

        with self._lock:
            self._index.add(embeddings)
            self._chunk_ids.extend(
                validated_chunk_ids
            )

        return len(validated_chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        clean_query = query.strip()

        if not clean_query:
            return []

        if top_k <= 0:
            return []

        with self._lock:
            total_vectors = int(self._index.ntotal)

            if total_vectors == 0:
                return []

            search_limit = total_vectors

            query_embedding = (
                self.embedding_service.embed_query(
                    clean_query
                )
            )

            query_matrix = np.ascontiguousarray(
                query_embedding.reshape(1, -1),
                dtype=np.float32,
            )

            scores, positions = self._index.search(
                query_matrix,
                search_limit,
            )

            results: list[VectorSearchResult] = []

            for score, position in zip(
                scores[0],
                positions[0],
                strict=True,
            ):
                position_value = int(position)

                if position_value < 0:
                    continue

                chunk_id = self._chunk_ids[
                    position_value
                ]

                if (
                    allowed_chunk_ids is not None
                    and chunk_id
                    not in allowed_chunk_ids
                ):
                    continue

                results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        score=round(
                            float(score),
                            6,
                        ),
                        position=position_value,
                    )
                )

                if len(results) >= top_k:
                    break

            return results

    def clear(self) -> int:
        with self._lock:
            removed_count = int(
                self._index.ntotal
            )

            self._index.reset()
            self._chunk_ids.clear()

            return removed_count

    def rebuild(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
    ) -> int:
        with self._lock:
            self._index.reset()
            self._chunk_ids.clear()

        return self.add(
            chunk_ids=chunk_ids,
            texts=texts,
        )

    def contains(
        self,
        chunk_id: str,
    ) -> bool:
        with self._lock:
            return chunk_id in self._chunk_ids

    @staticmethod
    def _validate_chunk_ids(
        chunk_ids: Sequence[str],
    ) -> list[str]:
        validated: list[str] = []
        seen: set[str] = set()

        for chunk_id in chunk_ids:
            clean_chunk_id = chunk_id.strip()

            if not clean_chunk_id:
                raise ValueError(
                    "chunk IDs cannot be empty."
                )

            if clean_chunk_id in seen:
                raise ValueError(
                    f"Duplicate chunk ID: {clean_chunk_id}"
                )

            seen.add(clean_chunk_id)
            validated.append(clean_chunk_id)

        return validated

    def _validate_embedding_matrix(
        self,
        embeddings: NDArray[np.float32],
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be two-dimensional."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match "
                "the FAISS index dimension."
            )

        if embeddings.dtype != np.float32:
            raise ValueError(
                "FAISS embeddings must use float32."
            )

        if not embeddings.flags.c_contiguous:
            raise ValueError(
                "FAISS embeddings must be contiguous."
            )