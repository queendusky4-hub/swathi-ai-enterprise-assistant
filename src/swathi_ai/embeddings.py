from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingService:
    """
    Generates normalized sentence embeddings for documents and queries.

    Normalized embeddings allow FAISS inner-product search to behave like
    cosine-similarity search.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self._model = self._load_model(
            model_name=model_name,
            device=device,
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(
        model_name: str,
        device: str | None,
    ) -> SentenceTransformer:
        model_arguments: dict[str, str] = {}

        if device is not None:
            model_arguments["device"] = device

        return SentenceTransformer(
            model_name,
            **model_arguments,
        )

    @property
    def dimension(self) -> int:
        dimension = (
            self._model.get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "The embedding model did not provide "
                "an embedding dimension."
            )

        return int(dimension)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        clean_texts = self._validate_texts(texts)

        if not clean_texts:
            return np.empty(
                shape=(0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self._model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return self._to_float32_matrix(embeddings)

    def embed_query(
        self,
        query: str,
    ) -> NDArray[np.float32]:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "query cannot be empty."
            )

        embeddings = self._model.encode(
            [clean_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        matrix = self._to_float32_matrix(embeddings)

        return matrix[0]

    @staticmethod
    def _validate_texts(
        texts: Sequence[str],
    ) -> list[str]:
        clean_texts: list[str] = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    "Every document must be a string."
                )

            clean_text = text.strip()

            if clean_text:
                clean_texts.append(clean_text)

        return clean_texts

    @staticmethod
    def _to_float32_matrix(
        embeddings: object,
    ) -> NDArray[np.float32]:
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise RuntimeError(
                "Expected a two-dimensional embedding matrix."
            )

        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(
                "Embedding output contains invalid values."
            )

        return np.ascontiguousarray(matrix)