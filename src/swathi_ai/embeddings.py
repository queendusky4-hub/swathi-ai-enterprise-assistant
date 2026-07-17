from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbeddingProvider:
    """Create lightweight, deterministic offline text embeddings."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def embed_text(self, text: str) -> np.ndarray:
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("text cannot be empty")

        matrix = self._vectorizer.transform([clean_text])

        return matrix.toarray()[0].astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)

    def embed_documents(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        clean_texts = [text.strip() for text in texts]

        if not clean_texts:
            raise ValueError("texts cannot be empty")

        if any(not text for text in clean_texts):
            raise ValueError("document text cannot be empty")

        matrix = self._vectorizer.transform(clean_texts)

        return matrix.toarray().astype(np.float32)