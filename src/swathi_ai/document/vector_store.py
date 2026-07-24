from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import faiss
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "vector_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DATA_DIR / "documents.index"
METADATA_PATH = DATA_DIR / "metadata.json"

_store_lock = Lock()


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def create(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Create a new index, replacing any existing index."""
        embeddings = self._validate_embeddings(embeddings)

        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                "Metadata count must match the number of embeddings."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = metadata

        self.save()

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Append new document chunks to the existing index."""
        embeddings = self._validate_embeddings(embeddings)

        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                "Metadata count must match the number of embeddings."
            )

        with _store_lock:
            if INDEX_PATH.exists() and METADATA_PATH.exists():
                self.load()

                if self.index is None:
                    raise RuntimeError("FAISS index could not be loaded.")

                if self.index.d != embeddings.shape[1]:
                    raise ValueError(
                        "Embedding dimension does not match the existing index."
                    )
            else:
                self.index = faiss.IndexFlatIP(
                    embeddings.shape[1]
                )
                self.metadata = []

            self.index.add(embeddings)
            self.metadata.extend(metadata)

            self.save()

    def load(self) -> None:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "No vector index exists. Upload a document first."
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            loaded_metadata = json.load(metadata_file)

        if not isinstance(loaded_metadata, list):
            raise ValueError("Stored metadata is invalid.")

        self.metadata = loaded_metadata

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata are out of sync."
            )

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("There is no FAISS index to save.")

        faiss.write_index(
            self.index,
            str(INDEX_PATH),
        )

        with METADATA_PATH.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                self.metadata,
                metadata_file,
                ensure_ascii=False,
                indent=2,
            )

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            self.load()

        if self.index is None or self.index.ntotal == 0:
            return []

        embedding = self._validate_embeddings(embedding)

        candidate_count = min(
            max(top_k * 10, top_k),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            embedding,
            candidate_count,
        )

        results: list[dict[str, Any]] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            item = dict(self.metadata[index_position])

            if (
                document_ids
                and item.get("document_id") not in document_ids
            ):
                continue

            item["score"] = float(score)
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def list_documents(self) -> list[dict[str, Any]]:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return []

        self.load()

        documents: dict[str, dict[str, Any]] = {}

        for item in self.metadata:
            document_id = str(
                item.get("document_id", "")
            )

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": item.get(
                        "filename",
                        "Unknown document",
                    ),
                    "chunks": 0,
                    "pages": set(),
                }

            documents[document_id]["chunks"] += 1
            documents[document_id]["pages"].add(
                int(item.get("page", 1))
            )

        result: list[dict[str, Any]] = []

        for document in documents.values():
            result.append(
                {
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "chunks": document["chunks"],
                    "pages": len(document["pages"]),
                }
            )

        return result

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(
            embeddings,
            dtype="float32",
        )

        if array.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if array.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty.")

        return array