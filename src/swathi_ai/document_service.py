
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .chunking import DocumentSection, TextChunker
from .documents import extract_document_text
from .embeddings import HashingEmbeddingProvider
from .vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    filename: str
    chunk_count: int


@dataclass(frozen=True)
class DocumentSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: int | None
    section_type: str
    chunk_index: int


class DocumentRAGService:
    def __init__(
        self,
        embedding_provider: HashingEmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        chunker: TextChunker | None = None,
        storage_directory: str | Path | None = None,
    ) -> None:
        self.embedding_provider = (
            embedding_provider
            or HashingEmbeddingProvider(dimension=384)
        )

        self.vector_store = (
            vector_store
            or VectorStore(
                dimension=self.embedding_provider.dimension
            )
        )

        if (
            self.vector_store.dimension
            != self.embedding_provider.dimension
        ):
            raise ValueError(
                "embedding dimension must match vector-store dimension"
            )

        self.chunker = chunker or TextChunker()

        self.storage_directory = (
            Path(storage_directory)
            if storage_directory is not None
            else None
        )

    def index_document(
        self,
        filename: str,
        file_bytes: bytes,
        document_id: str | None = None,
    ) -> IndexedDocument:
        clean_filename = Path(filename).name.strip()

        if not clean_filename:
            raise ValueError("filename cannot be empty")

        resolved_document_id = (
            document_id.strip()
            if document_id
            else str(uuid4())
        )

        if not resolved_document_id:
            raise ValueError("document_id cannot be empty")

        if self._document_exists(resolved_document_id):
            raise ValueError(
                f"document already exists: {resolved_document_id}"
            )

        extracted_text = extract_document_text(
            filename=clean_filename,
            file_bytes=file_bytes,
        )

        sections = [
            DocumentSection(
                text=extracted_text,
                page_number=None,
                section_type="text",
            )
        ]

        chunks = self.chunker.chunk_sections(
            document_id=resolved_document_id,
            filename=clean_filename,
            sections=sections,
        )

        if not chunks:
            raise ValueError(
                "the document did not produce any readable chunks"
            )

        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_provider.embed_documents(
            texts
        )

        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "section_type": chunk.section_type,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        record_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        self.vector_store.add(
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            record_ids=record_ids,
        )

        if self.storage_directory is not None:
            self.vector_store.save(
                self.storage_directory
            )

        return IndexedDocument(
            document_id=resolved_document_id,
            filename=clean_filename,
            chunk_count=len(chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[DocumentSearchResult]:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError("query cannot be empty")

        query_vector = (
            self.embedding_provider.embed_query(
                clean_query
            )
        )

        metadata_filter: dict[str, Any] | None = None

        if document_id:
            metadata_filter = {
                "document_id": document_id
            }

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        return [
            self._convert_result(result)
            for result in results
        ]

    def list_documents(
        self,
    ) -> list[IndexedDocument]:
        documents: dict[str, dict[str, Any]] = {}

        for record in self.vector_store.list_records():
            document_id = str(
                record.metadata.get(
                    "document_id",
                    "",
                )
            )

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "filename": str(
                        record.metadata.get(
                            "filename",
                            "",
                        )
                    ),
                    "chunk_count": 0,
                }

            documents[document_id][
                "chunk_count"
            ] += 1

        return [
            IndexedDocument(
                document_id=document_id,
                filename=details["filename"],
                chunk_count=details["chunk_count"],
            )
            for document_id, details
            in documents.items()
        ]

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        clean_document_id = document_id.strip()

        if not clean_document_id:
            raise ValueError(
                "document_id cannot be empty"
            )

        deleted_count = (
            self.vector_store.delete_by_metadata(
                {
                    "document_id":
                    clean_document_id
                }
            )
        )

        if (
            deleted_count
            and self.storage_directory is not None
        ):
            self.vector_store.save(
                self.storage_directory
            )

        return deleted_count

    @classmethod
    def load(
        cls,
        storage_directory: str | Path,
        embedding_provider:
            HashingEmbeddingProvider | None = None,
        chunker: TextChunker | None = None,
    ) -> "DocumentRAGService":
        directory = Path(storage_directory)

        loaded_store = VectorStore.load(
            directory
        )

        provider = (
            embedding_provider
            or HashingEmbeddingProvider(
                dimension=loaded_store.dimension
            )
        )

        return cls(
            embedding_provider=provider,
            vector_store=loaded_store,
            chunker=chunker,
            storage_directory=directory,
        )

    def _document_exists(
        self,
        document_id: str,
    ) -> bool:
        return any(
            record.metadata.get(
                "document_id"
            ) == document_id
            for record
            in self.vector_store.list_records()
        )

    @staticmethod
    def _convert_result(
        result: SearchResult,
    ) -> DocumentSearchResult:
        metadata = result.metadata

        return DocumentSearchResult(
            chunk_id=result.record_id,
            document_id=str(
                metadata.get(
                    "document_id",
                    "",
                )
            ),
            filename=str(
                metadata.get(
                    "filename",
                    "",
                )
            ),
            text=result.text,
            score=result.score,
            page_number=metadata.get(
                "page_number"
            ),
            section_type=str(
                metadata.get(
                    "section_type",
                    "text",
                )
            ),
            chunk_index=int(
                metadata.get(
                    "chunk_index",
                    0,
                )
            ),
        )
