from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Document search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results",
    )

    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional document IDs to search. "
            "When omitted, all uploaded documents are searched."
        ),
    )


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    score: float
    page_number: int | None = None
    section_type: str | None = None
    chunk_index: int


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class ClearDocumentsResponse(BaseModel):
    deleted_chunks: int
