from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question or text to search for.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching chunks to return.",
    )

    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document IDs to search.",
    )


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer from the uploaded documents.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to use.",
    )

    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document IDs to use.",
    )