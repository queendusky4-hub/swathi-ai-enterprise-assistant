from __future__ import annotations

from dataclasses import dataclass

from .document_service import (
    DocumentRAGService,
    DocumentSearchResult,
)


@dataclass(frozen=True)
class RAGSource:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: int | None
    chunk_index: int


@dataclass(frozen=True)
class RAGContext:
    query: str
    context: str
    sources: list[RAGSource]

    @property
    def has_context(self) -> bool:
        return bool(self.sources and self.context.strip())


class RAGContextBuilder:
    """
    Retrieves document chunks and converts them into context
    that can be supplied to an LLM.
    """

    def __init__(
        self,
        document_service: DocumentRAGService,
        default_top_k: int = 4,
        minimum_score: float = 0.0,
        maximum_context_characters: int = 5000,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than zero"
            )

        if maximum_context_characters <= 0:
            raise ValueError(
                "maximum_context_characters must be greater than zero"
            )

        self.document_service = document_service
        self.default_top_k = default_top_k
        self.minimum_score = minimum_score
        self.maximum_context_characters = (
            maximum_context_characters
        )

    def build_context(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> RAGContext:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError("query cannot be empty")

        resolved_top_k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        if resolved_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        search_results = self.document_service.search(
        query=query,
        document_ids=[document_id],
        )

        filtered_results = [
            result
            for result in search_results
            if result.score >= self.minimum_score
        ]

        sources = [
            self._to_source(result)
            for result in filtered_results
        ]

        context = self._format_context(sources)

        return RAGContext(
            query=clean_query,
            context=context,
            sources=sources,
        )

    def build_prompt(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> tuple[str, RAGContext]:
        rag_context = self.build_context(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )

        if not rag_context.has_context:
            prompt = (
                "Answer the user's question honestly. "
                "No relevant document context was found.\n\n"
                f"Question:\n{rag_context.query}"
            )

            return prompt, rag_context

        prompt = (
            "Answer the question using only the document context "
            "provided below.\n"
            "If the answer is not contained in the context, say that "
            "the uploaded documents do not contain enough information.\n"
            "Do not invent facts.\n\n"
            f"Document context:\n{rag_context.context}\n\n"
            f"Question:\n{rag_context.query}\n\n"
            "Answer:"
        )

        return prompt, rag_context

    def _format_context(
        self,
        sources: list[RAGSource],
    ) -> str:
        context_parts: list[str] = []
        current_length = 0

        for source_number, source in enumerate(
            sources,
            start=1,
        ):
            page_label = (
                f", page {source.page_number}"
                if source.page_number is not None
                else ""
            )

            header = (
                f"[Source {source_number}: "
                f"{source.filename}{page_label}]"
            )

            context_part = (
                f"{header}\n"
                f"{source.text.strip()}"
            )

            separator_length = (
                2 if context_parts else 0
            )

            remaining_characters = (
                self.maximum_context_characters
                - current_length
                - separator_length
            )

            if remaining_characters <= 0:
                break

            if len(context_part) > remaining_characters:
                context_part = context_part[
                    :remaining_characters
                ].rstrip()

            if not context_part:
                break

            context_parts.append(context_part)

            current_length += (
                len(context_part)
                + separator_length
            )

            if (
                current_length
                >= self.maximum_context_characters
            ):
                break

        return "\n\n".join(context_parts)

    @staticmethod
    def _to_source(
        result: DocumentSearchResult,
    ) -> RAGSource:
        return RAGSource(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            text=result.text,
            score=result.score,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
        )