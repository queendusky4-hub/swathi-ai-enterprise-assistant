from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

from .documents import extract_document_text


@dataclass
class DocumentUploadResult:
    document_id: str
    filename: str
    chunk_count: int


@dataclass
class DocumentSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: int | None
    section_type: str | None
    chunk_index: int


@dataclass
class StoredChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int | None
    section_type: str | None
    chunk_index: int


class DocumentRAGService:
    """
    Lightweight in-memory document indexing and search service.

    Supports PDF, DOCX, and TXT files through documents.py.
    Uploaded documents remain indexed while the FastAPI server is running.
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[StoredChunk] = []

    def index_document(
        self,
        filename: str,
        file_bytes: bytes,
    ) -> DocumentUploadResult:
        text = extract_document_text(
            filename=filename,
            file_bytes=file_bytes,
        )

        document_id = str(uuid4())

        chunks = self._split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            page_number = self._extract_page_number(chunk_text)
            section_type = self._detect_section_type(chunk_text)

            self._chunks.append(
                StoredChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    filename=filename,
                    text=chunk_text,
                    page_number=page_number,
                    section_type=section_type,
                    chunk_index=chunk_index,
                )
            )

        return DocumentUploadResult(
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[DocumentSearchResult]:
        clean_query = query.strip()

        if not clean_query or not self._chunks:
            return []

        query_tokens = self._tokenize(clean_query)
        query_vector = Counter(query_tokens)

        scored_results: list[DocumentSearchResult] = []

        for chunk in self._chunks:
            chunk_tokens = self._tokenize(chunk.text)
            chunk_vector = Counter(chunk_tokens)

            score = self._cosine_similarity(
                query_vector,
                chunk_vector,
            )

            if score <= 0:
                score = 0.000001

            scored_results.append(
                DocumentSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    score=round(score, 6),
                    page_number=chunk.page_number,
                    section_type=chunk.section_type,
                    chunk_index=chunk.chunk_index,
                )
            )

        scored_results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return scored_results[:top_k]

    def _split_text(self, text: str) -> list[str]:
        clean_text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        ).strip()

        if not clean_text:
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in clean_text.split("\n\n")
            if paragraph.strip()
        ]

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            candidate = (
                f"{current_chunk}\n\n{paragraph}".strip()
                if current_chunk
                else paragraph
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk)

            if len(paragraph) <= self.chunk_size:
                current_chunk = paragraph
                continue

            long_paragraph_chunks = self._split_long_text(
                paragraph
            )

            if long_paragraph_chunks:
                chunks.extend(long_paragraph_chunks[:-1])
                current_chunk = long_paragraph_chunks[-1]
            else:
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        chunks: list[str] = []

        step = max(
            1,
            self.chunk_size - self.chunk_overlap,
        )

        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step

        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"[\w\u0B80-\u0BFF]+",
            text.lower(),
            flags=re.UNICODE,
        )

    @staticmethod
    def _cosine_similarity(
        first: Counter[str],
        second: Counter[str],
    ) -> float:
        if not first or not second:
            return 0.0

        common_tokens = set(first) & set(second)

        dot_product = sum(
            first[token] * second[token]
            for token in common_tokens
        )

        first_length = math.sqrt(
            sum(value * value for value in first.values())
        )

        second_length = math.sqrt(
            sum(value * value for value in second.values())
        )

        denominator = first_length * second_length

        if denominator == 0:
            return 0.0

        return dot_product / denominator

    @staticmethod
    def _extract_page_number(
        text: str,
    ) -> int | None:
        match = re.search(
            r"\[Page\s+(\d+)\]",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def _detect_section_type(
        text: str,
    ) -> str | None:
        stripped_text = text.strip()

        if re.match(
            r"^\[Page\s+\d+\]",
            stripped_text,
            flags=re.IGNORECASE,
        ):
            return "page"

        if re.match(
            r"^\[Table\s+\d+\]",
            stripped_text,
            flags=re.IGNORECASE,
        ):
            return "table"

        first_line = stripped_text.splitlines()[0]

        if (
            len(first_line) <= 100
            and not first_line.endswith((".", "?", "!"))
        ):
            return "section"

        return "text"

