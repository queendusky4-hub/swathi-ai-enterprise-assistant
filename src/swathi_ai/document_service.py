from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

from rank_bm25 import BM25Okapi


from .documents import extract_document_text
from .embeddings import EmbeddingService
from .mlflow_tracker import MLflowTracker
from .vector_store import FaissVectorStore
from .monitoring import (
    RAG_RESULTS_TOTAL,
    RAG_SEARCH_DURATION_SECONDS,
    RAG_SEARCHES_TOTAL,
)

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
    """In-memory document indexing and hybrid retrieval service."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
        tracker: MLflowTracker | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if (
            vector_store is not None
            and embedding_service is not None
            and vector_store.embedding_service is not embedding_service
        ):
            raise ValueError(
                "vector_store and embedding_service must use "
                "the same EmbeddingService instance."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[StoredChunk] = []
        self.tracker = tracker

        if vector_store is not None:
            self.embedding_service = vector_store.embedding_service
            self.vector_store = vector_store
        else:
            self.embedding_service = embedding_service or EmbeddingService()
            self.vector_store = FaissVectorStore(
                embedding_service=self.embedding_service
            )

    def index_document(
        self,
        filename: str,
        file_bytes: bytes,
    ) -> DocumentUploadResult:
        clean_filename = filename.strip()
        if not clean_filename:
            raise ValueError("filename cannot be empty.")
        if not isinstance(file_bytes, bytes):
            raise TypeError("file_bytes must be bytes.")
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty.")

        text = extract_document_text(
            filename=clean_filename,
            file_bytes=file_bytes,
        )
        document_id = str(uuid4())
        chunk_texts = self._split_text(text)
        stored_chunks: list[StoredChunk] = []

        for chunk_index, chunk_text in enumerate(chunk_texts):
            stored_chunks.append(
                StoredChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    filename=clean_filename,
                    text=chunk_text,
                    page_number=self._extract_page_number(chunk_text),
                    section_type=self._detect_section_type(chunk_text),
                    chunk_index=chunk_index,
                )
            )

        if stored_chunks:
            self.vector_store.add(
                chunk_ids=[chunk.chunk_id for chunk in stored_chunks],
                texts=[chunk.text for chunk in stored_chunks],
            )
            self._chunks.extend(stored_chunks)

        return DocumentUploadResult(
            document_id=document_id,
            filename=clean_filename,
            chunk_count=len(stored_chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[DocumentSearchResult]:
        start_time = time.perf_counter()
        RAG_SEARCHES_TOTAL.inc()

        try:
            if self.tracker is None:
                results = self._search_internal(
                    query=query,
                    top_k=top_k,
                    document_ids=document_ids,
                )
            else:
                embedding_model = getattr(
                    self.embedding_service,
                    "model_name",
                    self.embedding_service.__class__.__name__,
                )

                with self.tracker.measure_retrieval(
                    query=query,
                    retrieval_method="hybrid-bm25-faiss",
                    top_k=top_k,
                    embedding_model=str(embedding_model),
                ) as metrics:
                    results = self._search_internal(
                        query=query,
                        top_k=top_k,
                        document_ids=document_ids,
                    )

                    metrics["retrieved_chunks"] = len(results)
                    metrics["average_score"] = (
                        sum(result.score for result in results) / len(results)
                        if results
                        else 0.0
                    )

            RAG_RESULTS_TOTAL.inc(len(results))
            return results

        finally:
            duration = time.perf_counter() - start_time
            RAG_SEARCH_DURATION_SECONDS.observe(duration)

    def _search_internal(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[DocumentSearchResult]:
        clean_query = query.strip()
        if not clean_query or top_k <= 0 or not self._chunks:
            return []

        query_tokens = self._tokenize(clean_query)
        if not query_tokens:
            return []

        allowed_document_ids = set(document_ids) if document_ids is not None else None
        candidate_chunks = [
            chunk
            for chunk in self._chunks
            if allowed_document_ids is None
            or chunk.document_id in allowed_document_ids
        ]
        if not candidate_chunks:
            return []

        candidate_chunk_ids = {chunk.chunk_id for chunk in candidate_chunks}
        semantic_results = self.vector_store.search(
            query=clean_query,
            top_k=len(candidate_chunks),
            allowed_chunk_ids=candidate_chunk_ids,
        )
        semantic_scores = {
            result.chunk_id: max(result.score, 0.0)
            for result in semantic_results
        }
        normalized_semantic_scores = self._normalize_score_map(semantic_scores)

        tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in candidate_chunks
        ]
        raw_bm25_scores = self._calculate_bm25_scores(
            query_tokens=query_tokens,
            tokenized_chunks=tokenized_chunks,
        )
        normalized_bm25_scores = self._normalize_scores(raw_bm25_scores)

        query_vector = Counter(query_tokens)
        unique_query_tokens = set(query_tokens)
        normalized_query = " ".join(query_tokens)
        scored_results: list[DocumentSearchResult] = []

        for chunk, chunk_tokens, bm25_score in zip(
            candidate_chunks,
            tokenized_chunks,
            normalized_bm25_scores,
            strict=True,
        ):
            if not chunk_tokens:
                continue

            unique_chunk_tokens = set(chunk_tokens)
            matched_query_tokens = unique_query_tokens & unique_chunk_tokens
            cosine_score = self._cosine_similarity(
                query_vector,
                Counter(chunk_tokens),
            )
            keyword_coverage = (
                len(matched_query_tokens) / len(unique_query_tokens)
            )
            normalized_chunk = " ".join(chunk_tokens)
            phrase_score = 1.0 if normalized_query in normalized_chunk else 0.0
            semantic_score = normalized_semantic_scores.get(chunk.chunk_id, 0.0)

            final_score = (
                0.45 * semantic_score
                + 0.25 * bm25_score
                + 0.15 * cosine_score
                + 0.10 * phrase_score
                + 0.05 * keyword_coverage
            )
            if final_score <= 0:
                continue

            scored_results.append(
                DocumentSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    score=round(final_score, 6),
                    page_number=chunk.page_number,
                    section_type=chunk.section_type,
                    chunk_index=chunk.chunk_index,
                )
            )

        scored_results.sort(
            key=lambda item: (item.score, -item.chunk_index),
            reverse=True,
        )
        return scored_results[:top_k]

    def delete_document(self, document_id: str) -> int:
        clean_document_id = document_id.strip()
        if not clean_document_id:
            return 0

        original_count = len(self._chunks)
        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.document_id != clean_document_id
        ]
        removed_count = original_count - len(self._chunks)
        if removed_count:
            self._rebuild_vector_store()
        return removed_count

    def clear(self) -> int:
        removed_count = len(self._chunks)
        self._chunks.clear()
        self.vector_store.clear()
        return removed_count

    def get_document_chunks(self, document_id: str) -> list[StoredChunk]:
        clean_document_id = document_id.strip()
        if not clean_document_id:
            return []
        return [
            chunk
            for chunk in self._chunks
            if chunk.document_id == clean_document_id
        ]

    def _rebuild_vector_store(self) -> None:
        self.vector_store.rebuild(
            chunk_ids=[chunk.chunk_id for chunk in self._chunks],
            texts=[chunk.text for chunk in self._chunks],
        )

    def _calculate_bm25_scores(
        self,
        query_tokens: list[str],
        tokenized_chunks: list[list[str]],
    ) -> list[float]:
        if not query_tokens or not tokenized_chunks:
            return []

        safe_corpus = [
            chunk_tokens if chunk_tokens else [""]
            for chunk_tokens in tokenized_chunks
        ]
        bm25 = BM25Okapi(safe_corpus)
        raw_scores = bm25.get_scores(query_tokens)
        return [round(float(score), 6) for score in raw_scores]

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        non_negative_scores = [max(float(score), 0.0) for score in scores]
        highest_score = max(non_negative_scores, default=0.0)
        if highest_score <= 0:
            return [0.0 for _ in non_negative_scores]
        return [score / highest_score for score in non_negative_scores]

    @staticmethod
    def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        highest_score = max(
            (max(float(score), 0.0) for score in scores.values()),
            default=0.0,
        )
        if highest_score <= 0:
            return {chunk_id: 0.0 for chunk_id in scores}
        return {
            chunk_id: max(float(score), 0.0) / highest_score
            for chunk_id, score in scores.items()
        }

    def _split_text(self, text: str) -> list[str]:
        clean_text = re.sub(r"\n{3,}", "\n\n", text).strip()
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

            long_paragraph_chunks = self._split_long_text(paragraph)
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
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
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
    def _extract_page_number(text: str) -> int | None:
        match = re.search(
            r"\[Page\s+(\d+)\]",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _detect_section_type(text: str) -> str | None:
        normalized_text = text.strip().lower()
        if not normalized_text:
            return None

        section_patterns = {
            "abstract": ("abstract", "executive summary"),
            "introduction": ("introduction", "background"),
            "literature_review": (
                "literature review",
                "related work",
                "previous research",
            ),
            "methodology": (
                "methodology",
                "methods",
                "research method",
                "materials and methods",
            ),
            "results": ("results", "findings", "evaluation results"),
            "discussion": ("discussion", "analysis"),
            "conclusion": (
                "conclusion",
                "conclusions",
                "summary and conclusion",
            ),
            "references": ("references", "bibliography"),
            "appendix": ("appendix", "appendices"),
        }

        opening_text = " ".join(normalized_text.splitlines()[:3])
        for section_type, headings in section_patterns.items():
            for heading in headings:
                pattern = rf"\b{re.escape(heading)}\b"
                if re.search(pattern, opening_text):
                    return section_type
        return None