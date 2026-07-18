from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DocumentSection:
    text: str
    page_number: int | None = None
    section_type: str = "text"


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int | None
    section_type: str
    chunk_index: int


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_sections(
        self,
        document_id: str,
        filename: str,
        sections: Iterable[DocumentSection],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section in sections:
            text = " ".join(section.text.split())

            if not text:
                continue

            start = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document_id}-{chunk_index}",
                            document_id=document_id,
                            filename=filename,
                            text=chunk_text,
                            page_number=section.page_number,
                            section_type=section.section_type,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                if end >= len(text):
                    break

                start = end - self.chunk_overlap

        return chunks