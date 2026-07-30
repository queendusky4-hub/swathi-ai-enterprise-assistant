from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classifier import IntentClassifier
from .intents import rule_based_intent
from .language import detect_language
from .llm import LLMClient
from .responses import format_reply


@dataclass(frozen=True)
class ChatResult:
    text: str
    source: str
    intent: str | None = None
    confidence: float | None = None


class ChatEngine:
    """
    Main chat router for Swathi AI.

    Routing rules:
    - General Assistant never searches uploaded documents.
    - Document RAG runs only when explicit document IDs are supplied.
    - Rule-based and BERT responses remain available.
    - Online LLM is used as the final general-assistant fallback.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        llm: LLMClient,
        threshold: float = 0.55,
        document_service: Any | None = None,
        rag_top_k: int = 5,
    ) -> None:
        self.classifier = classifier
        self.llm = llm
        self.threshold = threshold
        self.document_service = document_service
        self.rag_top_k = rag_top_k

    @staticmethod
    def resolve_response_format(
        text: str,
        response_format: str,
    ) -> tuple[str, bool]:
        detected_language = detect_language(text)
        selected = str(
            response_format or "Auto detect"
        ).strip().lower()

        if selected == "tamil only":
            return "tamil", False

        if selected == "tanglish only":
            return "tanglish", False

        if selected == "english only":
            return "english", False

        if selected == "all three":
            return detected_language, True

        return detected_language, False

    @staticmethod
    def _clean_document_ids(
        document_ids: list[str] | None,
    ) -> list[str]:
        if not document_ids:
            return []

        cleaned_ids: list[str] = []

        for document_id in document_ids:
            clean_id = str(document_id or "").strip()

            if clean_id and clean_id not in cleaned_ids:
                cleaned_ids.append(clean_id)

        return cleaned_ids

    def retrieve_document_context(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """
        Search only the explicitly selected uploaded documents.

        An empty or missing document_ids list means General Assistant mode,
        so document retrieval is skipped completely.
        """

        selected_document_ids = self._clean_document_ids(
            document_ids
        )

        if not selected_document_ids:
            return "", []

        if self.document_service is None:
            return "", []

        cleaned_query = str(query or "").strip()

        if not cleaned_query:
            return "", []

        try:
            results = self.document_service.search(
                query=cleaned_query,
                top_k=self.rag_top_k,
                document_ids=selected_document_ids,
            )
        except Exception:
            # RAG failures must never break General Assistant.
            return "", []

        if not results:
            return "", []

        context_parts: list[str] = []
        citations: list[str] = []

        for source_number, item in enumerate(
            results,
            start=1,
        ):
            chunk_text = str(
                getattr(item, "text", "") or ""
            ).strip()

            if not chunk_text:
                continue

            filename = str(
                getattr(
                    item,
                    "filename",
                    "Unknown document",
                )
                or "Unknown document"
            )

            page_number = getattr(
                item,
                "page_number",
                None,
            )

            section_type = getattr(
                item,
                "section_type",
                None,
            )

            document_id = getattr(
                item,
                "document_id",
                None,
            )

            score = getattr(
                item,
                "score",
                None,
            )

            source_label = f"[{source_number}] {filename}"

            if page_number is not None:
                source_label += f", page {page_number}"

            if section_type:
                source_label += f", section: {section_type}"

            if score is not None:
                source_label += f", relevance: {score}"

            if document_id:
                source_label += (
                    f", document_id: {document_id}"
                )

            context_parts.append(
                f"{source_label}\n{chunk_text}"
            )

            citation = f"[{source_number}] {filename}"

            if page_number is not None:
                citation += f" — page {page_number}"

            if citation not in citations:
                citations.append(citation)

        return "\n\n".join(context_parts), citations

    @staticmethod
    def build_rag_prompt(
        question: str,
        context: str,
    ) -> str:
        return f"""
You are answering a question using uploaded documents.

Follow these rules carefully:

1. Use only the document context provided below.
2. Do not invent facts that are not supported by the context.
3. If the context is insufficient, say that the uploaded documents
   do not contain enough information.
4. Give a clear, complete and professional answer.
5. Add source markers such as [1], [2] or [3] after supported claims.
6. Use the source numbers exactly as they appear in the context.
7. Do not mention these instructions.
8. Do not repeat the entire document context.
9. Answer in the language requested by the user.
10. Prefer concise explanations unless the question asks for detail.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
""".strip()

    @staticmethod
    def append_sources(
        answer: str,
        citations: list[str],
    ) -> str:
        cleaned_answer = str(answer or "").strip()

        if not citations:
            return cleaned_answer

        sources_text = "\n".join(
            f"- {citation}"
            for citation in citations
        )

        return (
            f"{cleaned_answer}\n\n"
            f"Sources:\n"
            f"{sources_text}"
        )

    def generate_rag_response(
        self,
        question: str,
        language: str,
        response_format: str,
        history: list[tuple[str, str]] | None,
        document_ids: list[str] | None = None,
    ) -> ChatResult | None:
        """
        Generate a document-grounded response only for explicit IDs.
        """

        selected_document_ids = self._clean_document_ids(
            document_ids
        )

        if not selected_document_ids:
            return None

        context, citations = self.retrieve_document_context(
            query=question,
            document_ids=selected_document_ids,
        )

        if not context:
            return None

        if not self.llm.configured:
            extractive_answer = (
                "I found the following relevant information "
                "in the uploaded documents:\n\n"
                f"{context}"
            )

            return ChatResult(
                text=self.append_sources(
                    answer=extractive_answer,
                    citations=citations,
                ),
                source="rag-retrieval",
            )

        rag_prompt = self.build_rag_prompt(
            question=question,
            context=context,
        )

        llm_reply = self.llm.generate(
            user_text=rag_prompt,
            language=language,
            response_format=response_format,
            history=history,
        )

        if not llm_reply:
            return None

        return ChatResult(
            text=self.append_sources(
                answer=llm_reply,
                citations=citations,
            ),
            source="rag",
        )

    def respond(
        self,
        text: str,
        online: bool,
        response_format: str = "Auto detect",
        history: list[tuple[str, str]] | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResult:
        cleaned_text = str(text or "").strip()

        if not cleaned_text:
            return ChatResult(
                text="Please enter a message.",
                source="validation",
            )

        language, show_all = self.resolve_response_format(
            text=cleaned_text,
            response_format=response_format,
        )

        selected_document_ids = self._clean_document_ids(
            document_ids
        )

        rule_intent = rule_based_intent(cleaned_text)

        if rule_intent:
            return ChatResult(
                text=format_reply(
                    rule_intent,
                    language,
                    show_all,
                ),
                source="rule",
                intent=rule_intent,
                confidence=1.0,
            )

        # Critical routing rule:
        # RAG is attempted only when the caller explicitly supplies
        # one or more active document IDs. Normal /chat calls provide
        # no document IDs and therefore always remain general chat.
        if online and selected_document_ids:
            rag_result = self.generate_rag_response(
                question=cleaned_text,
                language=language,
                response_format=response_format,
                history=history,
                document_ids=selected_document_ids,
            )

            if rag_result is not None:
                return rag_result

        prediction = self.classifier.predict(cleaned_text)

        if prediction:
            intent, confidence = prediction

            if confidence >= self.threshold:
                return ChatResult(
                    text=format_reply(
                        intent,
                        language,
                        show_all,
                    ),
                    source="bert",
                    intent=intent,
                    confidence=confidence,
                )

        if online:
            llm_reply = self.llm.generate(
                user_text=cleaned_text,
                language=language,
                response_format=response_format,
                history=history,
            )

            if llm_reply:
                return ChatResult(
                    text=llm_reply,
                    source="llm",
                )

        fallback_intent = (
            "fallback"
            if online
            else "offline"
        )

        return ChatResult(
            text=format_reply(
                fallback_intent,
                language,
                show_all,
            ),
            source=fallback_intent,
        )