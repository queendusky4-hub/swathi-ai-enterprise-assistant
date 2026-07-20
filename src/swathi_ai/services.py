from __future__ import annotations

from functools import lru_cache

from .classifier import IntentClassifier
from .config import settings
from .database import ChatRepository
from .document_service import DocumentRAGService
from .engine import ChatEngine
from .llm import LLMClient


@lru_cache(maxsize=1)
def get_repository() -> ChatRepository:
    return ChatRepository(settings.database_path)


@lru_cache(maxsize=1)
def get_engine() -> ChatEngine:
    classifier = IntentClassifier(settings.model_path)

    return ChatEngine(
        classifier,
        LLMClient(settings),
        settings.confidence_threshold,
    )


@lru_cache(maxsize=1)
def get_document_service() -> DocumentRAGService:
    return DocumentRAGService()
