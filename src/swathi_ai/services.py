from __future__ import annotations

from functools import lru_cache

from .auth import AuthService
from .classifier import IntentClassifier
from .config import settings
from .database import AuthRepository, ChatRepository
from .document_service import DocumentRAGService
from .engine import ChatEngine
from .llm import LLMClient
from .postgres_auth import PostgresAuthRepository


@lru_cache(maxsize=1)
def get_repository() -> ChatRepository:
    return ChatRepository(
        settings.database_path
    )


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    if (
        settings.auth_db_host
        and settings.auth_db_user
        and settings.auth_db_password
    ):
        repository = PostgresAuthRepository(
            host=settings.auth_db_host,
            database=settings.auth_db_name,
            user=settings.auth_db_user,
            password=settings.auth_db_password,
        )
    else:
        repository = AuthRepository(
            settings.database_path
        )

    return AuthService(
        repository=repository,
        token_expiry_hours=24,
    )


@lru_cache(maxsize=1)
def get_document_service() -> DocumentRAGService:
    """
    Return one shared document service instance.

    The same instance is used by:
    - document upload
    - document search
    - document management
    - ChatEngine RAG retrieval
    """
    return DocumentRAGService()


@lru_cache(maxsize=1)
def get_engine() -> ChatEngine:
    """
    Return one shared ChatEngine configured with:
    - intent classifier
    - online LLM
    - document RAG service
    """

    classifier = IntentClassifier(
        settings.model_path
    )

    llm = LLMClient(
        settings
    )

    document_service = get_document_service()

    return ChatEngine(
        classifier=classifier,
        llm=llm,
        threshold=settings.confidence_threshold,
        document_service=document_service,
        rag_top_k=5,
    )