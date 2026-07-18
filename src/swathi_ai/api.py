from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .services import get_engine, get_repository


ResponseFormat = Literal[
    "Auto detect",
    "Tamil only",
    "Tanglish only",
    "English only",
    "All three",
]


app = FastAPI(
    title=settings.app_name,
    description=(
        "Multilingual Tamil, Tanglish, and English AI assistant "
        "with intent classification, Gemini integration, "
        "session memory, and persistent chat history."
    ),
    version="2.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User message",
    )

    session_id: str | None = Field(
        default=None,
        description="Existing conversation session ID",
    )

    online: bool = Field(
        default=True,
        description="Use the online LLM for unknown questions",
    )

    response_format: ResponseFormat = Field(
        default="Auto detect",
        description=(
            "Auto detect, Tamil only, Tanglish only, "
            "English only, or All three"
        ),
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    source: str
    intent: str | None = None
    confidence: float | None = None


class ModelStatusResponse(BaseModel):
    bert_model_available: bool
    online_llm_configured: bool
    model_path: str
    llm_model: str
    llm_base_url: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": f"{settings.app_name} API is running",
        "docs": "/docs",
        "health": "/health",
        "model_status": "/model/status",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": app.version,
    }


@app.get(
    "/model/status",
    response_model=ModelStatusResponse,
)
def model_status() -> ModelStatusResponse:
    engine = get_engine()

    return ModelStatusResponse(
        bert_model_available=engine.classifier.available,
        online_llm_configured=engine.llm.configured,
        model_path=str(settings.model_path),
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    repository = get_repository()
    engine = get_engine()

    session_id = request.session_id or f"api_{uuid4()}"

    history = repository.load(session_id)

    result = engine.respond(
        text=request.message,
        online=request.online,
        response_format=request.response_format,
        history=history,
    )

    repository.save(
        session_id=session_id,
        role="user",
        message=request.message,
    )

    repository.save(
        session_id=session_id,
        role="assistant",
        message=result.text,
    )

    return ChatResponse(
        session_id=session_id,
        reply=result.text,
        source=result.source,
        intent=result.intent,
        confidence=result.confidence,
    )