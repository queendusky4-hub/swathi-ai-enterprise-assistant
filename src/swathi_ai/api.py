from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .services import get_engine, get_repository

app = FastAPI(
    title="Swathi AI API",
    version="2.0.0",
    description="Tamil, Tanglish and English conversational AI API.",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=120)
    online: bool = True
    show_all_formats: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    source: str
    intent: str | None = None
    confidence: float | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": settings.app_name}


@app.get("/model/status")
def model_status() -> dict[str, bool | str]:
    engine = get_engine()
    return {
        "bert_model_available": engine.classifier.available,
        "online_llm_configured": engine.llm.configured,
        "model_path": str(settings.model_path),
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or f"api_{uuid4().hex}"
    engine = get_engine()
    repository = get_repository()

    repository.save(session_id, "user", payload.message)
    result = engine.respond(
        payload.message,
        online=payload.online,
        show_all=payload.show_all_formats,
    )
    repository.save(session_id, "assistant", result.text)

    return ChatResponse(
        session_id=session_id,
        reply=result.text,
        source=result.source,
        intent=result.intent,
        confidence=result.confidence,
    )
