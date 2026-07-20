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

@app.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    service = get_document_service()
    file_bytes = await file.read()

    result = service.index_document(
        filename=file.filename or "uploaded_file",
        file_bytes=file_bytes,
    )

    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
    )


@app.post(
    "/documents/search",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
) -> SearchResponse:
    service = get_document_service()

    results = service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return SearchResponse(
        results=[
            SearchResult(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                text=item.text,
                score=item.score,
                page_number=item.page_number,
                section_type=item.section_type,
                chunk_index=item.chunk_index,
            )
            for item in results
        ]
    )
