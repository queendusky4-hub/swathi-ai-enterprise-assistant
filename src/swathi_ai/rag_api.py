
from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4
from .database import Base, engine
from . import models
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from .config import settings
from .auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserRecord,
    UserResponse,
    authenticate_user,
    build_token_response,
    build_user_response,
    get_current_user,
    register_user,
)

from .services import (
    get_document_service,
    get_engine,
    get_repository,
)


ResponseFormat = Literal[
    "Auto detect",
    "Tamil only",
    "Tanglish only",
    "English only",
    "All three",
]


app = FastAPI(
    title="Swathi AI",
    description=(
        "Multilingual Tamil, Tanglish, and English AI assistant "
        "with chat, document upload, vector search, and RAG."
    ),
    version="3.0.0",
)

Base.metadata.create_all(bind=engine)

# =========================================================
# Chat schemas
# =========================================================


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User message",
    )

    session_id: str | None = Field(
        default=None,
        description="Existing chat session ID",
    )

    online: bool = Field(
        default=True,
        description="Use the online LLM when needed",
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


# =========================================================
# Document schemas
# =========================================================


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
    )


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    score: float
    page_number: int | None = None
    section_type: str | None = None
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


# =========================================================
# Document question-answering schemas
# =========================================================


class DocumentAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about uploaded documents",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    online: bool = Field(
        default=True,
        description="Use the online LLM to generate the answer",
    )

    response_format: ResponseFormat = Field(
        default="Auto detect",
    )


class SourceItem(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    score: float
    text: str
    page_number: int | None = None
    section_type: str | None = None
    chunk_index: int | None = None


class DocumentAskResponse(BaseModel):
    answer: str
    source: str
    sources: list[SourceItem]


# =========================================================
# Basic routes
# =========================================================


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Swathi AI API is running",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
        "model_status": "/model/status",
        "document_upload": "/documents/upload",
        "document_search": "/documents/search",
        "document_ask": "/documents/ask",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "application": "Swathi AI",
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


# =========================================================
# Chat route
# =========================================================


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    current_user: Annotated[
        UserRecord,
        Depends(get_current_user),
    ],
) -> ChatResponse:
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


# =========================================================
# Document upload route
# =========================================================


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        service = get_document_service()

        result = service.index_document(
            filename=file.filename,
            file_bytes=file_bytes,
        )

        return UploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {error}",
        ) from error


# =========================================================
# Document search route
# =========================================================


@app.post(
    "/documents/search",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
) -> SearchResponse:
    try:
        service = get_document_service()

        results = service.search(
            query=request.query,
            top_k=request.top_k,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    filename=item.filename,
                    text=item.text,
                    score=float(item.score),
                    page_number=getattr(
                        item,
                        "page_number",
                        None,
                    ),
                    section_type=getattr(
                        item,
                        "section_type",
                        None,
                    ),
                    chunk_index=getattr(
                        item,
                        "chunk_index",
                        None,
                    ),
                )
                for item in results
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {error}",
        ) from error


# =========================================================
# Ask uploaded documents route
# =========================================================


@app.post(
    "/documents/ask",
    response_model=DocumentAskResponse,
)
def ask_documents(
    request: DocumentAskRequest,
) -> DocumentAskResponse:
    try:
        document_service = get_document_service()
        engine = get_engine()

        search_results = document_service.search(
            query=request.question,
            top_k=request.top_k,
        )

        if not search_results:
            return DocumentAskResponse(
                answer=(
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                source="document_rag",
                sources=[],
            )

        context_parts: list[str] = []

        for index, item in enumerate(
            search_results,
            start=1,
        ):
            page_number = getattr(
                item,
                "page_number",
                None,
            )

            page_text = (
                f", page {page_number}"
                if page_number is not None
                else ""
            )

            context_parts.append(
                f"[Source {index}: {item.filename}{page_text}]\n"
                f"{item.text}"
            )

        context = "\n\n".join(context_parts)

        prompt = (
            "You are answering a question using uploaded document "
            "content.\n\n"
            "Rules:\n"
            "1. Use only the document context below.\n"
            "2. Do not invent information.\n"
            "3. If the answer is not present, clearly say the "
            "uploaded documents do not contain enough information.\n"
            "4. Keep the answer clear and accurate.\n"
            "5. Refer to source numbers where useful.\n\n"
            f"Document context:\n{context}\n\n"
            f"User question:\n{request.question}"
        )

        result = engine.respond(
            text=prompt,
            online=request.online,
            response_format=request.response_format,
            history=[],
        )

        return DocumentAskResponse(
            answer=result.text,
            source="document_rag",
            sources=[
                SourceItem(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    filename=item.filename,
                    score=float(item.score),
                    text=item.text,
                    page_number=getattr(
                        item,
                        "page_number",
                        None,
                    ),
                    section_type=getattr(
                        item,
                        "section_type",
                        None,
                    ),
                    chunk_index=getattr(
                        item,
                        "chunk_index",
                        None,
                    ),
                )
                for item in search_results
            ],
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document question answering failed: {error}",
        ) from error
# =========================================================
# Authentication routes
# =========================================================

@app.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=201,
)
def register(
    request: RegisterRequest,
) -> UserResponse:
    user = register_user(request)
    return build_user_response(user)


@app.post(
    "/auth/login",
    response_model=TokenResponse,
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    user = authenticate_user(
        email=form_data.username,
        password=form_data.password,
    )

    return build_token_response(user)

@app.get(
    "/auth/me",
    response_model=UserResponse,
)
def read_current_user(
    current_user: Annotated[
        UserRecord,
        Depends(get_current_user),
    ],
) -> UserResponse:
    return build_user_response(current_user)