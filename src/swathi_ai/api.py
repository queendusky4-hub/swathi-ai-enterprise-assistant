from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4
from collections import Counter
from .monitoring import (
    metrics_response,
    prometheus_http_middleware,
)

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm
from .vision_service import get_vision_service
from pydantic import BaseModel, Field
from datetime import UTC, datetime

from .auth import (
    RegisterRequest,
    RegisterResponse,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
    create_current_user_dependency,
)
from .config import settings
from .schemas import SearchRequest, SearchResponse, SearchResult, UploadResponse
from .services import (
    get_auth_service,
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
    title=settings.app_name,
    description=(
        "Multilingual Tamil, Tanglish, and English AI assistant "
        "with authentication, chat history, document upload, "
        "document retrieval, citations, and document question answering."
    ),
    version="2.5.0",
)

app.middleware("http")(
    prometheus_http_middleware
)


@app.get(
    "/health",
    tags=["System"],
    summary="Check service health",
)
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "swathi-ai-api",
        "timestamp": datetime.now(UTC).isoformat(),
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("swathi_ai.api")


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


auth_service = get_auth_service()
get_current_user = create_current_user_dependency(auth_service)

@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics():
    return metrics_response()

# ---------------------------------------------------------------------
# REQUEST AND RESPONSE SCHEMAS
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(
        default=None,
        description="Existing conversation session ID",
    )
    online: bool = Field(
        default=True,
        description="Use the online LLM for unknown questions",
    )
    response_format: ResponseFormat = "Auto detect"
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional uploaded document IDs for document RAG.",
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


class DocumentAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about uploaded documents",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    online: bool = True
    response_format: ResponseFormat = "Auto detect"
    document_ids: list[str] | None = None


class DocumentSource(BaseModel):
    source_number: int
    document_id: str | None = None
    chunk_id: str | None = None
    filename: str
    page_number: int | None = None
    section_type: str | None = None
    chunk_index: int | None = None
    score: float | None = None
    text: str


class DocumentAskResponse(BaseModel):
    question: str
    answer: str
    source: str
    retrieved_chunks: int
    sources: list[DocumentSource]

class ImageAnalyzeResponse(BaseModel):
    filename: str
    question: str
    answer: str
    source: str
    mime_type: str


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _build_user_response(current_user: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        role=current_user.get("role", "user"),
    )


def _generate_document_answer(
    *,
    question: str,
    context: str,
    response_format: ResponseFormat,
    online: bool,
) -> tuple[str | None, str]:
    """Call the LLM directly and bypass greeting/intent classification."""
    if not online:
        return None, "document-rag-extractive"

    engine = get_engine()
    llm = engine.llm

    if not getattr(llm, "configured", False):
        return None, "document-rag-extractive"

    prompt = (
        "You are a document question-answering assistant.\n\n"
        "Rules:\n"
        "1. Answer only from the document context below.\n"
        "2. Do not greet the user.\n"
        "3. Do not introduce yourself.\n"
        "4. Do not use general knowledge or invent details.\n"
        "5. If the answer is absent, say: "
        "'The uploaded document does not contain enough information.'\n"
        "6. Give a direct and concise answer.\n"
        "7. For skills, tools, names, qualifications, or responsibilities, "
        "return a clean bullet list.\n"
        "8. Add source citations such as [1] or [2].\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )

    try:
        if hasattr(engine, "resolve_response_format"):
            language, _ = engine.resolve_response_format(
                text=question,
                response_format=response_format,
            )
        else:
            language = "english"

        answer = llm.generate(
            user_text=prompt,
            language=language,
            response_format=response_format,
            history=[],
        )
    except TypeError:
        try:
            answer = llm.generate(prompt)
        except Exception:
            return None, "document-rag-extractive"
    except Exception:
        return None, "document-rag-extractive"

    if not answer:
        return None, "document-rag-extractive"

    cleaned_answer = str(answer).strip()
    lowered = cleaned_answer.lower()

    greeting_phrases = (
        "hello! 🌺 i'm swathi ai",
        "hello! i'm swathi ai",
        "how can i help you",
    )

    if any(phrase in lowered for phrase in greeting_phrases):
        return None, "document-rag-extractive"

    return cleaned_answer, "document-rag-llm"


def _build_extractive_answer(
    question: str,
    sources: list[DocumentSource],
) -> str:
    if not sources:
        return "I could not find relevant information in the uploaded documents."

    question_lower = question.lower()
    asks_for_list = any(
        phrase in question_lower
        for phrase in (
            "list",
            "name the",
            "what are",
            "skills",
            "technologies",
            "tools",
            "qualifications",
            "responsibilities",
        )
    )

    if asks_for_list:
        return "\n".join(
            f"- {' '.join(source.text.split())} [{source.source_number}]"
            for source in sources
            if source.text.strip()
        )

    return "\n\n".join(
        f"{source.text} [{source.source_number}]"
        for source in sources
    )


# ---------------------------------------------------------------------
# BASIC ROUTES
# ---------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": f"{settings.app_name} API is running",
        "docs": "/docs",
        "register": "/auth/register",
        "login": "/auth/login",
        "profile": "/users/me",
        "health": "/health",
        "chat": "/chat",
        "document_upload": "/documents/upload",
        "document_search": "/documents/search",
        "document_ask": "/documents/ask",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "application": settings.app_name,
        "version": app.version,
    }


# ---------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------

@app.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
def register(request: RegisterRequest) -> RegisterResponse:
    try:
        user = auth_service.register(
            username=request.username,
            password=request.password,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return RegisterResponse(
        id=user["id"],
        username=user["username"],
        role=user.get("role", "user"),
        recovery_code=user["recovery_code"],
    )


@app.post(
    "/auth/reset-password",
    tags=["Authentication"],
)
def reset_password(
    request: ResetPasswordRequest,
) -> dict[str, str]:
    auth_service.reset_password(
        username=request.username,
        recovery_code=request.recovery_code,
        new_password=request.new_password,
    )

    return {
        "message": "Password reset successful."
    }


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    try:
        user = auth_service.authenticate(
            username=form_data.username,
            password=form_data.password,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    token = auth_service.create_access_token(user_id=user["id"])

    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


@app.post(
    "/auth/guest",
    response_model=TokenResponse,
    tags=["Authentication"],
)
def guest_login() -> TokenResponse:
    token = auth_service.create_guest_access_token()

    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


@app.get(
    "/auth/me",
    response_model=UserResponse,
    tags=["Authentication"],
)
def auth_me(
    current_user: dict = Depends(get_current_user),
) -> UserResponse:
    return _build_user_response(current_user)


@app.get(
    "/users/me",
    response_model=UserResponse,
    tags=["Authentication"],
)
def users_me(
    current_user: dict = Depends(get_current_user),
) -> UserResponse:
    return _build_user_response(current_user)


# ---------------------------------------------------------------------
# MODEL STATUS
# ---------------------------------------------------------------------

@app.get("/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    engine = get_engine()

    return ModelStatusResponse(
        bert_model_available=engine.classifier.available,
        online_llm_configured=engine.llm.configured,
        model_path=str(settings.model_path),
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )


# ---------------------------------------------------------------------
# NORMAL CHAT
# ---------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
) -> ChatResponse:
    repository = get_repository()
    engine = get_engine()

    session_id = request.session_id or str(uuid4())
    history = repository.load(session_id)

    try:
        result = engine.respond(
            text=request.message,
            online=request.online,
            response_format=request.response_format,
            history=history,
        )
    except TypeError:
        result = engine.respond(
            text=request.message,
            online=request.online,
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


@app.get("/documents/debug")
def documents_debug(
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    service = get_document_service()

    chunks = getattr(service, "_chunks", [])
    documents = getattr(service, "_documents", {})

    return {
        "service_class": service.__class__.__name__,
        "service_module": service.__class__.__module__,
        "service_object_id": id(service),
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "documents": (
            list(documents.values())
            if isinstance(documents, dict)
            else str(documents)
        ),
        "chunks": [
            {
                "document_id": getattr(chunk, "document_id", None),
                "chunk_id": getattr(chunk, "chunk_id", None),
                "filename": getattr(chunk, "filename", None),
                "text_preview": str(
                    getattr(chunk, "text", "")
                )[:200],
            }
            for chunk in chunks
        ],
    }

@app.get("/documents/debug-search")
def documents_debug_search(
    current_user: dict = Depends(get_current_user),
):
    service = get_document_service()

    if not service._chunks:
        return {
            "error": "No chunks are currently indexed."
        }

    chunk = service._chunks[0]
    query = "What degree does the candidate have?"

    query_tokens = service._tokenize(query)
    chunk_tokens = service._tokenize(chunk.text)

    query_vector = Counter(query_tokens)
    chunk_vector = Counter(chunk_tokens)

    cosine = service._cosine_similarity(
        query_vector,
        chunk_vector,
    )

    results = service.search(
        query=query,
        top_k=5,
        document_ids=None,
    )

    return {
        "service_object_id": id(service),
        "chunks_in_memory": len(service._chunks),
        "query_tokens": query_tokens,
        "chunk_tokens": chunk_tokens[:50],
        "cosine": cosine,
        "chunk_preview": chunk.text[:500],
        "result_count": len(results),
        "results": [
            {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "filename": result.filename,
                "score": result.score,
                "text_preview": result.text[:300],
            }
            for result in results
        ],
    }

@app.get("/llm/test")
def llm_test():
    engine = get_engine()

    answer = engine.llm.generate(
        user_text="Reply with exactly: Hello World",
        language="english",
        response_format="short",
        history=[],
    )

    return {
        "answer": answer,
        "last_error": getattr(
            engine.llm,
            "last_error",
            None,
        ),
    }
# ---------------------------------------------------------------------
# DOCUMENT UPLOAD
# ---------------------------------------------------------------------

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    service = get_document_service()

    try:
        result = service.index_document(
            filename=file.filename,
            file_bytes=file_bytes,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {error}",
        ) from error

    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
    )


# ---------------------------------------------------------------------
# DOCUMENT SEARCH
# ---------------------------------------------------------------------

@app.post("/documents/search", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
) -> SearchResponse:
    service = get_document_service()

    try:
        results = service.search(
            query=request.query,
            top_k=request.top_k,
            document_ids=getattr(request, "document_ids", None),
        )
    except TypeError:
        results = service.search(
            query=request.query,
            top_k=request.top_k,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {error}",
        ) from error

    return SearchResponse(
        results=[
            SearchResult(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                text=item.text,
                score=float(item.score),
                page_number=getattr(item, "page_number", None),
                section_type=getattr(item, "section_type", None),
                chunk_index=int(getattr(item, "chunk_index", 0) or 0),
            )
            for item in results
        ]
    )


# ---------------------------------------------------------------------
# DOCUMENT QUESTION ANSWERING
# ---------------------------------------------------------------------

@app.post("/documents/ask", response_model=DocumentAskResponse)
def ask_documents(
    request: DocumentAskRequest,
    current_user: dict = Depends(get_current_user),
) -> DocumentAskResponse:
    service = get_document_service()

    try:
        results = service.search(
            query=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except TypeError:
        results = service.search(
            query=request.question,
            top_k=request.top_k,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document retrieval failed: {error}",
        ) from error

    if not results:
        return DocumentAskResponse(
            question=request.question,
            answer=(
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            source="document-rag",
            retrieved_chunks=0,
            sources=[],
        )

    source_items: list[DocumentSource] = []
    context_parts: list[str] = []

    for number, item in enumerate(results, start=1):
        text = str(getattr(item, "text", "") or "").strip()

        if not text:
            continue

        filename = str(getattr(item, "filename", "Unknown document"))
        page_number = getattr(item, "page_number", None)
        location = f", page {page_number}" if page_number is not None else ""

        context_parts.append(
            f"[{number}] {filename}{location}\n{text}"
        )

        source_items.append(
            DocumentSource(
                source_number=number,
                document_id=getattr(item, "document_id", None),
                chunk_id=getattr(item, "chunk_id", None),
                filename=filename,
                page_number=page_number,
                section_type=getattr(item, "section_type", None),
                chunk_index=getattr(item, "chunk_index", None),
                score=float(getattr(item, "score", 0.0)),
                text=text,
            )
        )

    if not source_items:
        return DocumentAskResponse(
            question=request.question,
            answer=(
                "I could not find readable information "
                "in the retrieved document chunks."
            ),
            source="document-rag",
            retrieved_chunks=0,
            sources=[],
        )

    context = "\n\n".join(context_parts)

    answer, answer_source = _generate_document_answer(
        question=request.question,
        context=context,
        response_format=request.response_format,
        online=request.online,
    )

    if not answer:
        answer = _build_extractive_answer(
            question=request.question,
            sources=source_items,
        )

    return DocumentAskResponse(
        question=request.question,
        answer=answer,
        source=answer_source,
        retrieved_chunks=len(source_items),
        sources=source_items,
    )

# ---------------------------------------------------------------------
# IMAGE ANALYSIS
# ---------------------------------------------------------------------

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


@app.post(
    "/images/analyze",
    response_model=ImageAnalyzeResponse,
    tags=["Image Analysis"],
)
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form("Describe and analyze this image."),
    response_format: ResponseFormat = Form("Auto detect"),
    current_user: dict = Depends(get_current_user),
) -> ImageAnalyzeResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An image filename is required.",
        )

    mime_type = str(file.content_type or "").strip().lower()

    if mime_type == "image/jpg":
        mime_type = "image/jpeg"

    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported image format. "
                "Please upload PNG, JPG, JPEG, or WEBP."
            ),
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    # Maximum image size: 10 MB
    maximum_size = 10 * 1024 * 1024

    if len(image_bytes) > maximum_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded image must be smaller than 10 MB.",
        )

    cleaned_question = str(question or "").strip()

    if not cleaned_question:
        cleaned_question = "Describe and analyze this image."

    vision_service = get_vision_service()

    try:
        answer = vision_service.analyze_image(
            image_bytes=image_bytes,
            question=cleaned_question,
            mime_type=mime_type,
            response_format=response_format,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {error}",
        ) from error

    return ImageAnalyzeResponse(
        filename=file.filename,
        question=cleaned_question,
        answer=answer,
        source="vision-llm",
        mime_type=mime_type,
    )