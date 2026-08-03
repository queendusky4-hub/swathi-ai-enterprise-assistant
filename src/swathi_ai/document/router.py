from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from swathi_ai.document.embedding import embed
from swathi_ai.document.schemas import AskRequest, SearchRequest
from swathi_ai.document.service import DocumentRAGService
from swathi_ai.document.vector_store import VectorStore


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload and index a PDF, DOCX, or TXT document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    allowed_extensions = (
        ".pdf",
        ".docx",
        ".txt",
    )

    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported.",
        )

    try:
        service = DocumentRAGService()

        result = await service.upload(file)

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not upload document: {error}",
        ) from error


@router.get("")
def list_documents():
    """
    List all documents stored in the FAISS index.
    """

    try:
        store = VectorStore()

        documents = store.list_documents()

        return {
            "count": len(documents),
            "documents": documents,
        }

    except FileNotFoundError:
        return {
            "count": 0,
            "documents": [],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not list documents: {error}",
        ) from error


@router.post("/search")
def search_documents(
    request: SearchRequest,
):
    """
    Search relevant chunks from uploaded documents.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        query_embedding = embed([query])

        store = VectorStore()

        results = store.search(
            embedding=query_embedding,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found. Upload a document first.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {error}",
        ) from error


@router.post("/ask")
def ask_documents(
    request: AskRequest,
):
    """
    Retrieve the most relevant document passages for a question.

    This version returns an extractive answer from the retrieved chunks.
    LLM generation can be connected after identifying how the existing
    ChatEngine and LLMClient are initialized.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        question_embedding = embed([question])

        store = VectorStore()

        results = store.search(
            embedding=question_embedding,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
            }

        sources: list[dict] = []
        answer_parts: list[str] = []

        for source_number, result in enumerate(
            results,
            start=1,
        ):
            filename = result.get(
                "filename",
                "Unknown document",
            )

            page = result.get(
                "page",
                1,
            )

            text = (
                result.get("text")
                or result.get("chunk")
                or result.get("content")
                or ""
            ).strip()

            if not text:
                continue

            answer_parts.append(
                f"{text} [{source_number}]"
            )

            sources.append(
                {
                    "source_number": source_number,
                    "document_id": result.get(
                        "document_id"
                    ),
                    "filename": filename,
                    "page": page,
                    "score": result.get(
                        "score"
                    ),
                    "text": text,
                }
            )

        if not answer_parts:
            return {
                "question": question,
                "answer": (
                    "Relevant document records were found, "
                    "but they did not contain readable text."
                ),
                "sources": sources,
            }

        answer = "\n\n".join(answer_parts)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found. Upload a document first.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not answer the question: {error}",
        ) from error