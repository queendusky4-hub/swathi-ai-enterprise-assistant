from __future__ import annotations

import time

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


HTTP_REQUESTS_TOTAL = Counter(
    "swathi_ai_http_requests_total",
    "Total number of HTTP requests.",
    [
        "method",
        "path",
        "status",
    ],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "swathi_ai_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    [
        "method",
        "path",
    ],
)

HTTP_ERRORS_TOTAL = Counter(
    "swathi_ai_http_errors_total",
    "Total number of HTTP error responses.",
    [
        "method",
        "path",
        "status",
    ],
)

RAG_SEARCHES_TOTAL = Counter(
    "swathi_ai_rag_searches_total",
    "Total number of RAG searches.",
)

RAG_SEARCH_DURATION_SECONDS = Histogram(
    "swathi_ai_rag_search_duration_seconds",
    "RAG retrieval latency in seconds.",
)

RAG_RESULTS_TOTAL = Counter(
    "swathi_ai_rag_results_total",
    "Total number of chunks returned by RAG.",
)


async def prometheus_http_middleware(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        HTTP_ERRORS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status="500",
        ).inc()

        raise

    duration = (
        time.perf_counter()
        - start_time
    )

    status_code = str(
        response.status_code
    )

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status=status_code,
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=request.url.path,
    ).observe(duration)

    if response.status_code >= 400:
        HTTP_ERRORS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status=status_code,
        ).inc()

    return response


def metrics_response() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )