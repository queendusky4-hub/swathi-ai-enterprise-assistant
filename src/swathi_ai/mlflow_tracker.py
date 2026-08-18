from __future__ import annotations

import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlflow


@dataclass(frozen=True)
class RetrievalMetrics:
    query: str
    retrieval_method: str
    top_k: int
    retrieved_chunks: int
    latency_ms: float
    embedding_model: str | None = None
    average_score: float | None = None


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "swathi-ai-rag",
        tracking_directory: Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name

        if tracking_directory is None:
           tracking_directory = Path(
            os.getenv("MLFLOW_TRACKING_DIR", "mlruns")
    )

        tracking_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        tracking_uri = tracking_directory.resolve().as_uri()

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_retrieval(
        self,
        metrics: RetrievalMetrics,
    ) -> str:
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "retrieval_method": (
                        metrics.retrieval_method
                    ),
                    "top_k": metrics.top_k,
                    "embedding_model": (
                        metrics.embedding_model
                        or "not_configured"
                    ),
                }
            )

            mlflow.log_metrics(
                {
                    "retrieved_chunks": float(
                        metrics.retrieved_chunks
                    ),
                    "latency_ms": metrics.latency_ms,
                }
            )

            if metrics.average_score is not None:
                mlflow.log_metric(
                    "average_score",
                    metrics.average_score,
                )

            mlflow.set_tag(
                "query",
                metrics.query[:500],
            )

            return run.info.run_id

    @contextmanager
    def measure_retrieval(
        self,
        *,
        query: str,
        retrieval_method: str,
        top_k: int,
        embedding_model: str | None = None,
    ) -> Iterator[dict[str, float | int]]:
        start_time = time.perf_counter()

        result: dict[str, float | int] = {
            "retrieved_chunks": 0,
            "average_score": 0.0,
        }

        try:
            yield result
        finally:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            retrieved_chunks = int(
                result.get(
                    "retrieved_chunks",
                    0,
                )
            )

            average_score = float(
                result.get(
                    "average_score",
                    0.0,
                )
            )

            self.log_retrieval(
                RetrievalMetrics(
                    query=query,
                    retrieval_method=(
                        retrieval_method
                    ),
                    top_k=top_k,
                    retrieved_chunks=(
                        retrieved_chunks
                    ),
                    latency_ms=round(
                        latency_ms,
                        3,
                    ),
                    embedding_model=(
                        embedding_model
                    ),
                    average_score=(
                        average_score
                    ),
                )
            )