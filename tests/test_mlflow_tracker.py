from pathlib import Path

from swathi_ai.mlflow_tracker import (
    MLflowTracker,
    RetrievalMetrics,
)


def test_log_retrieval_creates_run(
    tmp_path: Path,
) -> None:
    tracker = MLflowTracker(
        experiment_name="test-rag",
        tracking_directory=tmp_path / "mlruns",
    )

    run_id = tracker.log_retrieval(
        RetrievalMetrics(
            query="What is machine learning?",
            retrieval_method="hybrid",
            top_k=5,
            retrieved_chunks=3,
            latency_ms=42.5,
            embedding_model="fake-model",
            average_score=0.81,
        )
    )

    assert run_id
    assert isinstance(run_id, str)


def test_measure_retrieval_logs_metrics(
    tmp_path: Path,
) -> None:
    tracker = MLflowTracker(
        experiment_name="test-context",
        tracking_directory=tmp_path / "mlruns",
    )

    with tracker.measure_retrieval(
        query="Explain RAG",
        retrieval_method="bm25-faiss",
        top_k=4,
        embedding_model="test-model",
    ) as result:
        result["retrieved_chunks"] = 4
        result["average_score"] = 0.72