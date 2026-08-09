from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow

from swathi_ai.document_service import (
    DocumentRAGService,
    StoredChunk,
)


DATASET_PATH = Path(
    "experiments/rag_eval_dataset.json"
)

CORPUS_PATH = Path(
    "experiments/rag_eval_corpus.json"
)

MLFLOW_DATABASE = Path("mlflow.db")

EXPERIMENT_NAME = "swathi-ai-rag-evaluation"

TOP_K = 5


@dataclass(frozen=True)
class EvaluationExample:
    query: str
    relevant_chunk_ids: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hit_rate: float


@dataclass(frozen=True)
class EvaluationSummary:
    queries: int
    corpus_chunks: int
    top_k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Swathi AI hybrid RAG retrieval."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print evaluation summary as JSON.",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging.",
    )

    return parser.parse_args()


def load_dataset(
    path: Path,
) -> list[EvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    raw_data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    examples: list[EvaluationExample] = []

    for item in raw_data:
        query = str(
            item["query"]
        ).strip()

        relevant_chunk_ids = [
            str(chunk_id).strip()
            for chunk_id
            in item["relevant_chunk_ids"]
            if str(chunk_id).strip()
        ]

        if not query:
            continue

        if not relevant_chunk_ids:
            continue

        examples.append(
            EvaluationExample(
                query=query,
                relevant_chunk_ids=relevant_chunk_ids,
            )
        )

    return examples


def load_corpus(
    path: Path,
) -> list[StoredChunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation corpus not found: {path}"
        )

    raw_data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    chunks: list[StoredChunk] = []

    for index, item in enumerate(raw_data):
        chunk_id = str(
            item["chunk_id"]
        ).strip()

        text = str(
            item["text"]
        ).strip()

        if not chunk_id or not text:
            continue

        chunks.append(
            StoredChunk(
                chunk_id=chunk_id,
                document_id="evaluation-document",
                filename="rag_eval_corpus.txt",
                text=text,
                page_number=None,
                section_type=None,
                chunk_index=index,
            )
        )

    return chunks


def build_evaluation_service() -> DocumentRAGService:
    service = DocumentRAGService()

    corpus = load_corpus(
        CORPUS_PATH
    )

    if not corpus:
        raise RuntimeError(
            "Evaluation corpus is empty."
        )

    service._chunks = corpus

    service.vector_store.rebuild(
        chunk_ids=[
            chunk.chunk_id
            for chunk in corpus
        ],
        texts=[
            chunk.text
            for chunk in corpus
        ],
    )

    return service


def real_retrieval(
    service: DocumentRAGService,
    query: str,
) -> list[str]:
    results = service.search(
        query=query,
        top_k=TOP_K,
    )

    return [
        result.chunk_id
        for result in results
    ]


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    top_results = retrieved[:k]

    if not top_results:
        return 0.0

    relevant_retrieved = sum(
        1
        for chunk_id in top_results
        if chunk_id in relevant
    )

    return (
        relevant_retrieved
        / len(top_results)
    )


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved[:k]
        if chunk_id in relevant
    )

    return (
        relevant_retrieved
        / len(relevant)
    )


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    for rank, chunk_id in enumerate(
        retrieved,
        start=1,
    ):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved[:k],
        start=1,
    ):
        if chunk_id in relevant:
            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    return float(
        any(
            chunk_id in relevant
            for chunk_id in retrieved[:k]
        )
    )


def evaluate_query(
    *,
    query: str,
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    top_k: int,
) -> EvaluationResult:
    relevant = set(
        relevant_chunk_ids
    )

    return EvaluationResult(
        query=query,
        precision_at_k=precision_at_k(
            retrieved_chunk_ids,
            relevant,
            top_k,
        ),
        recall_at_k=recall_at_k(
            retrieved_chunk_ids,
            relevant,
            top_k,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_chunk_ids,
            relevant,
        ),
        ndcg_at_k=ndcg_at_k(
            retrieved_chunk_ids,
            relevant,
            top_k,
        ),
        hit_rate=hit_rate_at_k(
            retrieved_chunk_ids,
            relevant,
            top_k,
        ),
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def build_summary(
    *,
    results: list[EvaluationResult],
    corpus_chunks: int,
) -> EvaluationSummary:
    return EvaluationSummary(
        queries=len(results),
        corpus_chunks=corpus_chunks,
        top_k=TOP_K,
        precision_at_k=mean(
            [
                result.precision_at_k
                for result in results
            ]
        ),
        recall_at_k=mean(
            [
                result.recall_at_k
                for result in results
            ]
        ),
        mrr=mean(
            [
                result.reciprocal_rank
                for result in results
            ]
        ),
        ndcg_at_k=mean(
            [
                result.ndcg_at_k
                for result in results
            ]
        ),
        hit_rate_at_k=mean(
            [
                result.hit_rate
                for result in results
            ]
        ),
    )


def log_to_mlflow(
    summary: EvaluationSummary,
) -> None:
    tracking_uri = (
        "sqlite:///"
        f"{MLFLOW_DATABASE.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        experiment_id=(
            experiment.experiment_id
        ),
        run_name=(
            "hybrid-rag-real-evaluation"
        ),
    ):
        mlflow.log_params(
            {
                "retrieval_method": (
                    "hybrid-bm25-faiss"
                ),
                "top_k": summary.top_k,
                "dataset_size": (
                    summary.queries
                ),
                "corpus_size": (
                    summary.corpus_chunks
                ),
                "evaluation_type": (
                    "controlled-real-retrieval"
                ),
            }
        )

        mlflow.log_metrics(
            {
                "precision_at_k": (
                    summary.precision_at_k
                ),
                "recall_at_k": (
                    summary.recall_at_k
                ),
                "mrr": summary.mrr,
                "ndcg_at_k": (
                    summary.ndcg_at_k
                ),
                "hit_rate_at_k": (
                    summary.hit_rate_at_k
                ),
            }
        )


def print_human_summary(
    summary: EvaluationSummary,
) -> None:
    print()
    print(
        "Swathi AI RAG Evaluation"
    )
    print(
        "-------------------------"
    )
    print(
        f"Queries: {summary.queries}"
    )
    print(
        "Corpus chunks: "
        f"{summary.corpus_chunks}"
    )
    print(
        f"Top-K: {summary.top_k}"
    )
    print(
        "Precision@K: "
        f"{summary.precision_at_k:.4f}"
    )
    print(
        "Recall@K: "
        f"{summary.recall_at_k:.4f}"
    )
    print(
        f"MRR: {summary.mrr:.4f}"
    )
    print(
        "NDCG@K: "
        f"{summary.ndcg_at_k:.4f}"
    )
    print(
        "Hit Rate@K: "
        f"{summary.hit_rate_at_k:.4f}"
    )


def main() -> None:
    args = parse_args()

    examples = load_dataset(
        DATASET_PATH
    )

    if not examples:
        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    service = build_evaluation_service()

    results: list[EvaluationResult] = []

    for example in examples:
        retrieved = real_retrieval(
            service=service,
            query=example.query,
        )

        results.append(
            evaluate_query(
                query=example.query,
                retrieved_chunk_ids=retrieved,
                relevant_chunk_ids=(
                    example.relevant_chunk_ids
                ),
                top_k=TOP_K,
            )
        )

    summary = build_summary(
        results=results,
        corpus_chunks=len(
            service._chunks
        ),
    )

    if not args.no_mlflow:
        log_to_mlflow(summary)

    if args.json_output:
        print(
            json.dumps(
                asdict(summary),
                indent=2,
            )
        )
    else:
        print_human_summary(
            summary
        )

        if not args.no_mlflow:
            print()
            print(
                "Real retrieval results "
                "logged to MLflow."
            )


if __name__ == "__main__":
    main()