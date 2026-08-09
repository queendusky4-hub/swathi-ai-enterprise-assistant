from __future__ import annotations

import json
import subprocess
import sys


MIN_RECALL_AT_K = 0.95
MIN_MRR = 0.80
MIN_NDCG_AT_K = 0.85
MIN_HIT_RATE_AT_K = 0.95


def run_evaluation() -> dict[str, float | int]:
    command = [
        sys.executable,
        "experiments/evaluate_rag.py",
        "--json",
        "--no-mlflow",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(
            "RAG evaluation failed to execute."
        )

    try:
        return json.loads(
            result.stdout
        )
    except json.JSONDecodeError as exc:
        print(result.stdout)
        raise SystemExit(
            f"Could not parse evaluation JSON: {exc}"
        ) from exc


def validate_metrics(
    metrics: dict[str, float | int],
) -> list[str]:
    failures: list[str] = []

    recall = float(
        metrics.get(
            "recall_at_k",
            0.0,
        )
    )

    mrr = float(
        metrics.get(
            "mrr",
            0.0,
        )
    )

    ndcg = float(
        metrics.get(
            "ndcg_at_k",
            0.0,
        )
    )

    hit_rate = float(
        metrics.get(
            "hit_rate_at_k",
            0.0,
        )
    )

    if recall < MIN_RECALL_AT_K:
        failures.append(
            f"Recall@K {recall:.4f} "
            f"is below {MIN_RECALL_AT_K:.4f}"
        )

    if mrr < MIN_MRR:
        failures.append(
            f"MRR {mrr:.4f} "
            f"is below {MIN_MRR:.4f}"
        )

    if ndcg < MIN_NDCG_AT_K:
        failures.append(
            f"NDCG@K {ndcg:.4f} "
            f"is below {MIN_NDCG_AT_K:.4f}"
        )

    if hit_rate < MIN_HIT_RATE_AT_K:
        failures.append(
            f"Hit Rate@K {hit_rate:.4f} "
            f"is below {MIN_HIT_RATE_AT_K:.4f}"
        )

    return failures


def main() -> None:
    metrics = run_evaluation()

    print()
    print("Swathi AI RAG Quality Gate")
    print("---------------------------")
    print(
        f"Recall@K: "
        f"{float(metrics['recall_at_k']):.4f}"
    )
    print(
        f"MRR: "
        f"{float(metrics['mrr']):.4f}"
    )
    print(
        f"NDCG@K: "
        f"{float(metrics['ndcg_at_k']):.4f}"
    )
    print(
        f"Hit Rate@K: "
        f"{float(metrics['hit_rate_at_k']):.4f}"
    )

    failures = validate_metrics(
        metrics
    )

    if failures:
        print()
        print("QUALITY GATE FAILED")

        for failure in failures:
            print(
                f"- {failure}"
            )

        raise SystemExit(1)

    print()
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()