from __future__ import annotations

import subprocess
import sys


MIN_RECALL_AT_K = 0.95
MIN_MRR = 0.80
MIN_NDCG_AT_K = 0.85
MIN_HIT_RATE_AT_K = 0.95


def main() -> None:
    command = [
        sys.executable,
        "experiments/evaluate_rag.py",
        "--json",
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
            "RAG evaluation could not complete."
        )

    print(result.stdout)


if __name__ == "__main__":
    main()