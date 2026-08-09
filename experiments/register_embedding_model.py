from __future__ import annotations

import json
from pathlib import Path

import mlflow

from swathi_ai.embeddings import DEFAULT_EMBEDDING_MODEL
from swathi_ai.model_registry import ModelRegistry


MODEL_NAME = "swathi-ai-embedding-model"
EXPERIMENT_NAME = "swathi-ai-model-registration"

MLFLOW_DATABASE = Path("mlflow.db")
ARTIFACT_DIRECTORY = Path("artifacts") / "embedding-model"


def create_model_metadata() -> Path:
    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        ARTIFACT_DIRECTORY
        / "embedding_model_metadata.json"
    )

    metadata = {
        "registered_model_name": MODEL_NAME,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "purpose": "Semantic document retrieval",
        "framework": "sentence-transformers",
        "vector_store": "FAISS",
        "retrieval_pipeline": "BM25 + FAISS hybrid search",
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata_path


def main() -> None:
    registry = ModelRegistry(
        database_path=MLFLOW_DATABASE,
    )

    mlflow.set_tracking_uri(
        registry.tracking_uri
    )

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    metadata_path = create_model_metadata()

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="register-embedding-model",
    ) as run:
        mlflow.log_params(
            {
                "embedding_model": (
                    DEFAULT_EMBEDDING_MODEL
                ),
                "framework": (
                    "sentence-transformers"
                ),
                "vector_store": "faiss",
                "retrieval_method": (
                    "hybrid-bm25-faiss"
                ),
            }
        )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="model",
        )

        run_id = run.info.run_id

    registered = registry.register_model_version(
        name=MODEL_NAME,
        run_id=run_id,
        source=f"runs:/{run_id}/model",
        description=(
            "Sentence Transformer embedding model "
            "used by Swathi AI hybrid RAG."
        ),
    )

    print(
        "Model registered successfully"
    )
    print(
        f"Name: {registered.name}"
    )
    print(
        f"Version: {registered.version}"
    )
    print(
        f"Status: {registered.status}"
    )
    print(
        f"Run ID: {registered.run_id}"
    )
    print(
        f"Tracking URI: {registry.tracking_uri}"
    )


if __name__ == "__main__":
    main()