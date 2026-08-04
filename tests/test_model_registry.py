from pathlib import Path

import mlflow

from swathi_ai.model_registry import ModelRegistry


def test_create_registered_model(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(
        database_path=tmp_path / "mlflow.db",
    )

    registry.create_registered_model(
        name="swathi-embedding-model",
        description="Test embedding model",
    )

    models = registry.client.search_registered_models()

    assert any(
        model.name == "swathi-embedding-model"
        for model in models
    )


def test_register_and_list_model_version(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"

    registry = ModelRegistry(
        database_path=database_path,
    )

    mlflow.set_tracking_uri(
        registry.tracking_uri
    )

    experiment = mlflow.set_experiment(
        "registry-test"
    )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id
    ) as run:
        artifact_directory = (
            tmp_path / "model-artifact"
        )
        artifact_directory.mkdir()

        artifact_file = (
            artifact_directory / "model.txt"
        )
        artifact_file.write_text(
            "test model",
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(artifact_file),
            artifact_path="model",
        )

        run_id = run.info.run_id

    source = f"runs:/{run_id}/model"

    registered = registry.register_model_version(
        name="swathi-embedding-model",
        run_id=run_id,
        source=source,
        description="Test version",
    )

    assert registered.name == "swathi-embedding-model"
    assert registered.version == "1"

    versions = registry.list_versions(
        "swathi-embedding-model"
    )

    assert len(versions) == 1
    assert versions[0].run_id == run_id


def test_get_latest_version(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(
        database_path=tmp_path / "mlflow.db",
    )

    assert (
        registry.get_latest_version(
            "missing-model"
        )
        is None
    )