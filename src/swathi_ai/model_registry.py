from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient


@dataclass(frozen=True)
class RegisteredModelVersion:
    name: str
    version: str
    status: str
    run_id: str
    source: str
    description: str | None = None


class ModelRegistry:
    """
    Small MLflow Model Registry wrapper.

    This module is completely separate from the live FastAPI and
    Streamlit runtime until explicitly connected later.
    """

    def __init__(
        self,
        database_path: Path | None = None,
    ) -> None:
        if database_path is None:
            database_path = Path("mlflow.db")

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tracking_uri = (
            f"sqlite:///{database_path.resolve().as_posix()}"
        )

        mlflow.set_tracking_uri(tracking_uri)

        self.tracking_uri = tracking_uri
        self.client = MlflowClient(
            tracking_uri=tracking_uri,
        )

    def create_registered_model(
        self,
        name: str,
        description: str | None = None,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError(
                "Model name cannot be empty."
            )

        existing_models = {
            model.name
            for model in self.client.search_registered_models()
        }

        if clean_name not in existing_models:
            self.client.create_registered_model(
                name=clean_name,
                description=description,
            )

    def register_model_version(
        self,
        *,
        name: str,
        run_id: str,
        source: str,
        description: str | None = None,
    ) -> RegisteredModelVersion:
        clean_name = name.strip()
        clean_run_id = run_id.strip()
        clean_source = source.strip()

        if not clean_name:
            raise ValueError(
                "Model name cannot be empty."
            )

        if not clean_run_id:
            raise ValueError(
                "run_id cannot be empty."
            )

        if not clean_source:
            raise ValueError(
                "source cannot be empty."
            )

        self.create_registered_model(
            name=clean_name,
            description=description,
        )

        version = self.client.create_model_version(
            name=clean_name,
            source=clean_source,
            run_id=clean_run_id,
            description=description,
        )

        return RegisteredModelVersion(
            name=clean_name,
            version=str(version.version),
            status=str(version.status),
            run_id=clean_run_id,
            source=clean_source,
            description=description,
        )

    def list_versions(
        self,
        name: str,
    ) -> list[RegisteredModelVersion]:
        clean_name = name.strip()

        if not clean_name:
            return []

        versions = self.client.search_model_versions(
            f"name='{clean_name}'"
        )

        return [
            RegisteredModelVersion(
                name=str(version.name),
                version=str(version.version),
                status=str(version.status),
                run_id=str(version.run_id or ""),
                source=str(version.source or ""),
                description=version.description,
            )
            for version in versions
        ]

    def get_latest_version(
        self,
        name: str,
    ) -> RegisteredModelVersion | None:
        versions = self.list_versions(name)

        if not versions:
            return None

        return max(
            versions,
            key=lambda item: int(item.version),
        )