from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Swathi AI")
    database_path: Path = resolve_project_path(
        os.getenv("DATABASE_PATH", "data/chat_history.db")
    )

    auth_database_url: str = os.getenv(
        "AUTH_DATABASE_URL",
        "",
    ).strip()


    model_path: Path = resolve_project_path(os.getenv("MODEL_PATH", "model"))
    confidence_threshold: float = float(
        os.getenv("CONFIDENCE_THRESHOLD", "0.55")
    )
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").strip()
    llm_api_key: str = os.getenv("LLM_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    llm_timeout_seconds: int = int(
        os.getenv("LLM_TIMEOUT_SECONDS", "60")
    )


settings = Settings()
