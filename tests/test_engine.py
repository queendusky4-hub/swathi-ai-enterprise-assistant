from pathlib import Path

from swathi_ai.classifier import IntentClassifier
from swathi_ai.config import Settings
from swathi_ai.engine import ChatEngine
from swathi_ai.llm import LLMClient


def make_engine(tmp_path: Path) -> ChatEngine:
    settings = Settings(
        database_path=tmp_path / "chat.db",
        model_path=tmp_path / "missing",
    )
    return ChatEngine(
        IntentClassifier(settings.model_path),
        LLMClient(settings),
    )


def test_rule_response(tmp_path: Path) -> None:
    result = make_engine(tmp_path).respond(
        "vanakkam",
        online=False,
        response_format="Auto detect",
    )

    assert result.source == "rule"
    assert result.intent == "greeting"


def test_offline_fallback(tmp_path: Path) -> None:
    result = make_engine(tmp_path).respond(
        "quantum banana unknown",
        online=False,
        response_format="Auto detect",
    )

    assert result.source == "offline"
    assert result.text