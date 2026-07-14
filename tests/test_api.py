from fastapi.testclient import TestClient

from swathi_ai.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_known_intent() -> None:
    response = client.post(
        "/chat",
        json={
            "message": "வணக்கம்",
            "online": False,
            "show_all_formats": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "rule"
    assert body["intent"] == "greeting"
    assert body["reply"]
    assert body["session_id"].startswith("api_")


def test_chat_rejects_empty_message() -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
