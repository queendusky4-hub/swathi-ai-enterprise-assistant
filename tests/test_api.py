from uuid import uuid4

from fastapi.testclient import TestClient

from swathi_ai.api import app


client = TestClient(app)


def authenticated_headers() -> dict[str, str]:
    """Create a temporary user and return a valid Bearer token."""
    username = f"pytest_{uuid4().hex[:12]}"
    password = "TestPassword123!"

    register_response = client.post(
        "/auth/register",
        json={
            "username": username,
            "password": password,
        },
    )
    assert register_response.status_code == 201, register_response.text

    login_response = client.post(
        "/auth/login",
        data={
            "username": username,
            "password": password,
        },
    )
    assert login_response.status_code == 200, login_response.text

    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["service"] == "swathi-ai-api"
    assert body["timestamp"]

def test_chat_known_intent() -> None:
    response = client.post(
        "/chat",
        headers=authenticated_headers(),
        json={
            "message": "வணக்கம்",
            "online": False,
            "response_format": "Auto detect",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "rule"
    assert body["intent"] == "greeting"
    assert body["reply"]
    assert body["session_id"]


def test_chat_rejects_empty_message() -> None:
    response = client.post(
        "/chat",
        headers=authenticated_headers(),
        json={
            "message": "",
            "online": False,
            "response_format": "Auto detect",
        },
    )

    assert response.status_code == 422