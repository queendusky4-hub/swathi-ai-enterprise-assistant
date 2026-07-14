from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import requests
import streamlit as st

from swathi_ai.config import settings
from swathi_ai.services import get_engine, get_repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 60

st.set_page_config(page_title=settings.app_name, page_icon="🌺", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 950px; padding-top: 1.4rem; padding-bottom: 2rem;}
    .main-title {text-align: center; font-size: 2.2rem; font-weight: 800;}
    .subtitle {text-align: center; color: #6b7280; margin-bottom: 1.3rem;}
    div[data-testid="stChatMessage"] {
        border: 1px solid #ececec;
        border-radius: 18px;
        padding: .55rem 1rem;
        margin: .7rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

repo = get_repository()
engine = get_engine()


def backend_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def model_status() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/model/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def api_chat(
    message: str,
    session_id: str,
    online: bool,
    show_all: bool,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "message": message,
            "session_id": session_id,
            "online": online,
            "show_all_formats": show_all,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def stream_text(text: str) -> None:
    placeholder = st.empty()
    shown = ""
    for token in text.split(" "):
        shown += token + " "
        placeholder.markdown(shown)
        time.sleep(0.004)
    placeholder.markdown(text)


if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime(
        "chat_%Y%m%d_%H%M%S_%f"
    )
if "chat" not in st.session_state:
    st.session_state.chat = repo.load(st.session_state.session_id)
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

backend_online = backend_health()
status = model_status() if backend_online else None

st.sidebar.title("🌺 Swathi AI")
mode = st.sidebar.radio("Mode", ["Online AI", "Offline intent model"])
show_all = st.sidebar.checkbox("Show Tamil + Tanglish + English", value=True)
show_history = st.sidebar.checkbox("Show saved conversations")

st.sidebar.divider()
st.sidebar.subheader("System status")
if backend_online:
    st.sidebar.success("FastAPI backend: Online")
else:
    st.sidebar.error("FastAPI backend: Offline")
    st.sidebar.caption("Local fallback will be used.")

if status:
    if status.get("bert_model_available"):
        st.sidebar.success("BERT intent model: Available")
    else:
        st.sidebar.warning("BERT intent model: Not available")

    if status.get("online_llm_configured"):
        st.sidebar.success("Online LLM: Configured")
    else:
        st.sidebar.info("Online LLM: Not configured")

if st.sidebar.button("New chat", use_container_width=True):
    st.session_state.session_id = datetime.now().strftime(
        "chat_%Y%m%d_%H%M%S_%f"
    )
    st.session_state.chat = []
    st.rerun()

if st.sidebar.button("Clear current chat", use_container_width=True):
    repo.delete(st.session_state.session_id)
    st.session_state.chat = []
    st.rerun()

if st.sidebar.button("Clear all saved history", use_container_width=True):
    repo.clear()
    st.session_state.session_id = datetime.now().strftime(
        "chat_%Y%m%d_%H%M%S_%f"
    )
    st.session_state.chat = []
    st.rerun()

st.sidebar.divider()
for index, prompt in enumerate(
    [
        "வணக்கம்",
        "epdi irukinga",
        "what is your name",
        "oru joke sollu",
        "Explain machine learning in simple Tamil",
    ]
):
    if st.sidebar.button(prompt, key=f"sample_{index}", use_container_width=True):
        st.session_state.pending_prompt = prompt
        st.rerun()

if show_history:
    st.sidebar.divider()
    st.sidebar.subheader("Saved conversations")
    for session_id, _, count in repo.sessions():
        if st.sidebar.button(
            f"{session_id} ({count})",
            key=f"history_{session_id}",
            use_container_width=True,
        ):
            st.session_state.session_id = session_id
            st.session_state.chat = repo.load(session_id)
            st.rerun()

st.markdown(
    f"<div class='main-title'>{settings.app_name}</div>"
    "<div class='subtitle'>Tamil • Tanglish • English</div>",
    unsafe_allow_html=True,
)

for role, message in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(message)

user_input = st.chat_input("Message Swathi AI…")
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    st.session_state.chat.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    used_api = False
    try:
        if not backend_online:
            raise requests.ConnectionError("Backend unavailable")
        result = api_chat(
            message=user_input,
            session_id=st.session_state.session_id,
            online=mode == "Online AI",
            show_all=show_all,
        )
        used_api = True
        reply = str(result["reply"])
        source = str(result.get("source", "api"))
        intent = result.get("intent")
        confidence = result.get("confidence")
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("API request failed; using local engine: %s", exc)
        local = engine.respond(
            user_input,
            online=mode == "Online AI",
            show_all=show_all,
        )
        reply = local.text
        source = f"local-{local.source}"
        intent = local.intent
        confidence = local.confidence

    with st.chat_message("assistant"):
        stream_text(reply)
        details = [f"Source: {source}"]
        if intent:
            details.append(f"Intent: {intent}")
        if confidence is not None:
            details.append(f"Confidence: {float(confidence):.2f}")
        st.caption(" • ".join(details))

    st.session_state.chat.append(("assistant", reply))

    if not used_api:
        repo.save(st.session_state.session_id, "user", user_input)
        repo.save(st.session_state.session_id, "assistant", reply)
