from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import speech_recognition as sr
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from swathi_ai.config import settings
from swathi_ai.services import get_engine, get_repository


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 90

WINDOWS_FFMPEG_FALLBACK = (
    r"C:\Users\gabis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.2-full_build\bin\ffmpeg.EXE"
)

FFMPEG_PATH = shutil.which("ffmpeg")

if not FFMPEG_PATH and Path(WINDOWS_FFMPEG_FALLBACK).exists():
    FFMPEG_PATH = WINDOWS_FFMPEG_FALLBACK


st.set_page_config(
    page_title=settings.app_name,
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# CSS — KEEP THIS BLOCK SEPARATE FROM THE HERO HTML
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --primary: #4f46e5;
        --secondary: #7c3aed;
        --background: #f8fafc;
        --surface: #ffffff;
        --text: #172033;
        --muted: #64748b;
        --border: #e2e8f0;
        --success: #15803d;
        --danger: #b91c1c;
        --warning: #a16207;
    }

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(124, 58, 237, 0.10),
                transparent 30rem
            ),
            radial-gradient(
                circle at bottom left,
                rgba(79, 70, 229, 0.08),
                transparent 28rem
            ),
            var(--background);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }

    .hero-card {
        padding: 2.4rem 2rem;
        border-radius: 26px;
        background:
            linear-gradient(
                135deg,
                #3730a3 0%,
                #6d28d9 52%,
                #9333ea 100%
            );
        box-shadow:
            0 22px 55px rgba(79, 70, 229, 0.24);
        text-align: center;
        color: white;
        margin: 0.4rem 0 2rem;
        overflow: visible;
    }

    .hero-logo {
        font-size: 2.9rem;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        color: white !important;
        font-size: clamp(2.2rem, 5vw, 3.3rem);
        line-height: 1.2;
        font-weight: 850;
        letter-spacing: -0.04em;
        margin: 0;
        padding: 0.15rem 0 0.25rem;
        overflow: visible;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.6rem;
        opacity: 0.95;
    }

    .hero-features {
        margin-top: 1rem;
        font-size: 0.91rem;
        opacity: 0.84;
        letter-spacing: 0.025em;
    }

    .welcome-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
    }

    .welcome-title {
        font-size: 1rem;
        font-weight: 750;
        color: var(--text);
        margin-bottom: 0.35rem;
    }

    .welcome-text {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.85rem 1.05rem;
        margin: 0.85rem 0;
        box-shadow: 0 7px 24px rgba(15, 23, 42, 0.055);
    }

    div[data-testid="stChatMessage"]:has(
        div[data-testid="chatAvatarIcon-user"]
    ) {
        background:
            linear-gradient(
                135deg,
                rgba(79, 70, 229, 0.09),
                rgba(124, 58, 237, 0.05)
            );
        border-color: rgba(79, 70, 229, 0.20);
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    div[data-testid="stSidebar"] > div:first-child {
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #f8fafc 100%
            );
    }

    div[data-testid="stSidebar"] button {
        border-radius: 12px;
        min-height: 2.65rem;
    }

    div[data-testid="stSidebar"] hr {
        margin: 1.15rem 0;
    }

    .sidebar-brand {
        padding: 0.4rem 0 0.85rem;
    }

    .sidebar-brand-title {
        font-size: 1.55rem;
        line-height: 1.25;
        font-weight: 850;
        color: var(--text);
        margin: 0;
    }

    .sidebar-brand-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.22rem;
    }

    .section-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 0.3rem 0 0.55rem;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        padding: 0.68rem 0.78rem;
        margin-bottom: 0.55rem;
        border: 1px solid var(--border);
        border-radius: 13px;
        background: white;
        font-size: 0.86rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
    }

    .status-online {
        color: var(--success);
        font-weight: 750;
    }

    .status-offline {
        color: var(--danger);
        font-weight: 750;
    }

    .status-warning {
        color: var(--warning);
        font-weight: 750;
    }

    .voice-help {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.6rem;
    }

    div[data-testid="stChatInput"] {
        background: rgba(248, 250, 252, 0.90);
        padding-bottom: 1rem;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 18px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 9px 26px rgba(15, 23, 42, 0.09);
    }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 2rem;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-top: 1.2rem;
        }

        .hero-card {
            padding: 1.8rem 1rem;
            border-radius: 20px;
        }

        .hero-title {
            font-size: 2.15rem;
        }

        .hero-subtitle {
            font-size: 0.98rem;
        }

        .hero-features {
            font-size: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SERVICES
# ---------------------------------------------------------------------

repo = get_repository()
engine = get_engine()


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def create_session_id() -> str:
    return datetime.now().strftime(
        "chat_%Y%m%d_%H%M%S_%f"
    )


def stream_response(
    text: str,
    delay: float = 0.004,
) -> None:
    placeholder = st.empty()
    displayed_text = ""

    for token in text.split(" "):
        displayed_text += token + " "
        placeholder.markdown(displayed_text)
        time.sleep(delay)

    placeholder.markdown(text)


# ---------------------------------------------------------------------
# FASTAPI HELPERS
# ---------------------------------------------------------------------

def check_backend_health() -> bool:
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )
        response.raise_for_status()
        return True

    except requests.RequestException:
        return False


def get_model_status() -> dict[str, Any] | None:
    try:
        response = requests.get(
            f"{API_BASE_URL}/model/status",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    except (requests.RequestException, ValueError):
        return None


def request_api_response(
    message: str,
    session_id: str,
    online: bool,
    response_format: str,
) -> dict[str, Any]:
    payload = {
        "message": message,
        "session_id": session_id,
        "online": online,
        "response_format": response_format,
    }

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    result = response.json()

    if "reply" not in result:
        raise ValueError(
            "The API response does not contain a reply."
        )

    return result


def local_fallback_response(
    message: str,
    online: bool,
    response_format: str,
) -> dict[str, Any]:
    history = repo.load(
        st.session_state.session_id
    )

    result = engine.respond(
        text=message,
        online=online,
        response_format=response_format,
        history=history,
    )

    return {
        "session_id": st.session_state.session_id,
        "reply": result.text,
        "source": f"local-{result.source}",
        "intent": result.intent,
        "confidence": result.confidence,
    }


# ---------------------------------------------------------------------
# VOICE HELPERS
# ---------------------------------------------------------------------

def get_recognition_language(
    response_format: str,
) -> str:
    language_map = {
        "Tamil only": "ta-IN",
        "Tanglish only": "en-IN",
        "English only": "en-GB",
        "All three": "en-IN",
        "Auto detect": "en-IN",
    }

    return language_map.get(
        response_format,
        "en-IN",
    )


def convert_audio_to_wav(
    audio_bytes: bytes,
) -> io.BytesIO:
    if not FFMPEG_PATH:
        raise FileNotFoundError(
            "FFmpeg executable was not found."
        )

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)

        input_path = temp_path / "voice_input.webm"
        output_path = temp_path / "voice_output.wav"

        input_path.write_bytes(audio_bytes)

        command = [
            FFMPEG_PATH,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "FFmpeg conversion failed: "
                f"{process.stderr.strip()}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "FFmpeg did not create the WAV output file."
            )

        wav_bytes = output_path.read_bytes()

    wav_buffer = io.BytesIO(wav_bytes)
    wav_buffer.seek(0)

    return wav_buffer


def transcribe_audio(
    audio_bytes: bytes,
    response_format: str,
) -> str | None:
    recognizer = sr.Recognizer()

    recognition_language = get_recognition_language(
        response_format
    )

    try:
        wav_buffer = convert_audio_to_wav(
            audio_bytes
        )

        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)

        recognised_text = recognizer.recognize_google(
            audio,
            language=recognition_language,
        )

        return recognised_text.strip()

    except sr.UnknownValueError:
        st.sidebar.warning(
            "The recording could not be understood. "
            "Please speak clearly and try again."
        )
        return None

    except sr.RequestError as error:
        st.sidebar.error(
            "Speech recognition is unavailable: "
            f"{error}"
        )
        return None

    except FileNotFoundError:
        st.sidebar.error(
            "FFmpeg could not be found by Python."
        )
        return None

    except RuntimeError as error:
        logger.exception(
            "Audio conversion failed: %s",
            error,
        )
        st.sidebar.error(str(error))
        return None

    except (
        ValueError,
        OSError,
        EOFError,
    ) as error:
        logger.exception(
            "Audio processing failed: %s",
            error,
        )
        st.sidebar.error(
            "The recorded audio could not be processed."
        )
        return None


def create_audio_hash(
    audio_bytes: bytes,
) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session_id()

if "chat" not in st.session_state:
    st.session_state.chat = repo.load(
        st.session_state.session_id
    )

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = {}

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None


# ---------------------------------------------------------------------
# BACKEND STATUS
# ---------------------------------------------------------------------

backend_online = check_backend_health()

model_status = (
    get_model_status()
    if backend_online
    else None
)


# ---------------------------------------------------------------------
# SIDEBAR BRAND
# ---------------------------------------------------------------------

sidebar_brand_html = (
    '<div class="sidebar-brand">'
    '<div class="sidebar-brand-title">&#127802; Swathi AI</div>'
    '<div class="sidebar-brand-subtitle">'
    'Enterprise Multilingual Assistant'
    '</div>'
    '</div>'
)

st.sidebar.markdown(
    sidebar_brand_html,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------------------

st.sidebar.markdown(
    '<div class="section-label">Assistant settings</div>',
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "AI Mode",
    [
        "Online AI",
        "Offline intent model",
    ],
)

response_format = st.sidebar.selectbox(
    "Response Format",
    [
        "Auto detect",
        "Tamil only",
        "Tanglish only",
        "English only",
        "All three",
    ],
)

show_history = st.sidebar.checkbox(
    "Show saved conversations",
)


# ---------------------------------------------------------------------
# VOICE INPUT
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="section-label">Voice input</div>',
    unsafe_allow_html=True,
)

voice_help_html = (
    '<div class="voice-help">'
    'Select the preferred response language before recording. '
    'Tamil speech works best with Tamil only.'
    '</div>'
)

st.sidebar.markdown(
    voice_help_html,
    unsafe_allow_html=True,
)

with st.sidebar:
    voice_recording = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        key="swathi_voice_recorder",
    )

if voice_recording:
    audio_bytes = voice_recording.get("bytes")

    if audio_bytes:
        current_audio_hash = create_audio_hash(
            audio_bytes
        )

        if (
            current_audio_hash
            != st.session_state.last_audio_hash
        ):
            st.session_state.last_audio_hash = (
                current_audio_hash
            )

            st.sidebar.audio(audio_bytes)

            with st.sidebar.spinner(
                "Transcribing voice..."
            ):
                transcribed_text = transcribe_audio(
                    audio_bytes=audio_bytes,
                    response_format=response_format,
                )

            if transcribed_text:
                st.session_state.last_transcription = (
                    transcribed_text
                )

                st.session_state.pending_prompt = (
                    transcribed_text
                )

                st.sidebar.success(
                    f"Recognised: {transcribed_text}"
                )

                st.rerun()


if st.session_state.last_transcription:
    st.sidebar.caption(
        "Last recognised voice"
    )

    st.sidebar.code(
        st.session_state.last_transcription,
        language=None,
    )

    if st.sidebar.button(
        "Clear voice text",
        use_container_width=True,
    ):
        st.session_state.last_transcription = None
        st.session_state.last_audio_hash = None
        st.session_state.pending_prompt = None
        st.rerun()


# ---------------------------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="section-label">System status</div>',
    unsafe_allow_html=True,
)

backend_class = (
    "status-online"
    if backend_online
    else "status-offline"
)

backend_text = (
    "Online"
    if backend_online
    else "Offline"
)

backend_status_html = (
    '<div class="status-row">'
    '<span>FastAPI backend</span>'
    f'<span class="{backend_class}">{backend_text}</span>'
    '</div>'
)

st.sidebar.markdown(
    backend_status_html,
    unsafe_allow_html=True,
)


if model_status:
    bert_available = model_status.get(
        "bert_model_available",
        False,
    )

    bert_class = (
        "status-online"
        if bert_available
        else "status-warning"
    )

    bert_text = (
        "Available"
        if bert_available
        else "Unavailable"
    )

    bert_status_html = (
        '<div class="status-row">'
        '<span>BERT intent model</span>'
        f'<span class="{bert_class}">{bert_text}</span>'
        '</div>'
    )

    st.sidebar.markdown(
        bert_status_html,
        unsafe_allow_html=True,
    )

    llm_configured = model_status.get(
        "online_llm_configured",
        False,
    )

    llm_class = (
        "status-online"
        if llm_configured
        else "status-warning"
    )

    llm_text = (
        "Configured"
        if llm_configured
        else "Not configured"
    )

    llm_status_html = (
        '<div class="status-row">'
        '<span>Online LLM</span>'
        f'<span class="{llm_class}">{llm_text}</span>'
        '</div>'
    )

    st.sidebar.markdown(
        llm_status_html,
        unsafe_allow_html=True,
    )


ffmpeg_available = bool(
    FFMPEG_PATH
    and Path(FFMPEG_PATH).exists()
)

ffmpeg_class = (
    "status-online"
    if ffmpeg_available
    else "status-warning"
)

ffmpeg_text = (
    "Ready"
    if ffmpeg_available
    else "Unavailable"
)

voice_status_html = (
    '<div class="status-row">'
    '<span>Voice engine</span>'
    f'<span class="{ffmpeg_class}">{ffmpeg_text}</span>'
    '</div>'
)

st.sidebar.markdown(
    voice_status_html,
    unsafe_allow_html=True,
)


with st.sidebar.expander(
    "Technical details"
):
    st.write(
        "API URL:",
        API_BASE_URL,
    )

    st.write(
        "LLM model:",
        (
            model_status.get("llm_model", "Unknown")
            if model_status
            else "Unknown"
        ),
    )

    st.write(
        "FFmpeg:",
        FFMPEG_PATH or "Not detected",
    )


# ---------------------------------------------------------------------
# CHAT MANAGEMENT
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="section-label">Conversation</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button(
    "New chat",
    use_container_width=True,
):
    st.session_state.session_id = create_session_id()
    st.session_state.chat = []
    st.session_state.pending_prompt = None
    st.session_state.last_metadata = {}
    st.session_state.last_audio_hash = None
    st.session_state.last_transcription = None
    st.rerun()

if st.sidebar.button(
    "Clear current chat",
    use_container_width=True,
):
    repo.delete(
        st.session_state.session_id
    )

    st.session_state.chat = []
    st.session_state.pending_prompt = None
    st.session_state.last_metadata = {}
    st.rerun()

if st.sidebar.button(
    "Clear all saved history",
    use_container_width=True,
):
    repo.clear()

    st.session_state.session_id = create_session_id()
    st.session_state.chat = []
    st.session_state.pending_prompt = None
    st.session_state.last_metadata = {}
    st.session_state.last_audio_hash = None
    st.session_state.last_transcription = None
    st.rerun()


# ---------------------------------------------------------------------
# EXAMPLE PROMPTS
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="section-label">Quick prompts</div>',
    unsafe_allow_html=True,
)

example_prompts = [
    "வணக்கம்",
    "epdi irukinga",
    "What is artificial intelligence?",
    "oru joke sollu",
    "Explain machine learning simply",
]

for index, prompt in enumerate(example_prompts):
    if st.sidebar.button(
        prompt,
        key=f"sample_{index}",
        use_container_width=True,
    ):
        st.session_state.pending_prompt = prompt
        st.rerun()


# ---------------------------------------------------------------------
# SAVED CONVERSATIONS
# ---------------------------------------------------------------------

if show_history:
    st.sidebar.divider()

    st.sidebar.markdown(
        '<div class="section-label">Saved conversations</div>',
        unsafe_allow_html=True,
    )

    sessions = repo.sessions()

    if sessions:
        for session_id, started_at, count in sessions:
            if st.sidebar.button(
                f"{session_id} ({count})",
                key=f"history_{session_id}",
                use_container_width=True,
            ):
                st.session_state.session_id = (
                    session_id
                )

                st.session_state.chat = (
                    repo.load(session_id)
                )

                st.session_state.pending_prompt = None
                st.session_state.last_metadata = {}

                st.rerun()

            st.sidebar.caption(
                str(started_at)
            )

    else:
        st.sidebar.caption(
            "No saved conversations yet."
        )


# ---------------------------------------------------------------------
# HERO SECTION — THIS IS OUTSIDE THE CSS BLOCK
# ---------------------------------------------------------------------

hero_html = (
    '<section class="hero-card">'
    '<div class="hero-logo">&#127802;</div>'
    f'<h1 class="hero-title">{settings.app_name}</h1>'
    '<div class="hero-subtitle">'
    'Enterprise Multilingual AI Assistant'
    '</div>'
    '<div class="hero-features">'
    'Tamil &bull; Tanglish &bull; English &bull; '
    'Voice &bull; Memory &bull; Gemini'
    '</div>'
    '</section>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


if not st.session_state.chat:
    welcome_html = (
        '<div class="welcome-card">'
        '<div class="welcome-title">Welcome to Swathi AI</div>'
        '<div class="welcome-text">'
        'Ask questions in Tamil, Tanglish, or English. '
        'Select a response format from the sidebar, type a message, '
        'or use the voice recorder.'
        '</div>'
        '</div>'
    )

    st.markdown(
        welcome_html,
        unsafe_allow_html=True,
    )


if mode == "Online AI":
    if not backend_online:
        st.warning(
            "FastAPI is unavailable. "
            "The local response engine will be used."
        )

    elif model_status and not model_status.get(
        "online_llm_configured",
        False,
    ):
        st.info(
            "The online LLM is not configured. "
            "Known intents will still work."
        )


# ---------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------------------

for role, message in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(message)


# ---------------------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------------------

user_input = st.chat_input(
    "Ask Swathi AI anything..."
)

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------
# PROCESS RESPONSE
# ---------------------------------------------------------------------

if user_input:
    st.session_state.chat.append(
        ("user", user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    online_enabled = (
        mode == "Online AI"
    )

    used_api = False

    try:
        if not backend_online:
            raise requests.ConnectionError(
                "FastAPI backend is unavailable."
            )

        api_result = request_api_response(
            message=user_input,
            session_id=(
                st.session_state.session_id
            ),
            online=online_enabled,
            response_format=response_format,
        )

        used_api = True

    except (
        requests.RequestException,
        ValueError,
        KeyError,
    ) as error:
        logger.warning(
            "FastAPI request failed. "
            "Using local engine: %s",
            error,
        )

        api_result = local_fallback_response(
            message=user_input,
            online=online_enabled,
            response_format=response_format,
        )

    reply = str(
        api_result["reply"]
    )

    source = str(
        api_result.get(
            "source",
            "unknown",
        )
    )

    intent = api_result.get(
        "intent"
    )

    confidence = api_result.get(
        "confidence"
    )

    with st.chat_message("assistant"):
        stream_response(reply)

        metadata_parts = [
            f"Source: {source}"
        ]

        if intent:
            metadata_parts.append(
                f"Intent: {intent}"
            )

        if confidence is not None:
            metadata_parts.append(
                "Confidence: "
                f"{float(confidence):.2f}"
            )

        st.caption(
            " | ".join(metadata_parts)
        )

    st.session_state.chat.append(
        ("assistant", reply)
    )

    st.session_state.last_metadata = {
        "source": source,
        "intent": intent,
        "confidence": confidence,
    }

    if not used_api:
        repo.save(
            st.session_state.session_id,
            "user",
            user_input,
        )

        repo.save(
            st.session_state.session_id,
            "assistant",
            reply,
        )


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

footer_html = (
    '<div class="footer-note">'
    'Swathi AI Enterprise Assistant '
    '&bull; FastAPI &bull; Gemini &bull; Streamlit'
    '</div>'
)

st.markdown(
    footer_html,
    unsafe_allow_html=True,
)