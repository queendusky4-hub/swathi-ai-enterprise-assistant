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
    "http://127.0.0.1:8002",
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


def _response_message(response: requests.Response, default: str) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip() or default

    if isinstance(data, dict):
        detail = data.get("detail") or data.get("message")
        if isinstance(detail, list):
            return "; ".join(
                str(item.get("msg", item))
                if isinstance(item, dict)
                else str(item)
                for item in detail
            )
        if detail:
            return str(detail)

    return default


def register_user(
    username: str,
    password: str,
) -> tuple[bool, str]:
    clean_username = username.strip().lower()

    if not clean_username:
        return False, "Enter a username."

    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={
                "username": clean_username,
                "password": password,
            },
            timeout=15,
        )

        if response.status_code in (200, 201):
            return True, "Account created successfully. Please log in."

        return False, _response_message(
            response,
            "Registration failed.",
        )

    except requests.RequestException as error:
        return False, f"Could not connect to FastAPI: {error}"


def load_current_user() -> tuple[bool, str]:
    token = st.session_state.get("access_token")

    if not token:
        return False, "No access token was found."

    try:
        response = requests.get(
            f"{API_BASE_URL}/users/me",
            headers={
                "Authorization": f"Bearer {token}",
            },
            timeout=15,
        )

        if response.status_code == 200:
            profile = response.json()

            if not isinstance(profile, dict):
                return False, "The profile response was invalid."

            st.session_state.current_user = profile
            return True, "Profile loaded successfully."

        return False, _response_message(
            response,
            "User profile could not be loaded.",
        )

    except requests.RequestException as error:
        return False, f"Could not connect to FastAPI: {error}"

    except ValueError as error:
        return False, f"Invalid profile response: {error}"


def login_user(
    username: str,
    password: str,
) -> tuple[bool, str]:
    clean_username = username.strip().lower()

    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={
                "username": clean_username,
                "password": password,
            },
            timeout=15,
        )

        if response.status_code != 200:
            return False, _response_message(
                response,
                "Invalid username or password.",
            )

        data = response.json()
        token = data.get("access_token")

        if not token:
            return False, "The login response did not contain an access token."

        st.session_state.access_token = token

        profile_loaded, profile_message = load_current_user()

        if profile_loaded:
            return True, "Login successful."

        st.session_state.current_user = {
            "id": None,
            "username": clean_username,
            "role": "user",
        }

        logger.warning(
            "Login succeeded but /users/me was unavailable: %s",
            profile_message,
        )

        return True, "Login successful."

    except (requests.RequestException, ValueError) as error:
        return False, f"Could not connect to FastAPI: {error}"


def logout_user() -> None:
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.chat = []
    st.session_state.document_chat = []
    st.session_state.pending_prompt = None
    st.rerun()


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

    token = st.session_state.get("access_token")
    headers = (
        {"Authorization": f"Bearer {token}"}
        if token
        else {}
    )

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=payload,
        headers=headers,
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
# DOCUMENT AND IMAGE HELPERS
# ---------------------------------------------------------------------

DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def _auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return (
        {"Authorization": f"Bearer {token}"}
        if token
        else {}
    )


def upload_document_to_api(uploaded_file: Any) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files=files,
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def ask_document_api(
    question: str,
    online: bool,
    top_k: int = 5,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/documents/ask",
        json={
            "question": question,
            "top_k": top_k,
            "online": online,
            "response_format": st.session_state.get(
                "selected_response_format",
                "Auto detect",
            ),
        },
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def get_indexed_documents() -> list[dict[str, Any]]:
    # The current FastAPI backend exposes upload/search/ask routes,
    # but it does not expose GET /documents.
    # Keep the uploaded-document status in Streamlit session state instead.
    return list(st.session_state.get("indexed_documents", []))


def delete_indexed_document(document_id: str) -> None:
    # The current backend does not expose DELETE /documents/{id}.
    # Remove the item from the local Streamlit list only.
    st.session_state.indexed_documents = [
        item
        for item in st.session_state.get("indexed_documents", [])
        if str(item.get("document_id")) != str(document_id)
    ]


def extract_http_error_detail(error: requests.HTTPError) -> str:
    try:
        payload = error.response.json()
        detail = payload.get("detail")

        if isinstance(detail, list):
            return "; ".join(
                str(item.get("msg", item))
                if isinstance(item, dict)
                else str(item)
                for item in detail
            )

        if detail:
            return str(detail)

    except (ValueError, AttributeError):
        pass

    return str(error)


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

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "auth_page" not in st.session_state:
    st.session_state.auth_page = "Login"


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


if "workspace_mode" not in st.session_state:
    st.session_state.workspace_mode = "General Assistant"

if "document_chat" not in st.session_state:
    st.session_state.document_chat = []

if "processed_file_signatures" not in st.session_state:
    st.session_state.processed_file_signatures = set()

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []


if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []


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
# AUTHENTICATION GATE
# ---------------------------------------------------------------------

if st.session_state.access_token and not st.session_state.current_user:
    profile_loaded, _profile_message = load_current_user()

    if not profile_loaded:
        st.session_state.current_user = {
            "id": None,
            "username": "user",
            "role": "user",
        }

if not st.session_state.access_token:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-logo">&#127802;</div>
            <h1 class="hero-title">Swathi AI</h1>
            <div class="hero-subtitle">Secure Enterprise Assistant</div>
            <div class="hero-features">JWT Authentication &bull; FastAPI &bull; Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.error(
            "FastAPI is unavailable. Start the backend on "
            f"{API_BASE_URL} before logging in."
        )
        st.stop()

    login_tab, register_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            login_username = st.text_input(
                "Username",
                placeholder="Enter your username",
            )
            login_password = st.text_input(
                "Password",
                type="password",
            )
            login_submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_submitted:
            if not login_username.strip() or not login_password:
                st.warning("Enter your username and password.")
            else:
                success, message = login_user(
                    username=login_username,
                    password=login_password,
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            register_username = st.text_input(
                "Username",
                placeholder="Choose a username",
            )
            register_password = st.text_input(
                "Create password",
                type="password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )
            register_submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )

        if register_submitted:
            if not register_username.strip() or not register_password or not confirm_password:
                st.warning("Complete every registration field.")
            elif register_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(register_password) < 8:
                st.error("Password must contain at least 8 characters.")
            else:
                success, message = register_user(
                    username=register_username,
                    password=register_password,
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()


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

current_user = st.session_state.current_user or {}
user_name = (
    current_user.get("username")
    or current_user.get("name")
    or current_user.get("full_name")
    or current_user.get("email")
    or "User"
)
st.sidebar.success(f"Signed in as {user_name}")

if st.sidebar.button(
    "Logout",
    use_container_width=True,
    key="logout_button",
):
    logout_user()


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


st.session_state.selected_response_format = response_format

workspace_mode = st.sidebar.radio(
    "Workspace",
    [
        "General Assistant",
        "Document Chat",
    ],
    index=(
        1
        if st.session_state.workspace_mode == "Document Chat"
        else 0
    ),
)

st.session_state.workspace_mode = workspace_mode

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
# FILES AND PHOTOS
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="section-label">Files &amp; photos</div>',
    unsafe_allow_html=True,
)

st.sidebar.caption(
    "Upload PDF, DOCX, TXT, PNG, JPG, JPEG, or WEBP files."
)

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=[
        "pdf",
        "docx",
        "txt",
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
    accept_multiple_files=True,
    key="files_and_photos_uploader",
)

if uploaded_files:
    document_files = []
    image_files = []

    for uploaded_file in uploaded_files:
        extension = Path(uploaded_file.name).suffix.lower().lstrip(".")

        if extension in DOCUMENT_EXTENSIONS:
            document_files.append(uploaded_file)
        elif extension in IMAGE_EXTENSIONS:
            image_files.append(uploaded_file)

    if image_files:
        st.sidebar.caption("Photo preview")

        for image_file in image_files:
            image_signature = (
                f"{image_file.name}:{image_file.size}:"
                f"{hashlib.sha256(image_file.getvalue()).hexdigest()}"
            )

            if not any(
                item["signature"] == image_signature
                for item in st.session_state.uploaded_images
            ):
                st.session_state.uploaded_images.append(
                    {
                        "name": image_file.name,
                        "bytes": image_file.getvalue(),
                        "signature": image_signature,
                    }
                )

            st.sidebar.image(
                image_file.getvalue(),
                caption=image_file.name,
                use_container_width=True,
            )

        st.sidebar.info(
            "Photos are attached and previewed. "
            "Image analysis needs a vision-enabled backend endpoint."
        )

    if document_files:
        if st.sidebar.button(
            "Process uploaded documents",
            type="primary",
            use_container_width=True,
            key="process_uploaded_documents",
        ):
            if not backend_online:
                st.sidebar.error(
                    "Start the FastAPI backend before uploading documents."
                )
            else:
                successful_uploads = 0

                for document_file in document_files:
                    signature = (
                        f"{document_file.name}:{document_file.size}:"
                        f"{hashlib.sha256(document_file.getvalue()).hexdigest()}"
                    )

                    if signature in st.session_state.processed_file_signatures:
                        continue

                    try:
                        with st.sidebar.spinner(
                            f"Processing {document_file.name}..."
                        ):
                            result = upload_document_to_api(document_file)

                        st.session_state.processed_file_signatures.add(
                            signature
                        )
                        successful_uploads += 1

                        document_record = {
                            "document_id": result.get("document_id"),
                            "filename": result.get(
                                "filename",
                                document_file.name,
                            ),
                            "chunk_count": result.get("chunk_count", 0),
                        }

                        existing_ids = {
                            str(item.get("document_id"))
                            for item in st.session_state.indexed_documents
                        }

                        if str(document_record["document_id"]) not in existing_ids:
                            st.session_state.indexed_documents.append(
                                document_record
                            )

                        chunk_count = document_record["chunk_count"]
                        st.sidebar.success(
                            f"{document_file.name}: "
                            f"{chunk_count} chunks indexed."
                        )

                    except requests.HTTPError as error:
                        st.sidebar.error(
                            f"{document_file.name}: "
                            f"{extract_http_error_detail(error)}"
                        )

                    except requests.RequestException as error:
                        st.sidebar.error(
                            f"{document_file.name}: upload failed: {error}"
                        )

                if successful_uploads:
                    st.session_state.workspace_mode = "Document Chat"
                    st.rerun()

if workspace_mode == "Document Chat":
    indexed_documents = get_indexed_documents()

    if indexed_documents:
        with st.sidebar.expander(
            f"Uploaded documents ({len(indexed_documents)})"
        ):
            for index, document in enumerate(indexed_documents):
                filename = str(
                    document.get("filename", "Unnamed document")
                )
                document_id = document.get(
                    "document_id",
                    document.get("id"),
                )
                chunk_count = document.get("chunk_count")

                label = filename
                if chunk_count is not None:
                    label += f" · {chunk_count} chunks"

                st.write(label)

                if document_id and st.button(
                    "Remove from list",
                    key=f"delete_document_{document_id}_{index}",
                    use_container_width=True,
                ):
                    delete_indexed_document(str(document_id))
                    st.rerun()
    else:
        st.sidebar.caption(
            "No documents processed in this session yet."
        )


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
    st.session_state.document_chat = []
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
    st.session_state.document_chat = []
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
    st.session_state.document_chat = []
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

active_chat = (
    st.session_state.document_chat
    if workspace_mode == "Document Chat"
    else st.session_state.chat
)

for chat_item in active_chat:
    if len(chat_item) == 2:
        role, message = chat_item
        metadata: dict[str, Any] = {}
    else:
        role, message, metadata = chat_item

    with st.chat_message(role):
        st.markdown(message)

        if (
            workspace_mode == "Document Chat"
            and role == "assistant"
            and metadata
        ):
            citations = metadata.get("citations", [])

            if citations:
                with st.expander("View document sources"):
                    for citation in citations:
                        number = citation.get("citation_number", "")
                        filename = citation.get(
                            "filename",
                            "Unknown document",
                        )
                        page = citation.get("page_number")
                        score = citation.get("score")
                        preview = citation.get("preview") or citation.get("text", "")

                        title = f"[{number}] {filename}" if number else filename
                        st.markdown(f"**{title}**")

                        caption_parts = []
                        if page is not None:
                            caption_parts.append(f"Page {page}")
                        if score is not None:
                            caption_parts.append(
                                f"Similarity: {float(score):.3f}"
                            )
                        if caption_parts:
                            st.caption(" | ".join(caption_parts))
                        if preview:
                            st.write(preview)


# ---------------------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------------------

input_placeholder = (
    "Ask Swathi AI anything..."
    if workspace_mode == "General Assistant"
    else "Ask a question about your uploaded documents..."
)

user_input = st.chat_input(input_placeholder)

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------
# PROCESS RESPONSE
# ---------------------------------------------------------------------

if user_input:
    online_enabled = mode == "Online AI"

    if workspace_mode == "Document Chat":
        st.session_state.document_chat.append(
            ("user", user_input, {})
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        if not backend_online:
            with st.chat_message("assistant"):
                st.error(
                    "Document Chat requires the FastAPI backend."
                )
        else:
            try:
                with st.chat_message("assistant"):
                    with st.spinner(
                        "Searching your documents..."
                    ):
                        result = ask_document_api(
                            question=user_input,
                            online=online_enabled,
                            top_k=5,
                        )

                    answer = str(
                        result.get(
                            "answer",
                            result.get(
                                "reply",
                                "No answer was returned.",
                            ),
                        )
                    )
                    citations = result.get("citations") or result.get("sources", [])
                    source = str(
                        result.get("source", "document-rag")
                    )
                    retrieved_chunks = int(
                        result.get(
                            "retrieved_chunks",
                            len(citations),
                        )
                    )

                    stream_response(answer)
                    st.caption(
                        f"Source: {source} | "
                        f"Retrieved chunks: {retrieved_chunks}"
                    )

                    if citations:
                        with st.expander("View document sources"):
                            for citation in citations:
                                number = citation.get(
                                    "citation_number",
                                    "",
                                )
                                filename = citation.get(
                                    "filename",
                                    "Unknown document",
                                )
                                page = citation.get("page_number")
                                score = citation.get("score")
                                preview = citation.get("preview") or citation.get("text", "")

                                title = (
                                    f"[{number}] {filename}"
                                    if number
                                    else filename
                                )
                                st.markdown(f"**{title}**")

                                caption_parts = []
                                if page is not None:
                                    caption_parts.append(
                                        f"Page {page}"
                                    )
                                if score is not None:
                                    caption_parts.append(
                                        "Similarity: "
                                        f"{float(score):.3f}"
                                    )
                                if caption_parts:
                                    st.caption(
                                        " | ".join(caption_parts)
                                    )
                                if preview:
                                    st.write(preview)
                                st.divider()

                st.session_state.document_chat.append(
                    (
                        "assistant",
                        answer,
                        {
                            "source": source,
                            "retrieved_chunks": retrieved_chunks,
                            "citations": citations,
                        },
                    )
                )

            except requests.HTTPError as error:
                with st.chat_message("assistant"):
                    st.error(extract_http_error_detail(error))

            except requests.RequestException as error:
                with st.chat_message("assistant"):
                    st.error(
                        f"Document Chat request failed: {error}"
                    )

    else:
        st.session_state.chat.append(
            ("user", user_input)
        )

        with st.chat_message("user"):
            st.markdown(user_input)

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
    '&bull; FastAPI &bull; Gemini &bull; Streamlit &bull; Document RAG'
    '</div>'
)

st.markdown(
    footer_html,
    unsafe_allow_html=True,
)