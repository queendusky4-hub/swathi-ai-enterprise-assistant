from __future__ import annotations

import logging
from typing import Any

import requests

from .config import Settings

logger = logging.getLogger(__name__)


LANGUAGE_INSTRUCTIONS = {
    "tamil": (
        "Reply only in natural Tamil script. "
        "Do not reply in English unless the user specifically requests English."
    ),
    "tanglish": (
        "Reply only in Tanglish: Tamil language written using English letters. "
        "Do not use Tamil script. Do not reply fully in English."
    ),
    "english": (
        "Reply only in clear English unless the user specifically requests "
        "Tamil or Tanglish."
    ),
}


SYSTEM_PROMPT = """
You are Swathi AI, a multilingual assistant supporting:

1. Tamil written in Tamil script
2. Tanglish, meaning Tamil written using English letters
3. English

Always follow the requested response language exactly.

Be helpful, friendly, accurate and concise.
Use conversation history when answering follow-up questions.
Remember relevant information shared earlier in the same conversation.
Never claim to have live information unless current information is provided.
""".strip()


class LLMClient:
    """Client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.llm_base_url
            and self.settings.llm_api_key
            and self.settings.llm_model
        )

    def generate(
    self,
    user_text: str,
    language: str = "english",
    response_format: str = "Auto detect",
    history: list[tuple[str, str]] | None = None,
) -> str | None:
        """Generate a response in Tamil, Tanglish or English."""

        if not self.configured:
            logger.warning("LLM is not configured.")
            return None

        url = (
            self.settings.llm_base_url.rstrip("/")
            + "/chat/completions"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": (
                f"Bearer {self.settings.llm_api_key}"
            ),
        }

        language_instruction = LANGUAGE_INSTRUCTIONS.get(
            language,
            LANGUAGE_INSTRUCTIONS["english"],
        )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Response-language requirement:\n"
                    f"{language_instruction}"
                ),
            }
        ]

        for role, message in (history or [])[-20:]:
            if role in {"user", "assistant"}:
                messages.append(
                    {
                        "role": role,
                        "content": message,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.25,
            "max_tokens": 600,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )

            if not response.ok:
                logger.error(
                    "LLM request failed: status=%s body=%s",
                    response.status_code,
                    response.text,
                )
                return None

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            if not isinstance(content, str):
                logger.error(
                    "LLM returned non-text content: %r",
                    content,
                )
                return None

            return content.strip()

        except requests.Timeout:
            logger.error(
                "LLM request timed out after %s seconds.",
                self.settings.llm_timeout_seconds,
            )
            return None

        except requests.RequestException as exc:
            logger.exception(
                "LLM connection failed: %s",
                exc,
            )
            return None

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            logger.exception(
                "Invalid LLM response: %s",
                exc,
            )
            return None