from __future__ import annotations

import logging
from typing import Any

import requests

from .config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Swathi AI, a helpful multilingual assistant.

Reply in the same language as the user:
- Tamil when the user writes in Tamil
- Tanglish when the user writes in romanised Tamil
- English when the user writes in English

Be clear, concise, friendly, and accurate.
Do not claim to have live information unless current data is supplied.
""".strip()


class LLMClient:
    """Client for an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        """Return True when the required LLM settings exist."""
        return bool(
            self.settings.llm_base_url
            and self.settings.llm_api_key
            and self.settings.llm_model
        )

    def generate(self, user_text: str) -> str | None:
        """Generate an answer using the configured online LLM."""

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

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            "temperature": 0.3,
            "max_tokens": 350,
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
                "Invalid LLM response structure: %s",
                exc,
            )
            return None