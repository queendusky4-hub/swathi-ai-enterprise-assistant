from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

from .config import Settings

logger = logging.getLogger(__name__)

LANGUAGE_INSTRUCTIONS = {
    "tamil": "Reply only in natural Tamil script. Do not reply in English unless explicitly requested.",
    "tanglish": "Reply only in Tanglish. Do not use Tamil script.",
    "english": "Reply only in clear English.",
}

SYSTEM_PROMPT = """
You are Swathi AI, an intelligent multilingual enterprise assistant.

Rules:
- Answer only the user's current question.
- Use conversation history only when relevant.
- Never invent facts or greetings.
- Be accurate, professional, clear, and friendly.
- Give concise answers for direct factual questions.
- When document context is supplied, answer only from that context.
- If the answer is absent from the context, say so clearly.
- Preserve citation markers such as [1], [2], and [3].
""".strip()


class LLMClient:
    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_error: str | None = None
        self.last_status_code: int | None = None
        self.last_response_body: str | None = None

    @property
    def configured(self) -> bool:
        return bool(
            self._value("llm_base_url")
            and self._value("llm_api_key")
            and self._value("llm_model")
        )

    def configuration_status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "base_url_present": bool(self._value("llm_base_url")),
            "api_key_present": bool(self._value("llm_api_key")),
            "model_present": bool(self._value("llm_model")),
            "base_url": self._value("llm_base_url") or None,
            "model": self._value("llm_model") or None,
            "timeout_seconds": self._timeout(),
            "max_retries": self._max_retries(),
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
        }

    def generate(
        self,
        user_text: str,
        language: str = "english",
        response_format: str = "Auto detect",
        history: list[tuple[str, str]] | None = None,
    ) -> str | None:
        self.last_error = None
        self.last_status_code = None
        self.last_response_body = None

        prompt = str(user_text or "").strip()
        if not prompt:
            return self._fail("The user prompt is empty.")

        if not self.configured:
            missing = [
                name
                for name in ("llm_base_url", "llm_api_key", "llm_model")
                if not self._value(name)
            ]
            return self._fail(
                "LLM is not configured. Missing settings: " + ", ".join(missing)
            )

        url = self._chat_url()
        payload = {
            "model": self._value("llm_model"),
            "messages": self._messages(
                prompt,
                language,
                response_format,
                history,
            ),
            "max_tokens": self._max_tokens(),
        }

        temperature = getattr(self.settings, "llm_temperature", None)
        if temperature is not None:
            try:
                payload["temperature"] = float(temperature)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid llm_temperature: %r", temperature)

        headers = {
            "Authorization": f"Bearer {self._value('llm_api_key')}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        attempts = self._max_retries() + 1

        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout(),
                )
                self.last_status_code = response.status_code
                self.last_response_body = response.text[:4000]

                if response.ok:
                    return self._parse_response(response)

                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < attempts
                ):
                    delay = self._retry_delay(attempt, response)
                    logger.warning(
                        "Transient LLM error %s. Retrying in %.2f seconds.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                return self._fail(self._http_error(response))

            except requests.Timeout:
                if attempt < attempts:
                    delay = self._retry_delay(attempt, None)
                    logger.warning(
                        "LLM request timed out. Retrying in %.2f seconds.",
                        delay,
                    )
                    time.sleep(delay)
                    continue
                return self._fail(
                    f"LLM request timed out after {self._timeout()} seconds."
                )

            except requests.ConnectionError as exc:
                if attempt < attempts:
                    delay = self._retry_delay(attempt, None)
                    logger.warning(
                        "LLM connection error. Retrying in %.2f seconds.",
                        delay,
                    )
                    time.sleep(delay)
                    continue
                return self._fail(
                    f"Unable to connect to the LLM endpoint: {exc}"
                )

            except requests.RequestException as exc:
                return self._fail(f"LLM request error: {exc}")

            except Exception as exc:
                logger.exception("Unexpected LLM error")
                return self._fail(
                    f"Unexpected LLM error: {type(exc).__name__}: {exc}"
                )

        return self._fail("The LLM request failed without a usable response.")

    def _messages(
        self,
        prompt: str,
        language: str,
        response_format: str,
        history: list[tuple[str, str]] | None,
    ) -> list[dict[str, str]]:
        selected_language = str(language or "english").strip().lower()
        instruction = LANGUAGE_INSTRUCTIONS.get(
            selected_language,
            LANGUAGE_INSTRUCTIONS["english"],
        )

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\n"
                    + instruction
                    + "\n\n"
                    + self._format_instruction(response_format)
                ),
            }
        ]

        if history:
            for item in history[-20:]:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                role, message = item
                role = str(role or "").strip().lower()
                message = str(message or "").strip()
                if role in {"user", "assistant"} and message:
                    messages.append({"role": role, "content": message})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_response(self, response: requests.Response) -> str | None:
        try:
            data = response.json()
        except ValueError:
            return self._fail(
                "LLM returned a non-JSON response: " + response.text[:1000]
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return self._fail(
                "LLM response did not contain choices. "
                f"Response: {str(data)[:2000]}"
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            return self._fail("The first LLM choice has an invalid format.")

        message = choice.get("message")
        if not isinstance(message, dict):
            return self._fail("LLM response did not contain a valid message.")

        content = self._extract_content(message.get("content"))
        if not content:
            return self._fail(
                "LLM returned an empty response. "
                f"finish_reason={choice.get('finish_reason')!r}"
            )

        return content

    def _chat_url(self) -> str:
        base = self._value("llm_base_url").rstrip("/")
        return (
            base
            if base.endswith("/chat/completions")
            else f"{base}/chat/completions"
        )

    def _http_error(self, response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            data = None

        error = None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            error = data["error"]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            candidate = data[0].get("error")
            if isinstance(candidate, dict):
                error = candidate

        if error:
            message = error.get("message")
            status = error.get("status")
            if message:
                suffix = f" ({status})" if status else ""
                return (
                    f"LLM request failed with HTTP {response.status_code}"
                    f"{suffix}: {message}"
                )

        return (
            f"LLM request failed with HTTP {response.status_code}: "
            f"{response.text[:2000]}"
        )

    def _retry_delay(
        self,
        attempt: int,
        response: requests.Response | None,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, min(float(retry_after), 60.0))
                except ValueError:
                    pass

        return min(2 ** (attempt - 1), 16) + random.uniform(0.0, 1.0)

    def _timeout(self) -> float:
        try:
            value = float(
                getattr(self.settings, "llm_timeout_seconds", 60)
            )
        except (TypeError, ValueError):
            value = 60.0
        return max(1.0, value)

    def _max_retries(self) -> int:
        try:
            value = int(getattr(self.settings, "llm_max_retries", 3))
        except (TypeError, ValueError):
            value = 3
        return max(0, min(value, 10))

    def _max_tokens(self) -> int:
        try:
            value = int(getattr(self.settings, "llm_max_tokens", 2048))
        except (TypeError, ValueError):
            value = 2048
        return max(1, value)

    def _value(self, name: str) -> str:
        return str(getattr(self.settings, name, "") or "").strip()

    def _fail(self, message: str) -> None:
        self.last_error = message
        logger.error(message)
        return None

    @staticmethod
    def _format_instruction(response_format: str) -> str:
        value = str(response_format or "").strip().lower()
        if value in {"short", "concise", "brief"}:
            return "Provide a short and direct answer."
        if value in {"detailed", "long", "complete"}:
            return "Provide a complete and detailed answer."
        if value in {"bullet points", "bullets", "list"}:
            return "Use clear bullet points where appropriate."
        return "Choose the response length and format that best fits the question."

    @staticmethod
    def _extract_content(content: Any) -> str | None:
        if isinstance(content, str):
            return content.strip() or None

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())

            combined = "\n".join(parts).strip()
            return combined or None

        return None