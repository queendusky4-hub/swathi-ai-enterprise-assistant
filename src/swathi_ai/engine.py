from __future__ import annotations

from dataclasses import dataclass

from .classifier import IntentClassifier
from .intents import rule_based_intent
from .language import detect_language
from .llm import LLMClient
from .responses import format_reply


@dataclass(frozen=True)
class ChatResult:
    text: str
    source: str
    intent: str | None = None
    confidence: float | None = None


class ChatEngine:
    def __init__(
        self,
        classifier: IntentClassifier,
        llm: LLMClient,
        threshold: float = 0.55,
    ) -> None:
        self.classifier = classifier
        self.llm = llm
        self.threshold = threshold

    @staticmethod
    def resolve_response_format(
        text: str,
        response_format: str,
    ) -> tuple[str, bool]:
        detected_language = detect_language(text)
        selected = response_format.strip().lower()

        if selected == "tamil only":
            return "tamil", False

        if selected == "tanglish only":
            return "tanglish", False

        if selected == "english only":
            return "english", False

        if selected == "all three":
            return detected_language, True

        return detected_language, False

    def respond(
        self,
        text: str,
        online: bool,
        response_format: str = "Auto detect",
        history: list[tuple[str, str]] | None = None,
        show_all: bool | None = None,
    ) -> ChatResult:
        cleaned_text = text.strip()

        if show_all is not None:
            response_format = (
                "All three"
                if show_all
                else "Auto detect"
            )

        language, resolved_show_all = (
            self.resolve_response_format(
                text=cleaned_text,
                response_format=response_format,
            )
        )

        rule_intent = rule_based_intent(
            cleaned_text
        )

        if rule_intent:
            return ChatResult(
                text=format_reply(
                    rule_intent,
                    language,
                    resolved_show_all,
                ),
                source="rule",
                intent=rule_intent,
                confidence=1.0,
            )

        prediction = self.classifier.predict(
            cleaned_text
        )

        if prediction:
            intent, confidence = prediction

            if confidence >= self.threshold:
                return ChatResult(
                    text=format_reply(
                        intent,
                        language,
                        resolved_show_all,
                    ),
                    source="bert",
                    intent=intent,
                    confidence=confidence,
                )

        if online:
            llm_reply = self.llm.generate(
                user_text=cleaned_text,
                language=language,
                response_format=response_format,
                history=history,
            )

            if llm_reply:
                return ChatResult(
                    text=llm_reply,
                    source="llm",
                )

        fallback_intent = (
            "fallback"
            if online
            else "offline"
        )

        return ChatResult(
            text=format_reply(
                fallback_intent,
                language,
                resolved_show_all,
            ),
            source=fallback_intent,
        )