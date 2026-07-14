from __future__ import annotations

from dataclasses import dataclass

from .classifier import IntentClassifier
from .language import detect_language
from .intents import rule_based_intent
from .llm import LLMClient
from .responses import format_reply


@dataclass(frozen=True)
class ChatResult:
    text: str
    source: str
    intent: str | None = None
    confidence: float | None = None


class ChatEngine:
    def __init__(self, classifier: IntentClassifier, llm: LLMClient, threshold: float = 0.55):
        self.classifier = classifier
        self.llm = llm
        self.threshold = threshold

    def respond(self, text: str, online: bool, show_all: bool = True) -> ChatResult:
        language = detect_language(text)
        rule_intent = rule_based_intent(text)
        if rule_intent:
            return ChatResult(format_reply(rule_intent, language, show_all), "rule", rule_intent, 1.0)

        prediction = self.classifier.predict(text)
        if prediction:
            intent, confidence = prediction
            if confidence >= self.threshold:
                return ChatResult(format_reply(intent, language, show_all), "bert", intent, confidence)

        if online:
            llm_reply = self.llm.generate(text)
            if llm_reply:
                return ChatResult(llm_reply, "llm")

        fallback = "fallback" if online else "offline"
        return ChatResult(format_reply(fallback, language, show_all), fallback)
