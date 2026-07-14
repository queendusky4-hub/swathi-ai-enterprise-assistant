from __future__ import annotations

from typing import Final

from .language import clean_text

LABEL_MAP: Final = {0: "greeting", 1: "how_are_you", 2: "name", 3: "age", 4: "joke", 5: "bye", 6: "thanks", 7: "help", 8: "study", 9: "food", 10: "weather", 11: "ai"}

RULES: Final = {
    "how_are_you": ["epdi iruka", "epdi irukinga", "eppadi iruk", "how are you", "how r u", "எப்படி இருக்க"],
    "greeting": ["vanakkam", "வணக்கம்", "hello", "hi", "hey", "ஹலோ"],
    "name": ["your name", "what is your name", "உன் பெயர்", "உங்கள் பெயர்", "un peyar", "un per"],
    "age": ["how old", "வயது", "vayasu", " age"],
    "joke": ["joke", "ஜோக்", "நகைச்சுவை"],
    "bye": ["bye", "goodbye", "பிரியாவிடை", "poitu varen", "see you"],
    "thanks": ["thanks", "thank you", "nandri", "நன்றி"],
    "help": ["help", "udhavi", "உதவி"],
    "study": ["study", "padippu", "படிப்பு", "exam", "school", "college"],
    "food": ["food", "saapadu", "sapadu", "சாப்பாடு", "உணவு", "hungry"],
    "weather": ["weather", "வானிலை", "mazhai", "veyil", "rain", "sun"],
    "movie": ["movie", "padam", "படம்", "cinema", "film"],
    "music": ["music", "paatu", "பாட்டு", "song"],
    "cricket": ["cricket", "match", "score", "ipl"],
    "ai": ["artificial intelligence", "chatbot", "chatgpt", " ai"],
}


def rule_based_intent(text: str) -> str | None:
    value = f" {clean_text(text)}"
    for intent, keywords in RULES.items():
        if any(keyword in value for keyword in keywords):
            return intent
    return None
