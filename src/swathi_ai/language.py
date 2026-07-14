from __future__ import annotations

import re

TANGLISH_WORDS = {
    "vanakkam", "epdi", "eppadi", "iruka", "irukinga", "irukkinga", "enna",
    "un", "unga", "peyar", "sollu", "poitu", "varen", "nandri", "romba",
    "seri", "saptiya", "saaptiya", "machi", "amma", "appa", "naan", "nee",
    "yaar", "venum", "pannu", "theriyuma", "illa", "aama", "oru", "udhavi",
    "poren", "nalla", "saapadu", "padippu", "paatu", "padam", "mazhai",
    "veyil", "vayasu", "sapadu",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_language(text: str) -> str:
    has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", text))
    latin_words = re.findall(r"[a-zA-Z]+", text.lower())
    if has_tamil:
        return "tanglish" if latin_words else "tamil"
    if any(word in TANGLISH_WORDS for word in latin_words):
        return "tanglish"
    return "english"
