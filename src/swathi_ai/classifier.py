from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .intents import LABEL_MAP

logger = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._pipeline: Any | None = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() and any(self.model_path.iterdir())

    def load(self) -> bool:
        if not self.available:
            return False
        try:
            import torch
            from transformers import pipeline
            self._pipeline = pipeline("text-classification", model=str(self.model_path), tokenizer=str(self.model_path), device=0 if torch.cuda.is_available() else -1)
            return True
        except Exception as exc:  # model loading should not crash the app
            logger.exception("Could not load intent model: %s", exc)
            self._pipeline = None
            return False

    def predict(self, text: str) -> tuple[str, float] | None:
        if self._pipeline is None and not self.load():
            return None
        try:
            result = self._pipeline(text)[0]
            label_text = str(result["label"])
            label_id = int(label_text.split("_")[-1]) if "_" in label_text else int(label_text)
            intent = LABEL_MAP.get(label_id)
            return (intent, float(result["score"])) if intent else None
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Invalid classifier output: %s", exc)
            return None
