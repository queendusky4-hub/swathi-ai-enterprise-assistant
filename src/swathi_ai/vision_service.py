from __future__ import annotations

from .config import settings
from .llm import LLMClient


class VisionService:

    def __init__(self):
        self.llm = LLMClient(settings)

    @property
    def available(self):
        return self.llm.configured

    def analyze_image(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/png",
        response_format: str = "Auto detect",
    ):

        answer = self.llm.generate_with_image(
            prompt=question,
            image_bytes=image_bytes,
            mime_type=mime_type,
            response_format=response_format,
        )

        if answer is None:
            raise RuntimeError(
                self.llm.last_error or "Vision model failed."
            )

        return answer


_vision_service = VisionService()


def get_vision_service():
    return _vision_service