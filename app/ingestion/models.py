from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractedInput(BaseModel):
    source: str = Field(description="Filename or label.")
    type: str = Field(description="One of: text, image, pdf, audio.")
    text: str = Field(default="", description="Extracted / transcribed text.")
    meta: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None)

    @property
    def ok(self) -> bool:
        return self.error is None
