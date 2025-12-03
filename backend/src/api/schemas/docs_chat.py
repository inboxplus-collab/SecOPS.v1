from __future__ import annotations

from pydantic import BaseModel


class QueryInput(BaseModel):
    text: str


class DocsChatAnswer(BaseModel):
    answer: str
    citations: list[str] = []
