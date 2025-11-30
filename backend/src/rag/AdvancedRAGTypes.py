from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RagContext:
    question: str
    intent: str
    retrieved: List[RetrievedChunk]
    extra: Dict[str, Any]


@dataclass
class RagAnswer:
    answer: str
    intent: str
    mode: str
    citations: List[Dict[str, Any]]
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
