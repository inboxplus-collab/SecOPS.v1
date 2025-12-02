from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RAGQuery(BaseModel):
    query: str
    top_k: int = 8
    filters: Optional[dict] = None


class RAGChunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    metadata: dict


class RAGResult(BaseModel):
    query: str
    chunks: List[RAGChunk]
    synthesized_answer: str
    citations: List[dict]


# ---------------------------------------------------------------------------
# Legacy compatibility types (kept to avoid breaking callers while RAG API
# migrates to the new RAGQuery/RAGChunk schema).
# ---------------------------------------------------------------------------


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
