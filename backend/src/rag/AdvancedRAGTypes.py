from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class RAGQuery(BaseModel):
    """Simple query container for the RAG pipeline."""

    query: str
    top_k: int = 8
    filters: Optional[dict] = None


class RAGChunk(BaseModel):
    """Normalized chunk returned from retrieval."""

    id: str
    text: str
    score: float
    source: str
    metadata: dict


class RAGResult(BaseModel):
    """Final synthesized result with citations."""

    query: str
    chunks: List[RAGChunk]
    synthesized_answer: str
    citations: List[dict]
