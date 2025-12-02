from typing import List, Optional
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
