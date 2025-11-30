from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import math


@dataclass
class VectorDoc:
    id: str
    embedding: List[float]
    metadata: Dict[str, str]


class InMemoryVectorStore:
    """
    Basic vector store for RAG.

    For production you should back this by pgvector in Supabase
    or a dedicated vector DB. API is intentionally simple: upsert + search.
    """

    def __init__(self) -> None:
        self._docs: Dict[str, VectorDoc] = {}

    def upsert(self, doc: VectorDoc) -> None:
        self._docs[doc.id] = doc

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[VectorDoc, float]]:
        scored = [
            (doc, self._cosine(query_embedding, doc.embedding))
            for doc in self._docs.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
