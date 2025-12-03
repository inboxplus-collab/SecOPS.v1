from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DocResult:
    content: str
    source: str
    score: float


class InMemoryVectorStore:
    """Tiny placeholder vector store used for demo docs chat."""

    def __init__(self) -> None:
        self._docs: List[DocResult] = [
            DocResult(content="Connect GitHub to start code scanning.", source="docs/getting-started.md", score=0.89),
            DocResult(content="Kubernetes agent requires a read-only service account.", source="docs/kubernetes.md", score=0.82),
            DocResult(content="RAG answers include citations for every chunk.", source="docs/rag.md", score=0.79),
        ]

    def search(self, query: str, limit: int = 3) -> List[DocResult]:
        # Return top items with a naive slice; in production wire to Qdrant/PGVector.
        _ = query
        return self._docs[:limit]


class SimpleLLM:
    """Stub LLM that assembles an answer using retrieved docs."""

    def answer_with_citations(self, query: str, results: List[DocResult]) -> dict:
        citations = [f"{r.source} (relevance={r.score:.2f})" for r in results]
        synthesized = f"Based on the knowledge base, here is guidance for '{query}'."
        key_takeaway = results[0].content if results else "No documents found."
        return {"answer": f"{synthesized} Key point: {key_takeaway}", "citations": citations}


vector_store = InMemoryVectorStore()
llm = SimpleLLM()
