from __future__ import annotations

from typing import Any, Dict, List

from .AdvancedRAGTypes import RetrievedChunk


class CitationProcessor:
    """Normalize retrieved chunks into API-friendly citations."""

    def build(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for ch in chunks:
            citations.append(
                {
                    "id": ch.id,
                    "title": ch.metadata.get("title") or ch.metadata.get("path"),
                    "url": ch.metadata.get("url"),
                    "snippet": ch.text[:256],
                    "source_type": ch.metadata.get("source_type", "doc"),
                    "score": ch.score,
                }
            )
        return citations

    def process(self, answer: Any, retrieved: Any) -> Dict[str, Any]:
        """
        Attach citations to an answer object. Designed to be tolerant of the
        different shapes returned by basic and advanced RAG orchestrators.
        """

        chunks: List[RetrievedChunk] = []
        if isinstance(retrieved, list):
            chunks = [c for c in retrieved if isinstance(c, RetrievedChunk)]
        elif getattr(retrieved, "chunks", None):
            chunks = [c for c in retrieved.chunks if isinstance(c, RetrievedChunk)]

        citations = self.build(chunks)
        answer_text = getattr(answer, "answer", None) or str(answer)
        return {"text": answer_text, "sources": citations}


def build_citations(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    return CitationProcessor().build(chunks)
