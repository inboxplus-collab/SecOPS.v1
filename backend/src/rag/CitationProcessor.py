from __future__ import annotations

from typing import List

from rag.AdvancedRAGTypes import RAGChunk


class CitationProcessor:
    """Utilities for ranking and formatting retrieved chunks."""

    def merge_and_rank(self, chunks: List[RAGChunk]) -> List[RAGChunk]:
        seen: set[str] = set()
        unique_chunks: List[RAGChunk] = []

        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
            text = chunk.text.strip()
            if text in seen:
                continue
            seen.add(text)
            unique_chunks.append(chunk)

        return unique_chunks[:10]

    def format_citations(self, chunks: List[RAGChunk]) -> List[dict]:
        return [
            {"source": chunk.source, "score": round(chunk.score, 3), "preview": chunk.text[:200]}
            for chunk in chunks
        ]


citation_processor = CitationProcessor()
