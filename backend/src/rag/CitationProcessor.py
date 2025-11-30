from __future__ import annotations

from typing import Any, Dict, List

from .AdvancedRAGTypes import RetrievedChunk


def build_citations(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for ch in chunks:
        citations.append(
            {
                "id": ch.id,
                "title": ch.metadata.get("title") or ch.metadata.get("path"),
                "url": ch.metadata.get("url"),
                "snippet": ch.text[:256],
                "source_type": ch.metadata.get("source_type", "doc"),
            }
        )
    return citations
