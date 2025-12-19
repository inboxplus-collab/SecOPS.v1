from __future__ import annotations

import os
from typing import List

import httpx

from rag.AdvancedRAGTypes import RAGChunk


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


class VectorStore:
    """Lightweight Supabase vector store wrapper with safe fallbacks."""

    async def search(self, query: str, top_k: int = 8) -> List[RAGChunk]:
        if not SUPABASE_URL or not SUPABASE_KEY:
            return []

        url = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers=headers,
                json={"query_text": query, "match_count": top_k},
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            RAGChunk(
                id=str(row.get("id", i)),
                text=row.get("content", ""),
                score=float(row.get("similarity", 0.0)),
                source=row.get("source", "unknown"),
                metadata=row.get("metadata", {}),
            )
            for i, row in enumerate(data)
        ]

    def keyword_search(self, query: str) -> List[RAGChunk]:
        # Placeholder keyword search for environments without Supabase
        return []


vector_store = VectorStore()
