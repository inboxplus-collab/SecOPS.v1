import os
from typing import List

import httpx

from rag.AdvancedRAGTypes import RAGChunk
import httpx
from typing import List
from backend.src.rag.AdvancedRAGTypes import RAGChunk

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


class VectorStore:

    async def search(self, query: str, top_k: int = 8) -> List[RAGChunk]:
        """
        Uses Supabase pgvector RPC function `match_documents`.
        """

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

            data = resp.json()

        chunks = [
            RAGChunk(
                id=str(i["id"]),
                text=i["content"],
                score=float(i["similarity"]),
                source=i.get("source", "unknown"),
                metadata=i.get("metadata", {}),
            )
            for i in data
        ]

        return chunks

    def keyword_search(self, query: str) -> List[RAGChunk]:
        # Simple local fallback (optional)
        return []


        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json={
                "query_text": query,
                "match_count": top_k
            })

            data = resp.json()

        chunks = [
            RAGChunk(
                id=str(i["id"]),
                text=i["content"],
                score=float(i["similarity"]),
                source=i.get("source", "unknown"),
                metadata=i.get("metadata", {}),
            )
            for i in data
        ]

        return chunks

    def keyword_search(self, query: str) -> List[RAGChunk]:
        # Simple local fallback (optional)
        return []


vector_store = VectorStore()
