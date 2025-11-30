from __future__ import annotations

import time
from typing import List

from integrations.storage.vector_store import InMemoryVectorStore, VectorDoc
from .AdvancedRAGTypes import RetrievedChunk, RagContext, RagAnswer
from .RAGSynthesizer import RAGSynthesizer
from .llm_client import LLMClient


class SearchOrchestrator:
    """
    High-level RAG orchestrator:
      - embed question
      - search vector store
      - call synthesizer
    """

    def __init__(
        self,
        store: InMemoryVectorStore | None = None,
        synthesizer: RAGSynthesizer | None = None,
    ) -> None:
        self._store = store or InMemoryVectorStore()
        self._synth = synthesizer or RAGSynthesizer(LLMClient())

    async def query(self, question: str, intent: str = "general") -> RagAnswer:
        start = time.time()

        # For MVP, we use a super simple "embedding": word-level hashing.
        # Replace with a real embedding model later.
        def embed(text: str) -> list[float]:
            return [float(len(text) % 13), float(len(text) % 17), float(len(text) % 19)]

        q_emb = embed(question)
        results = self._store.search(q_emb, top_k=5)

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(
                id=doc.id,
                text=doc.metadata.get("text", ""),
                score=score,
                metadata=doc.metadata,
            )
            for (doc, score) in results
        ]

        ctx = RagContext(
            question=question,
            intent=intent,
            retrieved=chunks,
            extra={},
        )
        ans = await self._synth.synthesize(ctx)
        ans.latency_ms = int((time.time() - start) * 1000)
        return ans

    def upsert_doc(self, id: str, text: str, metadata: dict[str, str]) -> None:
        def embed(text: str) -> list[float]:
            return [float(len(text) % 13), float(len(text) % 17), float(len(text) % 19)]

        doc = VectorDoc(id=id, embedding=embed(text), metadata={**metadata, "text": text})
        self._store.upsert(doc)
