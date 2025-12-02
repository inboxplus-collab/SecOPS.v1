from typing import List
from backend.src.integrations.storage.vector_store import vector_store
from backend.src.rag.AdvancedRAGTypes import RAGQuery, RAGChunk
from backend.src.rag.CitationProcessor import citation_processor


class SearchOrchestrator:

    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        """
        Hybrid search:
        - semantic vector search
        - keyword fallback
        - metadata routing
        """

        # 1. Semantic
        vector_results = await vector_store.search(query.query, top_k=query.top_k)

        # 2. Simple keyword fallback (if vector empty)
        if not vector_results:
            keyword_matches = vector_store.keyword_search(query.query)
            vector_results = keyword_matches

        # 3. Ranking & merge
        return citation_processor.merge_and_rank(vector_results)


search_orchestrator = SearchOrchestrator()
from __future__ import annotations

import hashlib
import time
from typing import List, Optional

from integrations.storage.vector_store import InMemoryVectorStore, VectorDoc
from .AdvancedRAGTypes import RagAnswer, RagContext, RetrievedChunk
from .RAGSynthesizer import RAGSynthesizer
from .llm_client import llm_client


class SearchOrchestrator:
    """Coordinate retrieval and synthesis for SecOpsAI responses."""

    def __init__(
        self,
        store: Optional[InMemoryVectorStore] = None,
        synthesizer: Optional[RAGSynthesizer] = None,
        settings: Optional[object] = None,
        llm_client_override=None,
    ) -> None:
        self._store = store or InMemoryVectorStore()
        self._synth = synthesizer or RAGSynthesizer(llm_client=llm_client_override or llm_client, settings=settings)

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        # Simple deterministic embedding for local use
        return [float(int.from_bytes(digest[i : i + 4], "big") % 9973) for i in range(0, 20, 4)]

    async def query(self, question: str, intent: str = "general") -> RagAnswer:
        start = time.time()
        q_emb = self._embed(question)
        results = self._store.search(q_emb, top_k=5)

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(id=doc.id, text=doc.metadata.get("text", ""), score=score, metadata=doc.metadata)
            for (doc, score) in results
        ]

        ctx = RagContext(question=question, intent=intent, retrieved=chunks, extra={})
        answer = await self._synth.synthesize(ctx)
        answer.latency_ms = int((time.time() - start) * 1000)
        return answer

    def upsert_doc(self, id: str, text: str, metadata: dict[str, str]) -> None:
        doc = VectorDoc(id=id, embedding=self._embed(text), metadata={**metadata, "text": text})
        self._store.upsert(doc)
