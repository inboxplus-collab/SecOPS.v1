from __future__ import annotations

from typing import List

from integrations.storage.vector_store import vector_store
from rag.AdvancedRAGTypes import RAGChunk, RAGQuery
from rag.CitationProcessor import citation_processor


class SearchOrchestrator:
    """Coordinates retrieval pipelines for RAG queries."""

    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        vector_results = await vector_store.search(query.query, top_k=query.top_k)

        if not vector_results:
            vector_results = vector_store.keyword_search(query.query)

        return citation_processor.merge_and_rank(vector_results)


search_orchestrator = SearchOrchestrator()
