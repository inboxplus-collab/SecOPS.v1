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
