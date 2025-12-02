from typing import List

from rag.AdvancedRAGTypes import RAGChunk


class CitationProcessor:

    def merge_and_rank(self, chunks: List[RAGChunk]) -> List[RAGChunk]:
        """
        Removes duplicates, ranks by score, merges similar data.
        """
        seen = set()
        unique_chunks = []

        for c in sorted(chunks, key=lambda x: x.score, reverse=True):
            if c.text.strip() not in seen:
                unique_chunks.append(c)
                seen.add(c.text.strip())

        return unique_chunks[:10]

    def format_citations(self, chunks: List[RAGChunk]):
        """
        Format citations for final answer.
        """
        return [
            {"source": c.source, "score": round(c.score, 3), "preview": c.text[:200]}
            for c in chunks
        ]


citation_processor = CitationProcessor()
