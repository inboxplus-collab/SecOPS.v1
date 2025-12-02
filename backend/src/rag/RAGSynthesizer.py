from backend.src.rag.llm_client import llm_client
from backend.src.rag.AdvancedRAGTypes import RAGChunk, RAGQuery

class RAGSynthesizer:

    async def synthesize(self, query: RAGQuery, chunks: list[RAGChunk]) -> str:
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])
from typing import List, Optional

from .AdvancedRAGTypes import RagAnswer, RagContext, RetrievedChunk
from .CitationProcessor import CitationProcessor, build_citations
from .llm_client import llm_client

        prompt = f"""
You are SecOpsAI — an autonomous DevSecOps assistant.

USER QUESTION:
{query.query}

CONTEXT (retrieved from customer systems, logs, errors, scans):
{context}

TASK:
- Answer the user's question accurately.
- Use ONLY provided context.
- Include step-by-step reasoning.
- Provide actionable DevSecOps recommendations.
- Explain security risks if relevant.

FORMAT:
Answer only, no system text.
"""

        return await llm_client.ask(prompt)


synthesizer = RAGSynthesizer()
class RAGSynthesizer:
    """Turn retrieved knowledge into a SecOps-focused answer."""

    def __init__(self, llm_client=llm_client, settings: Optional[object] = None) -> None:
        self._llm = llm_client
        self._citation_processor = CitationProcessor()
        self._settings = settings

    def _build_prompt(self, ctx: RagContext) -> str:
        pieces: List[str] = [
            "You are SecOpsAI, an autonomous DevSecOps assistant.",
            "Use the given context to answer the user's question with clear, actionable steps.",
            "",
            "Context:",
        ]
        for i, ch in enumerate(ctx.retrieved[:8]):
            pieces.append(f"[{i+1}] {ch.text}")

        pieces.extend(
            [
                "",
                f"Question: {ctx.question}",
                "",
                "If you are not sure, say so clearly. Focus on security, reliability, and DevOps best practices.",
            ]
        )
        return "\n".join(pieces)

    async def synthesize(
        self,
        ctx: Optional[RagContext] = None,
        *,
        question: str | None = None,
        retrieved: List[RetrievedChunk] | None = None,
        intent: str = "general",
        context: Optional[object] = None,
    ) -> RagAnswer:
        if ctx is None:
            ctx = RagContext(question=question or "", intent=intent, retrieved=retrieved or [], extra={})

        prompt = self._build_prompt(ctx)
        try:
            text = await self._llm.ask(prompt)
            error_message = None
        except Exception as exc:  # noqa: BLE001
            text = "RAG summary unavailable."
            error_message = str(exc)

        return RagAnswer(
            answer=text,
            intent=ctx.intent,
            mode="rag",
            citations=self._citation_processor.build(ctx.retrieved),
            error_message=error_message,
        )
