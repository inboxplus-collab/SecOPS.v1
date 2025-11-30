from __future__ import annotations

from typing import List

from .AdvancedRAGTypes import RagContext, RagAnswer, RetrievedChunk
from .CitationProcessor import build_citations
from .llm_client import LLMClient


class RAGSynthesizer:
    """
    Responsible for turning retrieved chunks into a final natural language answer.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

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

    async def synthesize(self, ctx: RagContext) -> RagAnswer:
        prompt = self._build_prompt(ctx)
        text = await self._llm.generate(prompt)
        return RagAnswer(
            answer=text,
            intent=ctx.intent,
            mode="rag",
            citations=build_citations(ctx.retrieved),
        )
