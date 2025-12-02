from rag.AdvancedRAGTypes import RAGChunk, RAGQuery
from rag.llm_client import llm_client

from backend.src.rag.llm_client import llm_client
from backend.src.rag.AdvancedRAGTypes import RAGChunk, RAGQuery

class RAGSynthesizer:

    async def synthesize(self, query: RAGQuery, chunks: list[RAGChunk]) -> str:
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])
from typing import List, Optional

class RAGSynthesizer:

    async def synthesize(self, query: RAGQuery, chunks: list[RAGChunk]) -> str:
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])

        prompt = f"""
You are SecOpsAI — an autonomous DevSecOps assistant.

USER QUESTION:
{query.query}
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
