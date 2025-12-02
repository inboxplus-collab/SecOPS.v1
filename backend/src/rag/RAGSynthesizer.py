from rag.AdvancedRAGTypes import RAGChunk, RAGQuery
from rag.llm_client import llm_client


class RAGSynthesizer:

    async def synthesize(self, query: RAGQuery, chunks: list[RAGChunk]) -> str:
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])

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
