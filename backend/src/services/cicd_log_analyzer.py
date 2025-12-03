from backend.src.rag.llm_client import llm_client


async def rag_explain_ci_failure(run_data):
    prompt = f"""
Explain this CI/CD failure in detail:

{run_data}

Return:
- High-level explanation
- Root cause
- Fix steps
- Risks
- How to prevent it
"""
    return await llm_client.ask(prompt)
