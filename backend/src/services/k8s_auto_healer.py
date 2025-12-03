from backend.src.rag.llm_client import llm_client


async def explain_issue(issue):
    prompt = f"""
Explain this Kubernetes issue in plain English and propose a fix:

ISSUE:
{issue}

Explain:
1. Root cause
2. Severity
3. Exact fix steps
4. Preventive best practices
"""
    return await llm_client.ask(prompt)
