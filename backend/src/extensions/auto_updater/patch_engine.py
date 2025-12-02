from backend.src.rag.llm_client import llm_client


class PatchEngine:

    async def generate_patch(self, filepath: str, code: str, issues: list[str]) -> str:
        prompt = f"""
You are an advanced code-fixing AI.
Your job: rewrite ONLY the necessary parts of the file.

Filepath: {filepath}

Existing code:
{code}

Issues detected:
{issues}

Output the FULL corrected file.
Do NOT leave placeholders.
Do NOT remove working logic.
"""

        return await llm_client.ask(prompt)


patch_engine = PatchEngine()
