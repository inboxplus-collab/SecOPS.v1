import os
from integrations.targets.code_client import CodeClient
from rag.llm_client import llm_client


class AIOrchestrator:
    async def suggest_fixes(self, repo_path: str):
        code = CodeClient(repo_path)
        issues = code.run_static_analysis()

        prompt = f"""
        Analyze these code issues and generate:
        - exact fixes for each issue
        - new code blocks
        - explanation of changes

        ISSUES:
        {issues}
        """

        return await llm_client.ask(prompt)

    async def auto_repair(self, repo_path: str, confirm_callback):
        """
        Only applies a fix if the user confirms.
        """

        suggestions = await self.suggest_fixes(repo_path)

        apply = await confirm_callback(suggestions)
        if not apply:
            return {"status": "aborted", "message": "User rejected auto-fix."}

        return {"status": "success", "applied_changes": suggestions}


aio_orchestrator = AIOrchestrator()
