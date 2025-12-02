"""LLM-guided remediation helper with optional auto-repair."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from integrations.targets.code_client import CodeClient
from rag.llm_client import llm_client


ConfirmCallback = Callable[[str], Awaitable[bool]]


class AIOrchestrator:
    """Coordinate LLM reasoning with repository-aware diagnostics."""

    async def suggest_fixes(self, repo_path: str) -> str:
        """
        Run static analysis on the repository and ask the LLM for precise fixes.
        """

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

    async def auto_repair(self, repo_path: str, confirm_callback: ConfirmCallback) -> Dict[str, Any]:
        """
        Suggest fixes and only apply them when the caller confirms.

        Currently returns the LLM suggestion payload so the caller can review
        and optionally apply patches with a follow-up step.
        """

        suggestions = await self.suggest_fixes(repo_path)

        apply = await confirm_callback(suggestions)
        if not apply:
            return {"status": "aborted", "message": "User rejected auto-fix."}

        # In a real remediation flow, this is where patches would be applied
        # using CodeClient.rewrite_file or similar helpers.
        return {"status": "success", "applied_changes": suggestions}


aio_orchestrator = AIOrchestrator()
