from __future__ import annotations

from typing import Optional


class LLMClient:
    """Minimal async client stub for generating answers."""

    async def ask(self, prompt: str, model: Optional[str] = None) -> str:
        _ = model
        return f"[t79 llm stub] {prompt.strip()[:200]}"


llm_client = LLMClient()
