from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from utils.config import settings


class LLMClient:
    """
    Minimal generic LLM client.

    For production you can:
      - Point LLM_BASE_URL to your gateway (OpenAI, Emergent, etc.).
      - Set LLM_API_KEY for authentication.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.rag_default_model
        self._base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self._api_key = os.getenv("LLM_API_KEY", "")

    async def generate(self, prompt: str, max_tokens: int = 768) -> str:
        """
        Send a simple completion-style request.

        This is intentionally generic: adjust based on your provider's API.
        """
        if not self._base_url:
            raise RuntimeError("LLM_BASE_URL not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else None,
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            # adjust based on your provider's response
            return data["choices"][0]["message"]["content"]
