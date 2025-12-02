# backend/src/rag/llm_client.py

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import httpx

from utils.config import settings as global_settings
import os
import httpx
from typing import Optional
from utils.config import settings
from utils.errors import LLMError


class LLMClient:
    """Provider-agnostic chat completion client with explicit settings support."""

    def __init__(self, settings: Optional[object] = None) -> None:
        # Allow dependency injection from AdvancedRAGSystem while keeping a
        # module-level singleton that reuses global settings.
        self.settings = settings or global_settings
        self.clients: Dict[str, Dict[str, str]] = self._load_providers()

    # ------------------------------------------------------------------
    # Provider discovery
    # ------------------------------------------------------------------
    def _load_providers(self) -> Dict[str, Dict[str, str]]:
        providers: Dict[str, Dict[str, str]] = {}

        if getattr(self.settings, "OPENAI_API_KEY", None):
            providers["openai"] = {
                "key": self.settings.OPENAI_API_KEY,
                "url": "https://api.openai.com/v1/chat/completions",
            }

        if getattr(self.settings, "GROQ_API_KEY", None):
            providers["groq"] = {
                "key": self.settings.GROQ_API_KEY,
                "url": "https://api.groq.com/openai/v1/chat/completions",
            }

        if getattr(self.settings, "MISTRAL_API_KEY", None):
            providers["mistral"] = {
                "key": self.settings.MISTRAL_API_KEY,
                "url": "https://api.mistral.ai/v1/chat/completions",
            }

        return providers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def ask(self, prompt: str, model: Optional[str] = None) -> str:
        model_name = model or getattr(self.settings, "DEFAULT_MODEL", "gpt-4o-mini")

        if not self.clients:
            raise LLMError("No LLM providers configured")

        errors = []
        for provider, cfg in self._provider_order(self.clients):
            try:
                response = await self._post_chat_completion(cfg, prompt, model_name)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

                errors.append(f"{provider}: {response.status_code} {response.text[:120]}")
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"{provider}: {exc}")

        raise LLMError(
            "All LLM providers failed" + ("; " + "; ".join(errors) if errors else "")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _post_chat_completion(self, cfg: Dict[str, str], prompt: str, model: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60) as client:
            return await client.post(
                cfg["url"],
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {cfg['key']}"},
            )

    def _provider_order(self, providers: Dict[str, Dict[str, str]]) -> Iterable[Tuple[str, Dict[str, str]]]:
        """Yield providers in deterministic order with a fallback model pass.

        Providers are iterated in insertion order. After exhausting them, the
        method switches the configured default model to the fallback (if
        different) so a second pass can try a more permissive model name.
        """

        fallback_model = getattr(self.settings, "FALLBACK_MODEL", None)
        default_model = getattr(self.settings, "DEFAULT_MODEL", None)

        for item in providers.items():
            yield item

        if fallback_model and fallback_model != default_model:
            # Update the default model in-place so the next iteration uses it.
            setattr(self.settings, "DEFAULT_MODEL", fallback_model)
            for item in providers.items():
                yield item

    async def ask(self, prompt: str, model: Optional[str] = None):
        """
        Fallback priority:
        1. OpenAI
        2. Google Gemini
        3. Local LM Studio
        """

        # ========= OpenAI =========
        try:
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return await self._openai(prompt, model)
        except:
            pass

        # ========= Gemini =========
        try:
            key = os.getenv("GEMINI_API_KEY")
            if key:
                return await self._gemini(prompt)
        except:
            pass

        # ========= LM Studio =========
        return await self._local(prompt)


    async def _openai(self, prompt, model):
        import openai
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        resp = await client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message["content"]

    async def _gemini(self, prompt):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText"
        params = {"key": os.getenv("GEMINI_API_KEY")}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, params=params, json={"prompt": {"text": prompt}})
            data = resp.json()
            return data["candidates"][0]["output"]

    async def _local(self, prompt):
        """
        LM Studio local model endpoint.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            return resp.json()["choices"][0]["message"]["content"]


llm_client = LLMClient()
