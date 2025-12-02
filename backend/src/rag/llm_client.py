# backend/src/rag/llm_client.py

import httpx
from backend.src.utils.config import settings
from backend.src.utils.errors import LLMError


class LLMClient:
    def __init__(self):
        self.clients = {}

        if settings.OPENAI_API_KEY:
            self.clients["openai"] = {
                "key": settings.OPENAI_API_KEY,
                "url": "https://api.openai.com/v1/chat/completions"
            }

        if settings.GROQ_API_KEY:
            self.clients["groq"] = {
                "key": settings.GROQ_API_KEY,
                "url": "https://api.groq.com/openai/v1/chat/completions"
            }

        if settings.MISTRAL_API_KEY:
            self.clients["mistral"] = {
                "key": settings.MISTRAL_API_KEY,
                "url": "https://api.mistral.ai/v1/chat/completions"
            }

    async def ask(self, prompt: str, model: str = None) -> str:
        model = model or settings.DEFAULT_MODEL

        for provider, cfg in self.clients.items():
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    res = await client.post(
                        cfg["url"],
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        headers={"Authorization": f"Bearer {cfg['key']}"}
                    )
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]

            except Exception as e:
                continue  # try next provider

        raise LLMError("All LLM providers failed")


llm_client = LLMClient()
