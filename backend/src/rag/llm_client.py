import os
import httpx
from typing import Optional


class LLMClient:

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
