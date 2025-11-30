from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from utils.config import settings


class SupabaseClient:
    """
    Very small Supabase REST client.

    This is NOT the official client, but a simple wrapper to call
    your Supabase project's REST endpoints and storage if needed.
    """

    def __init__(self, url: str, anon_key: str) -> None:
        self._url = url.rstrip("/")
        self._anon_key = anon_key
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers={
                "apikey": self._anon_key,
                "Authorization": f"Bearer {self._anon_key}",
            },
            timeout=15.0,
        )

    @classmethod
    def from_settings(cls) -> "SupabaseClient":
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise RuntimeError("Supabase URL/anon key not configured")
        return cls(str(settings.supabase_url), settings.supabase_anon_key)

    async def fetch_table(self, table: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = await self._client.get(f"/rest/v1/{table}", params=params)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()
