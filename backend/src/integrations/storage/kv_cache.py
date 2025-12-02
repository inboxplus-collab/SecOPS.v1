# backend/src/integrations/storage/kv_cache.py

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.config import Settings  # type: ignore[attr-defined]
from integrations.storage.supabase_client import (
    get_supabase_client,
    SupabaseClient,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Base Interface
# ======================================================================

class KeyValueCache(ABC):
    """
    Asynchronous Redis-like KV cache interface.

    Methods:
      - get(key) -> Any | None
      - set(key, value, ttl_seconds=None)
      - delete(key)
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError


# ======================================================================
# Memory backend (default)
# ======================================================================

class MemoryKeyValueCache(KeyValueCache):
    """
    Safe default KV cache: in-memory dictionary with TTL.

    Good for local/dev or fallback when Redis/Supabase unavailable.
    """

    def __init__(self) -> None:
        # key -> {"value": X, "expires": timestamp or None}
        self._store: Dict[str, Dict[str, Any]] = {}

        logger.info("MemoryKeyValueCache initialized (development mode)")

    async def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None

        expires = item.get("expires")
        if expires and time.time() > expires:
            # TTL expired
            self._store.pop(key, None)
            return None

        return item.get("value")

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None

        self._store[key] = {
            "value": value,
            "expires": expires,
        }

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


# ======================================================================
# Redis backend (production recommended)
# ======================================================================

class RedisKeyValueCache(KeyValueCache):
    """
    Redis-based KV cache using redis.asyncio.

    Automatically falls back to MemoryKeyValueCache if:
      - redis package is missing
      - host unreachable
      - any connection error arises
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Allow flexible env / config
        self.url = (
            getattr(settings, "REDIS_URL", None)
            or os.getenv("REDIS_URL")
            or "redis://localhost:6379/0"
        )

        try:
            import redis.asyncio as redis  # type: ignore
            self.redis = redis.Redis.from_url(self.url, decode_responses=True)
            self._healthy = True
            logger.info("RedisKeyValueCache connected to %s", self.url)
        except Exception as exc:
            logger.warning("RedisKeyValueCache fallback: redis missing or failed: %s", exc)
            self.redis = None
            self._healthy = False

        if not self._healthy:
            logger.warning("RedisKeyValueCache: using in-memory fallback")
            self.fallback = MemoryKeyValueCache()
        else:
            self.fallback = None

    async def get(self, key: str) -> Optional[Any]:
        if not self._healthy:
            return await self.fallback.get(key)

        try:
            raw = await self.redis.get(key)
        except Exception as exc:
            logger.error("Redis GET failed, falling back to memory: %s", exc)
            self._healthy = False
            return await self.fallback.get(key)

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self._healthy:
            return await self.fallback.set(key, value, ttl_seconds)

        try:
            raw = json.dumps(value)
            if ttl_seconds:
                await self.redis.set(key, raw, ex=ttl_seconds)
            else:
                await self.redis.set(key, raw)
        except Exception as exc:
            logger.error("Redis SET failed, falling back to memory: %s", exc)
            self._healthy = False
            await self.fallback.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        if not self._healthy:
            return await self.fallback.delete(key)

        try:
            await self.redis.delete(key)
        except Exception as exc:
            logger.error("Redis DEL failed, falling back to memory: %s", exc)
            self._healthy = False
            await self.fallback.delete(key)


# ======================================================================
# Supabase KV store backend (optional)
# ======================================================================

class SupabaseKeyValueCache(KeyValueCache):
    """
    Persistent KV cache stored in Supabase table.

    Schema expected:

      CREATE TABLE kv_cache (
        key text PRIMARY KEY,
        value jsonb,
        expires_at timestamptz
      );

    Settings:
      KV_SUPABASE_TABLE = "kv_cache"

    Notes:
      - Not as fast as Redis.
      - Safe fallback if Supabase not configured.
    """

    def __init__(self, settings: Settings, client: SupabaseClient) -> None:
        self.settings = settings
        self.client = client
        self.table = (
            getattr(settings, "KV_SUPABASE_TABLE", None)
            or os.getenv("KV_SUPABASE_TABLE")
            or "kv_cache"
        )

        if not client._enabled:  # type: ignore[attr-defined]
            logger.warning("Supabase KV cache disabled (Supabase not configured)")

    async def get(self, key: str) -> Optional[Any]:
        if not self.client._enabled:  # type: ignore[attr-defined]
            return None

        rows = await self.client.select(
            self.table,
            filters={"key": key},
            limit=1,
        )
        if not rows:
            return None

        row = rows[0]

        expires_at = row.get("expires_at")
        if expires_at:
            try:
                ts = float(expires_at)
                if time.time() > ts:
                    await self.delete(key)
                    return None
            except Exception as exc:
                logger.debug("SupabaseKeyValueCache expiry parse failed for %s: %s", key, exc)

        return row.get("value")

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self.client._enabled:  # type: ignore[attr-defined]
            return

        expires_at = time.time() + ttl_seconds if ttl_seconds else None

        row = {
            "key": key,
            "value": value,
            "expires_at": expires_at,
        }

        await self.client.upsert(
            self.table,
            rows=row,
            on_conflict="key",
            returning="minimal",
        )

    async def delete(self, key: str) -> None:
        if not self.client._enabled:  # type: ignore[attr-defined]
            return

        await self.client.delete(self.table, filters={"key": key})


# ======================================================================
# Factory
# ======================================================================

def get_kv_cache(settings: Settings) -> KeyValueCache:
    """
    Factory for dependency injection.

    KV_CACHE_BACKEND = "redis" | "supabase" | "memory"

    Defaults to "memory".

    Redis recommended for production.
    """

    backend = (
        getattr(settings, "KV_CACHE_BACKEND", None)
        or os.getenv("KV_CACHE_BACKEND")
        or "memory"
    ).lower()

    if backend == "redis":
        return RedisKeyValueCache(settings=settings)

    if backend == "supabase":
        return SupabaseKeyValueCache(
            settings=settings,
            client=get_supabase_client(settings),
        )

    # default
    return MemoryKeyValueCache()
