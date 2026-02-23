"""
Redis service — recommendation cache + user session store.

Key schema:
  rec:{user_id}          → cached recommendations list (JSON), TTL: 10 min
  session:{user_id}      → user session data (JSON), TTL: 30 min
  cf_model               → serialized CF model (pickle b64), TTL: 1 hour
"""

import json
import base64
import pickle
from typing import Optional, Any

import redis.asyncio as aioredis

from config.settings import get_settings

settings = get_settings()

# Key prefixes
REC_PREFIX = "rec:"
SESSION_PREFIX = "session:"
CF_MODEL_KEY = "cf_model"


class RedisService:
    def __init__(self):
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        self.client = await aioredis.from_url(
            settings.redis_uri,
            encoding="utf-8",
            decode_responses=True,
        )

    async def disconnect(self):
        if self.client:
            await self.client.aclose()

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    # ── Recommendation Cache ───────────────────────────────────────────────────

    async def get_cached_recommendations(self, user_id: str) -> Optional[list]:
        key = f"{REC_PREFIX}{user_id}"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    async def cache_recommendations(self, user_id: str, recommendations: list):
        key = f"{REC_PREFIX}{user_id}"
        await self.client.setex(
            key,
            settings.redis_ttl_seconds,
            json.dumps(recommendations),
        )

    async def invalidate_recommendations(self, user_id: str):
        """Call this when a user logs a new interaction — stale recs need refresh."""
        await self.client.delete(f"{REC_PREFIX}{user_id}")

    # ── User Session Store ─────────────────────────────────────────────────────

    async def set_session(self, user_id: str, session_data: dict, ttl: int = 1800):
        key = f"{SESSION_PREFIX}{user_id}"
        await self.client.setex(key, ttl, json.dumps(session_data))

    async def get_session(self, user_id: str) -> Optional[dict]:
        key = f"{SESSION_PREFIX}{user_id}"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def update_session(self, user_id: str, updates: dict):
        session = await self.get_session(user_id) or {}
        session.update(updates)
        await self.set_session(user_id, session)

    # ── CF Model Cache ─────────────────────────────────────────────────────────

    async def cache_cf_model(self, model: Any, ttl: int = 3600):
        """Serialize and cache the trained CF model to avoid retraining on every restart."""
        # Use raw bytes client for binary data
        raw_client = await aioredis.from_url(
            settings.redis_uri, decode_responses=False
        )
        try:
            serialized = base64.b64encode(pickle.dumps(model))
            await raw_client.setex(CF_MODEL_KEY, ttl, serialized)
        finally:
            await raw_client.aclose()

    async def get_cf_model(self) -> Optional[Any]:
        raw_client = await aioredis.from_url(
            settings.redis_uri, decode_responses=False
        )
        try:
            data = await raw_client.get(CF_MODEL_KEY)
            if data:
                return pickle.loads(base64.b64decode(data))
            return None
        finally:
            await raw_client.aclose()

    # ── Metrics helpers ────────────────────────────────────────────────────────

    async def increment_counter(self, key: str, ttl: int = 86400):
        """Simple hit counter for monitoring cache performance."""
        await self.client.incr(key)
        await self.client.expire(key, ttl)

    async def get_counter(self, key: str) -> int:
        val = await self.client.get(key)
        return int(val) if val else 0


# Module-level singleton
redis_service = RedisService()
