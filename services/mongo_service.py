"""
MongoDB service — async product & user data access via Motor.
Collections:
  products    : catalog with precomputed embeddings
  users       : user profiles and preferences
  interactions: click / purchase / like events (used to train CF)
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import numpy as np

from config.settings import get_settings

settings = get_settings()


class MongoService:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        self.client = AsyncIOMotorClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_db]
        # Indexes for fast lookups
        await self.db.products.create_index("product_id", unique=True)
        await self.db.products.create_index("category")
        await self.db.users.create_index("user_id", unique=True)
        await self.db.interactions.create_index([("user_id", 1), ("timestamp", -1)])

    async def disconnect(self):
        if self.client:
            self.client.close()

    # ── Products ──────────────────────────────────────────────────────────────

    async def get_product(self, product_id: str) -> Optional[dict]:
        return await self.db.products.find_one({"product_id": product_id}, {"_id": 0})

    async def get_products_by_ids(self, product_ids: list[str]) -> list[dict]:
        cursor = self.db.products.find(
            {"product_id": {"$in": product_ids}}, {"_id": 0}
        )
        return await cursor.to_list(length=len(product_ids))

    async def get_all_products_with_embeddings(self) -> list[dict]:
        """Returns all products that have precomputed embeddings."""
        cursor = self.db.products.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "product_id": 1, "name": 1, "category": 1,
             "tags": 1, "embedding": 1}
        )
        return await cursor.to_list(length=10_000)

    async def upsert_product_embedding(self, product_id: str, embedding: list[float]):
        await self.db.products.update_one(
            {"product_id": product_id},
            {"$set": {"embedding": embedding}},
            upsert=False,
        )

    # ── Users ─────────────────────────────────────────────────────────────────

    async def get_user(self, user_id: str) -> Optional[dict]:
        return await self.db.users.find_one({"user_id": user_id}, {"_id": 0})

    async def upsert_user(self, user_id: str, profile: dict):
        await self.db.users.update_one(
            {"user_id": user_id},
            {"$set": profile},
            upsert=True,
        )

    # ── Interactions ──────────────────────────────────────────────────────────

    async def log_interaction(self, user_id: str, product_id: str,
                               event_type: str, weight: float = 1.0):
        """
        event_type: 'view' | 'like' | 'add_to_cart' | 'purchase'
        weight: higher = stronger signal (purchase > cart > like > view)
        """
        from datetime import datetime, timezone
        await self.db.interactions.insert_one({
            "user_id": user_id,
            "product_id": product_id,
            "event_type": event_type,
            "weight": weight,
            "timestamp": datetime.now(timezone.utc),
        })

    async def get_interaction_matrix(self) -> dict:
        """
        Returns {user_id: {product_id: weight}} for CF training.
        Aggregates multiple events per (user, product) pair.
        """
        pipeline = [
            {
                "$group": {
                    "_id": {"user_id": "$user_id", "product_id": "$product_id"},
                    "score": {"$sum": "$weight"},
                }
            }
        ]
        cursor = self.db.interactions.aggregate(pipeline)
        results = await cursor.to_list(length=100_000)

        matrix: dict = {}
        for r in results:
            uid = r["_id"]["user_id"]
            pid = r["_id"]["product_id"]
            matrix.setdefault(uid, {})[pid] = r["score"]
        return matrix

    async def get_user_history(self, user_id: str, limit: int = 20) -> list[dict]:
        """Most recent interactions for a user — used to build LLM context."""
        cursor = self.db.interactions.find(
            {"user_id": user_id},
            {"_id": 0, "product_id": 1, "event_type": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)


# Module-level singleton
mongo_service = MongoService()
