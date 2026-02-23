from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

from services.mongo_service import mongo_service
from services.redis_service import redis_service

router = APIRouter()

EVENT_WEIGHTS = {
    "view": 1.0,
    "like": 1.5,
    "add_to_cart": 2.0,
    "purchase": 3.0,
}


class InteractionEvent(BaseModel):
    user_id: str
    product_id: str
    event_type: Literal["view", "like", "add_to_cart", "purchase"]


@router.post("")
async def log_interaction(event: InteractionEvent):
    """
    Log a user-product interaction. Automatically:
    - Persists to MongoDB (used for CF training)
    - Invalidates Redis recommendation cache for this user
      so they get fresh recs on next request
    """
    weight = EVENT_WEIGHTS[event.event_type]
    await mongo_service.log_interaction(
        user_id=event.user_id,
        product_id=event.product_id,
        event_type=event.event_type,
        weight=weight,
    )
    # Stale cache invalidation — key design decision
    await redis_service.invalidate_recommendations(event.user_id)

    return {
        "status": "logged",
        "user_id": event.user_id,
        "product_id": event.product_id,
        "event_type": event.event_type,
        "cache_invalidated": True,
    }
