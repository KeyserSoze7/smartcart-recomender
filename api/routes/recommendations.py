"""
Recommendations routes — orchestrates the 3-stage pipeline:
  Stage 1: CF + CB candidate generation
  Stage 2: Redis cache check
  Stage 3: LLM re-ranking
"""

import time
import logging
from fastapi import APIRouter, Request, Query, HTTPException

from services.mongo_service import mongo_service
from services.redis_service import redis_service
from services.llm_reranker import llm_reranker
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{user_id}")
async def get_recommendations(
    user_id: str,
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
    use_llm: bool = Query(default=True, description="Enable LLM re-ranking"),
    force_refresh: bool = Query(default=False, description="Bypass cache"),
):
    """
    Get personalized product recommendations for a user.

    Pipeline:
      1. Check Redis cache → return immediately if hit
      2. Generate CF candidates (collaborative filtering)
      3. Generate CB candidates (content-based / embedding similarity)
      4. Merge and deduplicate candidates
      5. LLM re-rank top-20 candidates
      6. Cache results in Redis
      7. Return top-N
    """
    start_time = time.monotonic()
    cf_model = request.app.state.cf_model
    cb_model = request.app.state.cb_model

    # ── Stage 2: Cache check ───────────────────────────────────────────────────
    if not force_refresh:
        cached = await redis_service.get_cached_recommendations(user_id)
        if cached:
            await redis_service.increment_counter("cache_hits")
            return {
                "user_id": user_id,
                "recommendations": cached[:limit],
                "cached": True,
                "latency_ms": round((time.monotonic() - start_time) * 1000, 1),
            }

    await redis_service.increment_counter("cache_misses")

    # ── Fetch user data ────────────────────────────────────────────────────────
    user_profile = await mongo_service.get_user(user_id)
    if not user_profile:
        # Auto-create anonymous profile for new users
        user_profile = {"user_id": user_id, "name": "New User", "preferences": {}}

    user_history = await mongo_service.get_user_history(user_id, limit=30)
    seen_pids = {h["product_id"] for h in user_history}

    # ── Stage 1: Candidate generation ─────────────────────────────────────────
    cf_candidates: list[tuple[str, float]] = []
    cb_candidates: list[tuple[str, float]] = []

    # Collaborative filtering candidates
    if cf_model.is_trained:
        cf_candidates = cf_model.get_candidates(
            user_id,
            top_n=settings.candidate_pool_size,
            exclude_seen=True,
            seen_products=seen_pids,
        )

    # Content-based candidates
    if user_history and cb_model.product_matrix is not None:
        history_pids = [h["product_id"] for h in user_history]
        weights = [_event_weight(h["event_type"]) for h in user_history]
        taste_vec = cb_model.build_user_taste_vector(history_pids, weights)
        if taste_vec is not None:
            cb_candidates = cb_model.get_candidates(
                taste_vec,
                top_n=settings.candidate_pool_size,
                exclude_ids=seen_pids,
            )

    # Merge: take union, prefer higher score when both models agree
    merged: dict[str, float] = {}
    for pid, score in cf_candidates:
        merged[pid] = merged.get(pid, 0) + score * 0.5  # CF weight
    for pid, score in cb_candidates:
        merged[pid] = merged.get(pid, 0) + score * 0.5  # CB weight

    if not merged:
        # Cold-start fallback: return popular products
        popular = await mongo_service.get_all_products_with_embeddings()
        return {
            "user_id": user_id,
            "recommendations": popular[:limit],
            "cached": False,
            "note": "Cold start — no interaction history",
            "latency_ms": round((time.monotonic() - start_time) * 1000, 1),
        }

    # Sort merged candidates and take top candidates for LLM
    top_candidate_ids = sorted(merged, key=merged.get, reverse=True)[:settings.llm_input_size]

    # Fetch full product details for candidates
    candidate_products = await mongo_service.get_products_by_ids(top_candidate_ids)
    # Add pre-LLM scores to products
    for p in candidate_products:
        p["pre_llm_score"] = merged.get(p["product_id"], 0.0)
    candidate_products.sort(key=lambda x: x["pre_llm_score"], reverse=True)

    # ── Stage 3: LLM re-ranking ────────────────────────────────────────────────
    history_product_ids = [h["product_id"] for h in user_history]
    history_products = await mongo_service.get_products_by_ids(history_product_ids[:10])

    if use_llm and settings.openai_api_key:
        final_recommendations = await llm_reranker.rerank(
            candidates=candidate_products,
            user_profile=user_profile,
            user_history=user_history,
            history_products=history_products,
            top_n=limit,
        )
    else:
        final_recommendations = candidate_products[:limit]
        for r in final_recommendations:
            r["stage"] = "no_llm"

    # ── Cache and return ───────────────────────────────────────────────────────
    await redis_service.cache_recommendations(user_id, final_recommendations)

    latency = round((time.monotonic() - start_time) * 1000, 1)
    logger.info(f"Recommendations for {user_id}: {len(final_recommendations)} items, {latency}ms")

    return {
        "user_id": user_id,
        "recommendations": final_recommendations,
        "cached": False,
        "latency_ms": latency,
    }


@router.get("/{user_id}/similar/{product_id}")
async def get_similar_products(
    user_id: str,
    product_id: str,
    request: Request,
    limit: int = Query(default=5, ge=1, le=20),
):
    """Find products similar to a given product using content-based similarity."""
    cb_model = request.app.state.cb_model

    product = await mongo_service.get_product(product_id)
    if not product or not product.get("embedding"):
        raise HTTPException(status_code=404, detail="Product not found or missing embedding")

    import numpy as np
    product_vec = np.array(product["embedding"], dtype=np.float32)
    norm = np.linalg.norm(product_vec)
    if norm > 0:
        product_vec /= norm

    similar = cb_model.get_candidates(product_vec, top_n=limit + 1,
                                       exclude_ids={product_id})
    similar_ids = [pid for pid, _ in similar[:limit]]
    similar_products = await mongo_service.get_products_by_ids(similar_ids)

    return {"source_product_id": product_id, "similar_products": similar_products}


def _event_weight(event_type: str) -> float:
    weights = {"purchase": 3.0, "add_to_cart": 2.0, "like": 1.5, "view": 1.0}
    return weights.get(event_type, 1.0)
