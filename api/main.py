"""
SmartCart API — main FastAPI application entrypoint.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.mongo_service import mongo_service
from services.redis_service import redis_service
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedFilter
from api.routes import recommendations, products, interactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared ML model instances (loaded at startup)
cf_model = CollaborativeFilter(n_components=10)
cb_model = ContentBasedFilter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to DBs, load and train models. Shutdown: clean up."""
    logger.info("🚀 Starting SmartCart API...")

    # Connect to infrastructure
    await mongo_service.connect()
    await redis_service.connect()
    logger.info("✅ Connected to MongoDB and Redis")

    # Load product embeddings for content-based filtering
    products_with_embeddings = await mongo_service.get_all_products_with_embeddings()
    cb_model.load_embeddings(products_with_embeddings)
    logger.info(f"✅ Loaded {len(products_with_embeddings)} product embeddings")

    # Train collaborative filter from interaction history
    # Check Redis cache first to avoid retraining on every restart
    cached_model = await redis_service.get_cf_model()
    if cached_model:
        app.state.cf_model = cached_model
        logger.info("✅ Loaded CF model from Redis cache")
    else:
        interaction_matrix = await mongo_service.get_interaction_matrix()
        if interaction_matrix:
            cf_model.train(interaction_matrix)
            await redis_service.cache_cf_model(cf_model, ttl=3600)
            logger.info(f"✅ Trained CF model on {len(interaction_matrix)} users")
        else:
            logger.warning("⚠️  No interaction data — CF model untrained (cold start)")

    # Attach models to app state so routes can access them
    app.state.cf_model = cf_model
    app.state.cb_model = cb_model

    yield  # App runs here

    # Shutdown
    await mongo_service.disconnect()
    await redis_service.disconnect()
    logger.info("👋 SmartCart API shut down cleanly")


app = FastAPI(
    title="SmartCart Recommender API",
    description="LLM-enhanced hybrid e-commerce recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
app.include_router(products.router, prefix="/products", tags=["Products"])
app.include_router(interactions.router, prefix="/interactions", tags=["Interactions"])


@app.get("/health")
async def health():
    redis_ok = await redis_service.ping()
    cache_hits = await redis_service.get_counter("cache_hits")
    cache_misses = await redis_service.get_counter("cache_misses")
    hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0

    return {
        "status": "healthy",
        "redis": "ok" if redis_ok else "unreachable",
        "mongo": "ok",
        "cf_model_trained": app.state.cf_model.is_trained,
        "products_in_cb_index": len(app.state.cb_model.product_ids),
        "cache_hit_rate": f"{hit_rate:.1%}",
    }
