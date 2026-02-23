# SmartCart — LLM-Enhanced E-commerce Recommender System

A production-grade hybrid recommendation engine that combines **Collaborative Filtering**, **Content-Based Filtering**, and **LLM re-ranking** to deliver personalized product recommendations. Built with FastAPI, MongoDB, and Redis.

---

## What Does This Project Do?

Imagine you're building Amazon. You have 10,000 products and a user visits your homepage. You can't show them everything — you need to pick the 10 most relevant products *for that specific user*. How?

That's exactly what SmartCart solves. It looks at what a user has viewed, liked, added to cart, and purchased — and uses that behavioral data to predict what they'd want to see next.

---

## How It Works — The 3-Stage Pipeline

Every time a user requests recommendations, the system runs through three stages:

```
User Request
     │
     ▼
┌─────────────────────────────┐
│  Stage 2: Redis Cache Check  │ ──── Cache HIT ──→ Return in < 5ms
└──────────────┬──────────────┘
               │ Cache MISS
               ▼
┌─────────────────────────────┐
│  Stage 1: Candidate Gen      │
│  ┌─────────────────────┐    │
│  │ Collaborative Filter │    │  ← "Users like you also bought..."
│  │ top-100 candidates   │    │
│  └─────────────────────┘    │
│  ┌─────────────────────┐    │
│  │ Content-Based Filter │    │  ← "Similar to products you liked..."
│  │ top-100 candidates   │    │
│  └─────────────────────┘    │
│         Merge → top-20       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3: LLM Re-ranking     │  ← GPT-4o-mini picks the best 10
└──────────────┬──────────────┘
               │
               ▼
        Cache in Redis
        Return top-10
```

### Stage 1a — Collaborative Filtering (CF)

**Core idea:** *"Find users who behave like you, then recommend what they liked."*

CF builds a matrix where:
- Each **row** is a user
- Each **column** is a product
- Each **cell** is how strongly that user interacted with that product (purchases = 3pts, add to cart = 2pts, likes = 1.5pts, views = 1pt)

This matrix is huge and mostly empty (sparse). We compress it using **SVD (Singular Value Decomposition)** — a mathematical technique that finds hidden patterns. Instead of storing "Alice bought headphones, a laptop stand, and a keyboard," SVD learns "Alice scores 0.9 on the productivity dimension and 0.7 on the audio dimension." These are called **latent factors**.

To find recommendations, we compute the cosine similarity between Alice's latent vector and every product's latent vector — the closer they are, the more likely Alice would like that product.

**Weakness:** Fails for new users with no history (cold-start problem).

### Stage 1b — Content-Based Filtering (CB)

**Core idea:** *"Find products semantically similar to what you've already liked."*

CB ignores other users entirely. Instead it looks at the products themselves. Each product is converted into a **vector embedding** using `BAAI/bge-small-en-v1.5` (via FastEmbed) — a model that reads the product's name, category, and tags and produces a 384-dimensional vector capturing its meaning.

For example:
- "Wireless noise-cancelling headphones" → vector A
- "Bluetooth earbuds" → vector B
- "Running shoes" → vector C

Vectors A and B will be very close together. Vector C will be far away from both.

A user's taste is the **weighted average** of the vectors of everything they've interacted with. Then we rank all products by cosine similarity to that taste vector.

**Weakness:** Creates filter bubbles — if you only buy audio gear, you only get recommended audio gear.

### Why Both Together?

| | CF | CB |
|---|---|---|
| New users | Fails (no history) | Works |
| Cross-category discovery | Great | Poor |
| Filter bubbles |  Avoids them | Creates them |

Using both covers each other's weaknesses. This is how Spotify, Netflix, and Amazon do it.

### Stage 2 — Redis Cache

Before doing any ML, the system checks Redis: *"Have we computed recommendations for this user in the last 10 minutes?"*

Redis is an **in-memory key-value store** — think of it as a giant dictionary that lives in RAM. The key is `rec:{user_id}`, the value is a JSON blob of recommendations, and it auto-expires after 10 minutes (TTL = Time To Live).

When a user logs a new interaction (buys something, clicks something), their cache entry is immediately deleted so the next request recomputes fresh. This pattern is called **cache invalidation on write**.

Why does this matter? Computing CF + CB takes ~100-200ms. Serving from cache takes < 5ms. In a real system with millions of users, a 90%+ cache hit rate is what makes the whole thing economically viable.

### Stage 3 — LLM Re-ranking

The top 20 candidates from Stage 1 are passed to **GPT-4o-mini** along with the user's profile and interaction history. The LLM scores each candidate 0-1 and writes a one-sentence reasoning for each.

**Critical design decision:** The LLM never searches the full catalog. It only re-ranks 20 pre-filtered candidates. Calling GPT on 10,000 products would take minutes and cost a fortune. Calling it on 20 takes ~500ms and costs fractions of a cent. This is how production systems actually use LLMs in recommendations.

The prompt uses four engineering techniques:
1. **Structured role priming** — "You are a personalization engine..."
2. **Dense user context** — profile + last 10 interactions
3. **Chain-of-thought scoring** — model must give score + reasoning (forces better decisions)
4. **JSON-mode output** — guarantees parseable responses every time

---

## Project Structure

```
smartcart-recommender/
│
├── api/
│   ├── main.py                  # FastAPI app — startup, DB connections, model loading
│   └── routes/
│       ├── recommendations.py   # Core pipeline — orchestrates all 3 stages
│       ├── products.py          # Product catalog endpoints
│       └── interactions.py      # Logs user events, invalidates cache
│
├── models/
│   ├── collaborative_filter.py  # SVD-based CF — training + candidate scoring
│   └── content_based.py         # Embedding-based CB — cosine similarity ranking
│
├── services/
│   ├── mongo_service.py         # All MongoDB reads/writes (async Motor driver)
│   ├── redis_service.py         # Cache, session store, model cache
│   └── llm_reranker.py          # Builds prompts, calls OpenAI, parses response
│
├── data/
│   └── seed_data.py             # Seeds MongoDB with 20 products, 4 users, 24 interactions
│
├── scripts/
│   └── generate_embeddings.py   # Computes product embeddings, stores in MongoDB
│
├── config/
│   └── settings.py              # Pydantic settings — reads from .env file
│
├── docker-compose.yml           # Spins up MongoDB + Redis + API
├── requirements.txt
└── .env.example
```

---

## Data Layer

### MongoDB Collections

**`products`** — the product catalog
```json
{
  "product_id": "prod_001",
  "name": "Sony WH-1000XM5 Headphones",
  "category": "Electronics",
  "price": 349.99,
  "tags": ["wireless", "noise-cancelling", "audio", "bluetooth"],
  "embedding": [0.023, -0.14, ...]  // 384-dim vector, stored after generate_embeddings.py
}
```

**`users`** — user profiles
```json
{
  "user_id": "user_001",
  "name": "Alice Chen",
  "preferences": {
    "favorite_categories": ["Electronics", "Books"],
    "price_range": "$50-$300"
  }
}
```

**`interactions`** — event log (what trains the CF model)
```json
{
  "user_id": "user_001",
  "product_id": "prod_001",
  "event_type": "purchase",
  "weight": 3.0,
  "timestamp": "2026-01-15T10:30:00Z"
}
```

### Redis Key Schema

| Key | Value | TTL |
|-----|-------|-----|
| `rec:{user_id}` | JSON recommendations list | 10 min |
| `session:{user_id}` | User session data | 30 min |
| `cf_model` | Serialized trained CF model | 1 hour |

---

## Quick Start

### Prerequisites
- Python 3.12
- Docker + Docker Compose

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/smartcart-recommender
cd smartcart-recommender
```

### 2. Start MongoDB and Redis
```bash
docker-compose up -d mongo redis
```

### 3. Set up Python environment
```bash
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key (optional — system works without it, just skips LLM re-ranking):
```
OPENAI_API_KEY=sk-...        # optional
MONGO_URI=mongodb://localhost:27017/smartcart
REDIS_URI=redis://localhost:6379
```

### 5. Seed the database
```bash
python data/seed_data.py
```

### 6. Generate product embeddings
```bash
python scripts/generate_embeddings.py
```
This downloads `BAAI/bge-small-en-v1.5` (~130MB) on first run and stores 384-dim vectors for all 20 products in MongoDB.

### 7. Start the API
```bash
uvicorn api.main:app --reload
```

You should see:
```
 Connected to MongoDB and Redis
 Loaded 20 product embeddings
 Trained CF model on 4 users
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

## API Reference

### Get recommendations
```bash
GET /recommendations/{user_id}
```
```bash
curl http://localhost:8000/recommendations/user_001 | python3 -m json.tool
```

**Response:**
```json
{
  "user_id": "user_001",
  "recommendations": [
    {
      "product_id": "prod_007",
      "name": "Mechanical Keyboard TKL RGB",
      "category": "Electronics",
      "price": 89.99,
      "llm_score": 0.91,
      "reasoning": "Matches user's pattern of buying productivity-focused peripherals at mid-range prices",
      "stage": "llm_reranked"
    }
  ],
  "cached": false,
  "latency_ms": 183.4
}
```

### Get similar products
```bash
GET /recommendations/{user_id}/similar/{product_id}
```
```bash
curl http://localhost:8000/recommendations/user_001/similar/prod_001 | python3 -m json.tool
```

### Log a user interaction
```bash
POST /interactions
```
```bash
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "product_id": "prod_005", "event_type": "purchase"}'
```
Event types: `view`, `like`, `add_to_cart`, `purchase`

### List products
```bash
GET /products?category=Electronics&limit=10
```

### Health check
```bash
curl http://localhost:8000/health | python3 -m json.tool
```
```json
{
  "status": "healthy",
  "redis": "ok",
  "mongo": "ok",
  "cf_model_trained": true,
  "products_in_cb_index": 20,
  "cache_hit_rate": "87.3%"
}
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | Async, fast, auto-generates docs at `/docs` |
| Database | MongoDB + Motor | Flexible schema, async driver for non-blocking I/O |
| Cache / Session | Redis | In-memory, sub-ms reads, built-in TTL |
| CF Model | scikit-learn SVD | Matrix factorization for latent factor discovery |
| Embeddings | FastEmbed (bge-small-en-v1.5) | Lightweight, no PyTorch needed, 384-dim vectors |
| LLM Re-ranker | GPT-4o-mini | Fast, cheap, smart final ranking layer |
| Infrastructure | Docker Compose | One command to spin up all services |

---

## Key Design Decisions

**Why re-rank with LLM instead of generate?**
Generation means calling the LLM against your entire catalog — slow and expensive. Re-ranking means calling it on 20 pre-filtered candidates — fast and cheap. The heavy lifting is done by fast ML models; LLM only adds the final personalization layer.

**Why store embeddings in MongoDB instead of a vector DB?**
For a catalog of this size, MongoDB is sufficient. At scale (millions of products), you'd swap this for Pinecone or Weaviate for approximate nearest-neighbor search.

**Why async everywhere?**
FastAPI runs on Python's asyncio event loop. Using blocking I/O (regular PyMongo, regular Redis) would kill concurrency. Motor and redis-asyncio let thousands of requests run concurrently on a single process.

**Why pickle the CF model into Redis?**
SVD training reads the entire interaction matrix and does heavy linear algebra. Doing this on every server restart wastes 2-3 seconds. Caching the trained model means instant startup after the first training run.

---

## Scaling This Further

This project is intentionally scoped as a working MVP. Here's how you'd scale each piece in production:

- **More data** → CF model accuracy improves dramatically with 10K+ users
- **Vector search** → Replace MongoDB embedding lookup with Pinecone/Weaviate for ANN search
- **Model retraining** → Add a scheduled job (Celery + Redis) to retrain CF nightly
- **A/B testing** → Run CF-only vs hybrid vs LLM re-ranked in parallel, measure CTR
- **Cold start** → Use demographic data (age, location) for brand new users
- **Big data pipeline** → Replace in-memory matrix with Spark for distributed SVD at scale

---
