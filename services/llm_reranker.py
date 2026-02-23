"""
LLM Re-ranker — the intelligence layer on top of CF + CB candidates.

This is the key differentiator of this system. Rather than using LLMs to generate
recommendations (slow, expensive), we use them to re-rank a small pre-filtered
candidate set — exactly how production systems at Netflix, Spotify, and Amazon work.

Prompt engineering techniques used:
  1. Structured role + context priming
  2. User preference summarization from interaction history
  3. Chain-of-thought reasoning for scoring
  4. JSON-mode output for reliable parsing
  5. Few-shot implicit guidance via score format spec
"""

import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a personalization engine for an e-commerce platform.
Your job is to re-rank a list of product candidates for a specific user based on their 
browsing and purchase history.

You must return a JSON array — nothing else. Each element must have:
  - "product_id": string
  - "score": float between 0 and 1 (higher = better fit for this user)
  - "reasoning": one sentence explaining the match

Rank by predicted relevance to this user's taste. Be precise and consistent.
Do NOT include products not in the input list."""


def _build_user_context(user_profile: dict, history: list[dict],
                         history_products: list[dict]) -> str:
    """Builds a concise, information-dense user context string for the prompt."""
    name = user_profile.get("name", "Unknown User")
    prefs = user_profile.get("preferences", {})
    fav_cats = prefs.get("favorite_categories", [])
    price_range = prefs.get("price_range", "unknown")

    # Summarize recent interactions
    event_weights = {"purchase": 3, "add_to_cart": 2, "like": 1.5, "view": 1}
    history_lines = []
    pid_to_product = {p["product_id"]: p for p in history_products}
    for h in history[:10]:
        pid = h["product_id"]
        prod = pid_to_product.get(pid, {})
        name_str = prod.get("name", pid)
        event = h.get("event_type", "view")
        history_lines.append(f"  - {event.upper()}: {name_str}")

    context = f"""USER PROFILE:
  Name: {name}
  Favorite categories: {', '.join(fav_cats) if fav_cats else 'not set'}
  Preferred price range: {price_range}

RECENT ACTIVITY (most recent first):
{chr(10).join(history_lines) if history_lines else '  (no history — new user)'}"""
    return context


def _build_candidates_context(candidates: list[dict]) -> str:
    """Format candidate products for the prompt."""
    lines = []
    for p in candidates:
        tags = ", ".join(p.get("tags", []))
        lines.append(
            f"  - product_id: {p['product_id']} | "
            f"name: {p.get('name', 'N/A')} | "
            f"category: {p.get('category', 'N/A')} | "
            f"price: ${p.get('price', 0):.2f} | "
            f"tags: {tags}"
        )
    return "\n".join(lines)


class LLMReranker:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def rerank(
        self,
        candidates: list[dict],
        user_profile: dict,
        user_history: list[dict],
        history_products: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """
        candidates: list of product dicts (product_id, name, category, price, tags)
        Returns top_n candidates re-ranked by LLM with explanations.
        Falls back to input order on any error.
        """
        if not candidates:
            return []

        if not settings.openai_api_key:
            logger.warning("No OpenAI API key — skipping LLM re-ranking.")
            return candidates[:top_n]

        user_ctx = _build_user_context(user_profile, user_history, history_products)
        candidates_ctx = _build_candidates_context(candidates)

        user_message = f"""{user_ctx}

CANDIDATE PRODUCTS TO RANK:
{candidates_ctx}

Return a JSON array of all {len(candidates)} products, ranked from most to least relevant 
for this user, with a score (0-1) and one-sentence reasoning for each."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,  # Low temp for consistent, deterministic ranking
                max_tokens=1500,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Handle both {"rankings": [...]} and direct [...] responses
            if isinstance(parsed, dict):
                ranked_list = parsed.get("rankings") or parsed.get("products") or list(parsed.values())[0]
            else:
                ranked_list = parsed

            # Merge LLM scores back into full product dicts
            pid_to_llm = {r["product_id"]: r for r in ranked_list}
            pid_to_product = {c["product_id"]: c for c in candidates}

            results = []
            for item in ranked_list[:top_n]:
                pid = item.get("product_id")
                if pid and pid in pid_to_product:
                    product = dict(pid_to_product[pid])
                    product["llm_score"] = item.get("score", 0.0)
                    product["reasoning"] = item.get("reasoning", "")
                    product["stage"] = "llm_reranked"
                    results.append(product)

            return results

        except Exception as e:
            logger.error(f"LLM re-ranking failed: {e}. Falling back to input order.")
            for c in candidates:
                c["stage"] = "fallback_no_llm"
            return candidates[:top_n]


# Module-level singleton
llm_reranker = LLMReranker()
