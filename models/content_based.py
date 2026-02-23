"""
Content-Based Filtering using Sentence-Transformer embeddings.

Approach:
  - Each product has a precomputed embedding (stored in MongoDB)
  - A user's "taste vector" = mean of embeddings of their interacted products
  - Rank all products by cosine similarity to the taste vector
  - Return top-N candidates

This captures semantic similarity — e.g., "wireless headphones" ≈ "bluetooth earbuds"
even if they share no interaction history.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


class ContentBasedFilter:
    def __init__(self):
        self.product_embeddings: dict[str, np.ndarray] = {}  # product_id → vector
        self.product_matrix: Optional[np.ndarray] = None     # (n_products, dim)
        self.product_ids: list[str] = []

    def load_embeddings(self, products: list[dict]):
        """
        products: list of {product_id, embedding: list[float], ...}
        Call this at startup after fetching from MongoDB.
        """
        valid = [(p["product_id"], np.array(p["embedding"], dtype=np.float32))
                 for p in products if p.get("embedding")]

        if not valid:
            return

        self.product_ids = [pid for pid, _ in valid]
        embeddings = [emb for _, emb in valid]

        # Normalize once for fast cosine sim later
        matrix = np.stack(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid divide-by-zero
        self.product_matrix = matrix / norms
        self.product_embeddings = {pid: self.product_matrix[i]
                                   for i, pid in enumerate(self.product_ids)}

    def build_user_taste_vector(self,
                                 interacted_product_ids: list[str],
                                 weights: Optional[list[float]] = None) -> Optional[np.ndarray]:
        """
        Compute a weighted mean of product embeddings the user has interacted with.
        weights: if provided, higher-weight events (purchases) count more than views.
        """
        vecs = []
        ws = []
        for i, pid in enumerate(interacted_product_ids):
            if pid in self.product_embeddings:
                vecs.append(self.product_embeddings[pid])
                ws.append(weights[i] if weights else 1.0)

        if not vecs:
            return None

        ws_arr = np.array(ws, dtype=np.float32)
        taste = np.average(np.stack(vecs), axis=0, weights=ws_arr)
        norm = np.linalg.norm(taste)
        return taste / norm if norm > 0 else taste

    def get_candidates(self, taste_vector: np.ndarray,
                       top_n: int = 100,
                       exclude_ids: Optional[set] = None) -> list[tuple[str, float]]:
        """
        Returns [(product_id, cosine_similarity_score), ...] sorted descending.
        """
        if self.product_matrix is None or len(self.product_ids) == 0:
            return []

        # Batch cosine similarity (fast via matrix multiply — both normalized)
        scores = self.product_matrix @ taste_vector  # (n_products,)

        ranked = sorted(
            zip(self.product_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        if exclude_ids:
            ranked = [(pid, s) for pid, s in ranked if pid not in exclude_ids]

        return ranked[:top_n]
