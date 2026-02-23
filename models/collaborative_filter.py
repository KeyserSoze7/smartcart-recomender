"""
Collaborative Filtering model using truncated SVD (Matrix Factorization).

Approach:
  - Build a user-item interaction matrix from MongoDB events
  - Apply TruncatedSVD to decompose into latent factors
  - For a target user, compute cosine similarity with all items in latent space
  - Return top-N product candidates

This is the same core idea used by Netflix's early recommendation system.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import Optional


class CollaborativeFilter:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors: Optional[np.ndarray] = None   # (n_users, k)
        self.item_factors: Optional[np.ndarray] = None   # (n_items, k)
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.item_ids: list[str] = []
        self.is_trained = False

    def train(self, interaction_matrix: dict[str, dict[str, float]]):
        """
        interaction_matrix: {user_id: {product_id: weight}}
        Builds the sparse matrix and fits SVD.
        """
        all_users = list(interaction_matrix.keys())
        all_items = list({pid for ui in interaction_matrix.values() for pid in ui})

        self.user_index = {uid: i for i, uid in enumerate(all_users)}
        self.item_index = {pid: i for i, pid in enumerate(all_items)}
        self.item_ids = all_items

        # Build dense interaction matrix (for small datasets; use scipy sparse for scale)
        matrix = np.zeros((len(all_users), len(all_items)), dtype=np.float32)
        for uid, items in interaction_matrix.items():
            for pid, weight in items.items():
                matrix[self.user_index[uid], self.item_index[pid]] = weight

        # Fit SVD
        self.user_factors = self.svd.fit_transform(matrix)  # (n_users, k)
        self.item_factors = self.svd.components_.T          # (n_items, k)

        # L2-normalize for cosine similarity via dot product
        self.user_factors = normalize(self.user_factors)
        self.item_factors = normalize(self.item_factors)

        self.is_trained = True

    def get_candidates(self, user_id: str, top_n: int = 100,
                       exclude_seen: bool = True,
                       seen_products: Optional[set] = None) -> list[tuple[str, float]]:
        """
        Returns [(product_id, score), ...] sorted by predicted preference.
        Falls back to item popularity if user is unseen (cold-start).
        """
        if not self.is_trained:
            return []

        if user_id not in self.user_index:
            # Cold-start: return globally popular items (by index order as proxy)
            return [(pid, 1.0 / (i + 1)) for i, pid in enumerate(self.item_ids[:top_n])]

        user_vec = self.user_factors[self.user_index[user_id]]
        # Cosine similarity = dot product (both are L2-normalized)
        scores = self.item_factors @ user_vec  # (n_items,)

        ranked = sorted(
            zip(self.item_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        if exclude_seen and seen_products:
            ranked = [(pid, s) for pid, s in ranked if pid not in seen_products]

        return ranked[:top_n]
