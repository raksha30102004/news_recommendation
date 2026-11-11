"""ALS collaborative filtering with category boosting (report §3.4, §4.8).

Training uses `implicit`'s AlternatingLeastSquares. Serving never needs per-user factors from training:
a user's vector is recomputed on demand from their click history with the exact ALS user-step solve
("fold-in"), so new users and new clicks are reflected instantly without retraining (report §4.9.1).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit is multithreaded itself; avoid oversubscription

import numpy as np
from scipy import sparse

from .config import ALS_PARAMS, DEFAULT_BOOST


def train_als(user_items: sparse.csr_matrix, params: dict | None = None) -> np.ndarray:
    """Fit ALS on the binary user x item matrix and return the item factor matrix (n_items x factors)."""
    from implicit.cpu.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(**(params or ALS_PARAMS))
    model.fit(user_items.tocsr().astype(np.float32), show_progress=False)
    return np.asarray(model.item_factors, dtype=np.float32)


class ALSScorer:
    def __init__(
        self,
        item_factors: np.ndarray,
        categories: np.ndarray,
        subcategories: np.ndarray,
        alpha: float = ALS_PARAMS["alpha"],
        regularization: float = ALS_PARAMS["regularization"],
        boost: dict | None = None,
    ):
        self.Y = item_factors.astype(np.float64)
        norms = np.linalg.norm(self.Y, axis=1)
        self.has_factors = norms > 0
        self.Yn = self.Y / np.where(norms > 0, norms, 1.0)[:, None]
        self.YtY = self.Y.T @ self.Y
        self.alpha = alpha
        self.reg = regularization
        self.cat = categories.astype(np.int64)
        self.sub = subcategories.astype(np.int64)
        self.n_cat = int(self.cat.max()) + 1
        self.n_sub = int(self.sub.max()) + 1
        self.boost = dict(boost or DEFAULT_BOOST)

    @property
    def n_items(self) -> int:
        return self.Y.shape[0]

    def user_vector(self, history: np.ndarray) -> np.ndarray | None:
        """Exact ALS user step: x_u = (YᵀY + Yᵀ(C_u − I)Y + λI)⁻¹ Yᵀ C_u p_u, with confidence c = alpha·r."""
        h = np.unique(history)
        h = h[self.has_factors[h]] if len(h) else h
        if len(h) == 0:
            return None
        Yh = self.Y[h]
        c = self.alpha
        A = self.YtY + (c - 1.0) * (Yh.T @ Yh) + self.reg * np.eye(self.Y.shape[1])
        b = c * Yh.sum(axis=0)
        return np.linalg.solve(A, b)

    def preferences(self, history: np.ndarray, interest_categories: np.ndarray | None = None, interest_weight: float = 3.0):
        """P(category | user) and P(subcategory | user) from clicks.

        Declared interests (onboarding) act as `interest_weight` pseudo-clicks spread over the chosen
        categories, so they dominate for brand-new users and fade as real clicks accumulate.
        """
        pc = np.bincount(self.cat[history], minlength=self.n_cat).astype(np.float64)
        ps = np.bincount(self.sub[history], minlength=self.n_sub).astype(np.float64)
        if interest_categories is not None and len(interest_categories):
            pc[interest_categories] += interest_weight / len(interest_categories)
        pc = pc / pc.sum() if pc.sum() else pc
        ps = ps / ps.sum() if ps.sum() else ps
        return pc, ps

    def score(
        self,
        history: np.ndarray,
        candidates: np.ndarray | None = None,
        interest_categories: np.ndarray | None = None,
        user_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        """cosine(user, item) + w_cat·P(cat|user) + w_sub·P(subcat|user) for each candidate."""
        cand = np.arange(self.n_items) if candidates is None else candidates
        u = user_vec if user_vec is not None else self.user_vector(history)
        cos = self.Yn[cand] @ (u / np.linalg.norm(u)) if u is not None else np.zeros(len(cand))
        pc, ps = self.preferences(history, interest_categories)
        return cos + self.boost["category"] * pc[self.cat[cand]] + self.boost["subcategory"] * ps[self.sub[cand]]

    def because(self, history: np.ndarray, item: int) -> int | None:
        """Most similar previously-clicked item (in ALS latent space) from the same subcategory, else the same
        category, used to explain a recommendation. None when no clicked item shares the category."""
        h = history[self.has_factors[history]] if len(history) else history
        if len(h) == 0 or not self.has_factors[item]:
            return None
        same_sub = h[self.sub[h] == self.sub[item]]
        h = same_sub if len(same_sub) else h[self.cat[h] == self.cat[item]]
        if len(h) == 0:
            return None
        sims = self.Yn[h] @ self.Yn[item]
        return int(h[int(np.argmax(sims))])
