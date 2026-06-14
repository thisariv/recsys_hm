"""BPR matrix factorization на базе библиотеки implicit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


class BPRScorer:
    def __init__(self, factors: int = 128, regularization: float = 0.01,
                 iterations: int = 100, learning_rate: float = 0.01, seed: int = 42) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.n_items = 0

    def fit(self, train_df: pd.DataFrame, catalog: np.ndarray) -> "BPRScorer":
        from implicit.bpr import BayesianPersonalizedRanking

        catalog = np.asarray(catalog)
        self.n_items = len(catalog)
        pos = {int(it): i for i, it in enumerate(catalog.tolist())}
        uniq = train_df[["user_id", "item_id"]].drop_duplicates()
        urow = uniq["user_id"].to_numpy()
        icol = uniq["item_id"].map(pos).to_numpy()
        user_items = sp.csr_matrix(
            (np.ones(len(uniq), dtype=np.float32), (urow, icol)),
            shape=(int(urow.max()) + 1, self.n_items),
        )
        model = BayesianPersonalizedRanking(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, learning_rate=self.learning_rate,
            random_state=self.seed, num_threads=1,
        )
        model.fit(user_items, show_progress=False)
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def score(self, user_id: int, prefix_items=None) -> np.ndarray:  # noqa: ARG002
        if self.user_factors is None:
            raise RuntimeError("BPRScorer.score вызван до fit()/load()")
        if user_id >= len(self.user_factors):
            return np.zeros(self.n_items, dtype=np.float64)
        return (self.user_factors[user_id] @ self.item_factors.T).astype(np.float64)

    def score_batch(self, batch):
        return [self.score(u, p) for u, p in batch]

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".user.npy"), self.user_factors)
        np.save(path.with_suffix(".item.npy"), self.item_factors)
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, catalog: np.ndarray, **hp) -> "BPRScorer":
        obj = cls(**hp)
        obj.n_items = len(np.asarray(catalog))
        obj.user_factors = np.load(path.with_suffix(".user.npy"))
        obj.item_factors = np.load(path.with_suffix(".item.npy"))
        return obj
