"""ALS с пересчётом пользовательского вектора по текущей истории.

Fold-in использует то же линейное решение, что и implicit ALS:
  u = (YtY + alpha * Yp^T Yp + reg*I)^{-1} * (1+alpha) * Σ_{i∈prefix} y_i
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Один BLAS-поток делает обучение implicit воспроизводимым.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class ALSScorer:
    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.Y: np.ndarray | None = None
        self._YtY: np.ndarray | None = None
        self.pos: dict[int, int] = {}
        self._n_items = 0

    def fit(self, train_df: pd.DataFrame, catalog: np.ndarray) -> "ALSScorer":
        from implicit.als import AlternatingLeastSquares

        catalog = np.asarray(catalog)
        self.pos = {int(it): i for i, it in enumerate(catalog.tolist())}
        self._n_items = len(catalog)

        uniq = train_df[["user_id", "item_id"]].drop_duplicates()
        urow = uniq["user_id"].astype("category").cat.codes.to_numpy()
        icol = uniq["item_id"].map(self.pos).to_numpy()
        user_items = sp.csr_matrix(
            (np.ones(len(uniq), dtype=np.float32), (urow, icol)),
            shape=(int(urow.max()) + 1, self._n_items),
        )

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=self.seed,
            num_threads=1,
            calculate_training_loss=False,
        )
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(user_items, show_progress=False)
        self._set_factors(np.asarray(model.item_factors))
        return self

    def _set_factors(self, Y: np.ndarray) -> None:
        self.Y = Y.astype(np.float64)
        self._YtY = self.Y.T @ self.Y

    def score(self, user_id: int, prefix_items: Sequence[int]) -> np.ndarray:  # noqa: ARG002
        if self.Y is None:
            raise RuntimeError("ALSScorer.score вызван до fit()/load()")

        idx = list({self.pos[it] for it in prefix_items if it in self.pos})
        if not idx:
            return np.zeros(self._n_items, dtype=np.float64)

        Yp = self.Y[idx]
        a = self._YtY + self.alpha * (Yp.T @ Yp) + self.regularization * np.eye(self.factors)
        b = (1.0 + self.alpha) * Yp.sum(axis=0)
        u = np.linalg.solve(a, b)
        return self.Y @ u

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.Y)
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, catalog: np.ndarray, **hp):
        obj = cls(**hp)
        catalog = np.asarray(catalog)
        obj.pos = {int(it): i for i, it in enumerate(catalog.tolist())}
        obj._n_items = len(catalog)
        obj._set_factors(np.load(path.with_suffix(".npy")))
        return obj
