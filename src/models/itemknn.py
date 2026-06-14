"""ItemKNN с косинусной близостью по совместным покупкам."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


class ItemKNNScorer:
    def __init__(self, k_neighbors: int = 200, aggregation: str = "sum") -> None:
        if aggregation not in ("sum", "recency_decay"):
            raise ValueError(f"aggregation должен быть sum|recency_decay, получено {aggregation!r}")
        self.k_neighbors = k_neighbors
        self.aggregation = aggregation
        self.sim: sp.csr_matrix | None = None
        self.pos: dict[int, int] = {}
        self._n_items = 0

    def fit(self, train_df: pd.DataFrame, catalog: np.ndarray) -> "ItemKNNScorer":
        catalog = np.asarray(catalog)
        self.pos = {int(it): i for i, it in enumerate(catalog.tolist())}
        self._n_items = len(catalog)

        uniq = train_df[["user_id", "item_id"]].drop_duplicates()
        urow = uniq["user_id"].astype("category").cat.codes.to_numpy()
        icol = uniq["item_id"].map(self.pos).to_numpy()
        X = sp.csr_matrix(
            (np.ones(len(uniq), dtype=np.float32), (urow, icol)),
            shape=(int(urow.max()) + 1, self._n_items),
        )

        # Диагональ X.T @ X содержит число пользователей для каждого товара.
        xtx = (X.T @ X).tocsr()
        cnt = xtx.diagonal()
        inv_norm = np.zeros_like(cnt)
        nz = cnt > 0
        inv_norm[nz] = 1.0 / np.sqrt(cnt[nz])

        d = sp.diags(inv_norm)
        sim = (d @ xtx @ d).tocsr()
        sim.setdiag(0.0)
        sim.eliminate_zeros()
        self.sim = _topk_per_row(sim, self.k_neighbors)
        return self

    def score(self, user_id: int, prefix_items: Sequence[int]) -> np.ndarray:  # noqa: ARG002
        if self.sim is None:
            raise RuntimeError("ItemKNNScorer.score вызван до fit()")

        idx, weights = self._prefix_weights(prefix_items)
        if not idx:
            return np.zeros(self._n_items, dtype=np.float64)

        p = sp.csr_matrix(
            (np.asarray(weights, dtype=np.float64), (np.zeros(len(idx), dtype=int), idx)),
            shape=(1, self._n_items),
        )
        return np.asarray((p @ self.sim).todense()).ravel()

    def _prefix_weights(self, prefix_items: Sequence[int]) -> tuple[list[int], list[float]]:
        """Преобразовать историю в индексы каталога и веса."""
        positions = [self.pos[it] for it in prefix_items if it in self.pos]
        if self.aggregation == "sum":
            seen: dict[int, float] = {}
            for p in positions:
                seen[p] = 1.0
            return list(seen.keys()), list(seen.values())

        seen = {}
        n = len(positions)
        for r, p in enumerate(positions):
            seen[p] = 0.85 ** (n - 1 - r)
        return list(seen.keys()), list(seen.values())

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path.with_suffix(".npz"), self.sim)
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, catalog: np.ndarray, k_neighbors: int, aggregation: str):
        obj = cls(k_neighbors=k_neighbors, aggregation=aggregation)
        catalog = np.asarray(catalog)
        obj.pos = {int(it): i for i, it in enumerate(catalog.tolist())}
        obj._n_items = len(catalog)
        obj.sim = sp.load_npz(path.with_suffix(".npz")).tocsr()
        return obj


def _topk_per_row(sim: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Оставить top-k соседей в каждой строке."""
    sim = sim.tocsr()
    rows, cols, data = [], [], []
    indptr, indices, values = sim.indptr, sim.indices, sim.data
    for i in range(sim.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end - start <= k:
            sel = slice(start, end)
            cols.extend(indices[sel]); data.extend(values[sel])
            rows.extend([i] * (end - start))
            continue
        local = np.argpartition(values[start:end], -k)[-k:]
        cols.extend(indices[start:end][local])
        data.extend(values[start:end][local])
        rows.extend([i] * k)
    return sp.csr_matrix((data, (rows, cols)), shape=sim.shape)
