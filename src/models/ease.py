"""EASE: линейная item-item модель с замкнутым решением.

  G = XᵀX (item-item Gram);  P = (G + λI)⁻¹;  B = -P / diag(P), diag(B)=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def compute_ease_B(X: sp.csr_matrix, lam: float) -> np.ndarray:
    """Вычислить матрицу весов EASE для бинарной user-item матрицы."""
    G = (X.T @ X).toarray().astype(np.float64)
    diag_idx = np.diag_indices(G.shape[0])
    G[diag_idx] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[diag_idx] = 0.0
    return B


def topk_sparse(B: np.ndarray, k: int) -> sp.csr_matrix:
    """Оставить top-k значений в каждой строке."""
    n = B.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        r = B[i]
        if n > k:
            idx = np.argpartition(r, -k)[-k:]
        else:
            idx = np.arange(n)
        rows.extend([i] * len(idx)); cols.extend(idx.tolist()); data.extend(r[idx].tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class EASEScorer:
    def __init__(self, B, catalog: np.ndarray) -> None:
        self.B = B
        self._sparse = sp.issparse(B)
        if self._sparse:
            self.B = self.B.tocsr()
        catalog = np.asarray(catalog)
        self.pos = {int(c): i for i, c in enumerate(catalog.tolist())}
        self.n_items = len(catalog)

    def score(self, user_id: int, prefix_items) -> np.ndarray:  # noqa: ARG002
        idx = list({self.pos[i] for i in prefix_items if i in self.pos})
        if not idx:
            return np.zeros(self.n_items, dtype=np.float64)
        if self._sparse:
            return np.asarray(self.B[idx].sum(axis=0)).ravel().astype(np.float64)
        return self.B[idx].astype(np.float64).sum(axis=0)

    def score_batch(self, batch):
        return [self.score(u, p) for u, p in batch]


def load_real_ease(ease_dir, catalog, expected_hash: str | None = None) -> "EASEScorer":
    """Загрузить разреженную или плотную матрицу весов."""
    ease_dir = Path(ease_dir)
    meta_path = ease_dir / "meta.json"
    if expected_hash is not None and meta_path.exists():
        meta = json.load(meta_path.open(encoding="utf-8"))
        assert meta.get("catalog_hash") == expected_hash, (
            "catalog_hash EASE-чекпойнта != текущего V_train"
        )
    topk = ease_dir / "ease_B_topk.npz"
    dense = ease_dir / "ease_B.npy"
    if topk.exists():
        B = sp.load_npz(topk)
    elif dense.exists():
        B = np.load(dense)
    else:
        raise RuntimeError(f"в {ease_dir} нет ease_B_topk.npz или ease_B.npy")
    return EASEScorer(B, catalog)
