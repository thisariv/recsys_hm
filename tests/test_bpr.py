"""Тесты BPR matrix factorization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.bpr import BPRScorer

CATALOG = np.array([10, 20, 30, 40, 50, 60])


def _toy_train():
    rows = []
    for u in range(8):
        rows += [(u, 10), (u, 20), (u, 30)]
    for u in range(8, 16):
        rows += [(u, 40), (u, 50), (u, 60)]
    return pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(t_dat=pd.Timestamp("2020-01-01"))


def test_scoring_shape_and_dot():
    uf = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    itf = np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1]]
    s = BPRScorer.__new__(BPRScorer)
    s.user_factors, s.item_factors, s.n_items = uf, itf, 6
    out = s.score(0, [])
    assert out.shape == (6,)
    assert np.allclose(out, uf[0] @ itf.T)


def test_cold_user_zero():
    s = BPRScorer.__new__(BPRScorer)
    s.user_factors = np.zeros((3, 2), np.float32); s.item_factors = np.zeros((6, 2), np.float32); s.n_items = 6
    assert np.all(s.score(999, []) == 0.0)


def test_fit_separates_clusters():
    s = BPRScorer(factors=16, iterations=150, seed=42).fit(_toy_train(), CATALOG)
    assert s.user_factors.shape[1] == s.item_factors.shape[1]
    pos = {c: i for i, c in enumerate(CATALOG.tolist())}
    sc = s.score(0, [])
    mean_A = np.mean([sc[pos[i]] for i in (10, 20, 30)])
    mean_B = np.mean([sc[pos[i]] for i in (40, 50, 60)])
    assert mean_A > mean_B
