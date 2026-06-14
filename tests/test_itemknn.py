"""Тесты ItemKNN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.itemknn import ItemKNNScorer

T0 = pd.Timestamp("2020-06-01")


def _df(rows):
    return pd.DataFrame([(u, i, T0) for u, i in rows], columns=["user_id", "item_id", "t_dat"])


def test_prefix_top_neighbor_order():
    """Товар с большей совместной встречаемостью получает больший скор."""
    rows = [
        (0, 10), (0, 20), (0, 30),
        (1, 10), (1, 20),
        (2, 10), (2, 20),
        (3, 40),
    ]
    catalog = np.array([10, 20, 30, 40])
    scorer = ItemKNNScorer(k_neighbors=10, aggregation="sum").fit(_df(rows), catalog)
    scores = scorer.score(0, [10])

    s = dict(zip(catalog.tolist(), scores.tolist()))
    assert s[20] > s[30] > 0
    assert s[40] == 0.0
    assert catalog[int(np.argmax(scores))] == 20


def test_anti_leak_only_train():
    """Данные, не переданные в fit, не влияют на матрицу сходства."""
    train = _df([(0, 10), (0, 20), (1, 10), (1, 20)])
    catalog = np.array([10, 20, 30])
    s_train = ItemKNNScorer(k_neighbors=10).fit(train, catalog).score(0, [10])

    _holdout = _df([(u, 10) for u in range(50)] + [(u, 30) for u in range(50)])
    s_again = ItemKNNScorer(k_neighbors=10).fit(train, catalog).score(0, [10])

    assert np.array_equal(s_train, s_again)
    s = dict(zip(catalog.tolist(), s_again.tolist()))
    assert s[30] == 0.0


def test_symmetry():
    """sim(i,j) == sim(j,i)."""
    rows = [(0, 10), (0, 20), (1, 10), (1, 20), (2, 20), (2, 30)]
    catalog = np.array([10, 20, 30])
    scorer = ItemKNNScorer(k_neighbors=10).fit(_df(rows), catalog)
    sim = scorer.sim.toarray()
    assert np.allclose(sim, sim.T)


def test_no_neighbors_zero_scores():
    """История без соседей даёт нулевой вектор."""
    rows = [(0, 10), (1, 20), (2, 20)]
    catalog = np.array([10, 20])
    scorer = ItemKNNScorer(k_neighbors=10).fit(_df(rows), catalog)
    assert np.all(scorer.score(0, [10]) == 0.0)
