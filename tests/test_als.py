"""Тесты ALS и fold-in по истории пользователя."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.als import ALSScorer

T0 = pd.Timestamp("2020-06-01")


def _df(rows):
    return pd.DataFrame([(u, i, T0) for u, i in rows], columns=["user_id", "item_id", "t_dat"])


def _toy_train():
    rows = []
    for u in range(8):
        rows += [(u, 10), (u, 20), (u, 30)]
    for u in range(8, 16):
        rows += [(u, 40), (u, 50), (u, 60)]
    return _df(rows)


CATALOG = np.array([10, 20, 30, 40, 50, 60])


def test_vector_shape_full_catalog():
    scorer = ALSScorer(factors=8, iterations=5, seed=42).fit(_toy_train(), CATALOG)
    scores = scorer.score(0, [10, 20])
    assert scores.shape == (len(CATALOG),)


def test_anti_leak_factors_from_train_only():
    """Повторное обучение с тем же seed даёт те же факторы."""
    scorer = ALSScorer(factors=8, iterations=5, seed=42).fit(_toy_train(), CATALOG)
    assert scorer.Y.shape == (len(CATALOG), 8)
    s2 = ALSScorer(factors=8, iterations=5, seed=42).fit(_toy_train(), CATALOG)
    assert np.allclose(scorer.Y, s2.Y)


def test_foldin_conditions_on_prefix():
    """Добавление товара в историю меняет fold-in вектор."""
    scorer = ALSScorer(factors=8, iterations=10, seed=42).fit(_toy_train(), CATALOG)

    s_train = scorer.score(0, [10, 20])
    s_with_val = scorer.score(0, [10, 20, 40])

    assert not np.allclose(s_train, s_with_val)
    assert s_with_val[CATALOG.tolist().index(50)] > s_train[CATALOG.tolist().index(50)]


def test_empty_prefix_zero():
    scorer = ALSScorer(factors=8, iterations=5, seed=42).fit(_toy_train(), CATALOG)
    assert np.all(scorer.score(0, []) == 0.0)
