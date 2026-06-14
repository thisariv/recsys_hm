"""Тесты popularity baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.popularity import PopularityScorer

T0 = pd.Timestamp("2020-06-01")


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "t_dat"])


def test_raw_ranks_by_frequency():
    """Товары ранжируются по частоте в train."""
    rows = (
        [(u, 10, T0) for u in range(4)]
        + [(u, 20, T0) for u in range(3)]
        + [(u, 30, T0) for u in range(2)]
        + [(0, 40, T0)]
    )
    catalog = np.array([40, 30, 20, 10])
    scorer = PopularityScorer("popularity").fit(_df(rows), catalog)
    scores = scorer.score(0, [])

    assert scores.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert catalog[np.argsort(-scores, kind="stable")].tolist() == [10, 20, 30, 40]


def test_anti_leak_only_train_counts():
    """Частоты считаются только по данным, переданным в fit."""
    train = _df([(0, 10, T0), (1, 10, T0), (0, 20, T0)])
    _holdout = _df([(u, 20, T0) for u in range(100)])

    catalog = np.array([10, 20])
    scores = PopularityScorer("popularity").fit(train, catalog).score(0, [])

    assert scores.tolist() == [2.0, 1.0]
    assert scores[0] > scores[1]


def test_decay_prefers_fresh():
    """При равной частоте свежий товар получает больший decay-score."""
    t_ref = T0
    old = t_ref - pd.Timedelta(days=60)
    fresh = _df([(0, 100, t_ref), (1, 100, t_ref)])
    stale = _df([(0, 200, old), (1, 200, old)])
    train = pd.concat([fresh, stale], ignore_index=True)

    catalog = np.array([100, 200])
    scores = PopularityScorer("popularity_decay", half_life_days=30).fit(train, catalog).score(0, [])

    assert scores[0] > scores[1]
    assert scores.tolist() == [2.0, 0.5]


def test_determinism():
    train = _df([(0, 10, T0), (1, 10, T0), (0, 20, T0)])
    catalog = np.array([10, 20])
    s1 = PopularityScorer("popularity").fit(train, catalog).score(0, [])
    s2 = PopularityScorer("popularity").fit(train, catalog).score(0, [])
    assert np.array_equal(s1, s2)
