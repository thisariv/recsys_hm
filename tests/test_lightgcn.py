"""Тесты LightGCNScorer на небольших эмбеддингах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.lightgcn import LightGCNScorer, gen_interactions, gen_valid

CATALOG = np.array([10, 20, 30, 40, 50])


def _stub_scorer():
    item_t = {"[PAD]": 0, "10": 1, "20": 2, "30": 3, "40": 4, "50": 5}
    user_t = {"[PAD]": 0, "0": 1, "1": 2}
    d = 4
    item_emb = np.zeros((6, d), dtype=np.float32)
    item_emb[1] = [1, 0, 0, 0]
    item_emb[2] = [0, 1, 0, 0]
    item_emb[3] = [2, 0, 0, 0]
    item_emb[4] = [0, 0, 1, 0]
    item_emb[5] = [0, 2, 0, 0]
    user_emb = np.zeros((3, d), dtype=np.float32)
    user_emb[1] = [1, 0, 0, 0]
    user_emb[2] = [0, 1, 0, 0]
    return LightGCNScorer(user_emb, item_emb, user_t, item_t, CATALOG)


def test_scoring_dot_product_and_order():
    s = _stub_scorer()
    sc0 = s.score(0, [])
    assert CATALOG[int(np.argmax(sc0))] == 30
    sc1 = s.score(1, [])
    assert CATALOG[int(np.argmax(sc1))] == 50
    assert sc0.shape == (len(CATALOG),)


def test_different_users_different_scores():
    s = _stub_scorer()
    assert not np.allclose(s.score(0, []), s.score(1, []))


def test_cold_user_abstains():
    s = _stub_scorer()
    out = s.score(999, [])
    assert np.all(out == 0.0) and s.n_cold_user == 1


def test_missing_item_gets_neg_inf():
    cat = np.array([10, 60])
    item_t = {"[PAD]": 0, "10": 1}
    user_t = {"[PAD]": 0, "0": 1}
    s = LightGCNScorer(np.array([[0, 0], [1, 0]], dtype=np.float32),
                       np.array([[0, 0], [1, 0]], dtype=np.float32),
                       user_t, item_t, cat)
    assert s.n_missing == 1
    out = s.score(0, [])
    assert out[1] <= -1e8


def test_generators():
    train = pd.DataFrame({"user_id": [0, 0, 0, 1], "item_id": [10, 10, 20, 30],
                          "t_dat": pd.Timestamp("2020-01-01")})
    inter = gen_interactions(train)
    assert len(inter) == 3
    hold = pd.DataFrame([(0, 99, None, 1, None, True, True)],
                        columns=["user_id", "val_item", "val_date", "test_item",
                                 "test_date", "is_val_eval", "is_test_eval"])
    v = gen_valid(hold)
    assert v.iloc[0]["item_id"] == 99 and v.iloc[0]["user_id"] == 0
