"""Тесты NumPy-инференса MultiVAE."""
from __future__ import annotations

import numpy as np

from src.models.multivae import MultiVAEScorer

CATALOG = np.array([10, 20, 30, 40])


def _toy_scorer(token2id=None):
    enc = [(np.eye(4, 5, dtype=np.float64), np.zeros(4))]
    dec = [(np.ones((5, 2), dtype=np.float64), np.zeros(5))]
    t = token2id or {"[PAD]": 0, "10": 1, "20": 2, "30": 3, "40": 4}
    return MultiVAEScorer(enc, dec, t, CATALOG)


def test_forward_shape_and_perm():
    s = _toy_scorer()
    out = s.score(0, [10, 20])
    assert out.shape == (4,)
    assert s.n_recbole == 5 and s.n_missing == 0


def test_numpy_forward_matches_manual():
    s = _toy_scorer()
    x = np.zeros(5); x[[1, 2]] = 1.0
    h = x / np.linalg.norm(x)
    enc_out = h @ s.enc[0][0].T + s.enc[0][1]
    mu = enc_out[:2]
    manual = mu @ s.dec[0][0].T + s.dec[0][1]
    assert np.allclose(s._forward(x), manual)


def test_missing_item_neg_inf():
    cat = np.array([10, 99])
    s = MultiVAEScorer([(np.eye(2, 3), np.zeros(2))], [(np.ones((3, 1)), np.zeros(3))],
                       {"[PAD]": 0, "10": 1}, cat)
    assert s.n_missing == 1
    out = s.score(0, [10])
    assert out[1] <= -1e8


def test_empty_prefix_zero():
    s = _toy_scorer()
    assert np.all(s.score(0, [99999]) == 0.0)
