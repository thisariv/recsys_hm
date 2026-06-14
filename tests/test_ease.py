"""Тесты формулы и скоринга EASE."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.models.ease import EASEScorer, compute_ease_B, topk_sparse

CATALOG = np.array([10, 20, 30, 40])


def _toy_X():
    rows = [
        (0, 0), (0, 1),  # u0: 10,20
        (1, 0), (1, 1),  # u1: 10,20
        (2, 0), (2, 1),  # u2: 10,20
        (3, 2), (3, 3),  # u3: 30,40
        (4, 2), (4, 3),  # u4: 30,40
    ]
    r, c = zip(*rows)
    return sp.csr_matrix((np.ones(len(rows)), (r, c)), shape=(5, 4))


def test_B_formula_diag_zero_and_symmetry_of_signal():
    B = compute_ease_B(_toy_X(), lam=1.0)
    assert B.shape == (4, 4)
    assert np.allclose(np.diag(B), 0.0)
    assert B[0, 1] > B[0, 2]
    assert B[2, 3] > B[2, 0]


def test_scoring_ranks_cooccurring_item():
    B = compute_ease_B(_toy_X(), lam=1.0)
    s = EASEScorer(B, CATALOG)
    sc = s.score(0, [10])
    order = CATALOG[np.argsort(-sc)]
    assert order[0] == 20
    assert sc.shape == (4,)


def test_topk_sparse_matches_dense_ranking():
    B = compute_ease_B(_toy_X(), lam=1.0)
    s_dense = EASEScorer(B, CATALOG)
    s_sparse = EASEScorer(topk_sparse(B, k=2), CATALOG)
    a = CATALOG[int(np.argmax(s_dense.score(0, [30])))]
    b = CATALOG[int(np.argmax(s_sparse.score(0, [30])))]
    assert a == b == 40


def test_empty_prefix_zero():
    B = compute_ease_B(_toy_X(), lam=1.0)
    s = EASEScorer(B, CATALOG)
    assert np.all(s.score(0, [99999]) == 0.0)
