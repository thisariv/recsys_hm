"""Тесты ранжирования и offline-метрик."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import mrr, ndcg_at_k, rank_of_target, recall_at_k

CATALOG = np.array([1, 2, 3, 4, 5])

CASES = [
    (np.array([0.9, 0.8, 0.7, 0.6, 0.5]), {1}, 3, 2),
    (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), set(), 5, 1),
    (np.array([0.5, 0.4, 0.3, 0.2, 0.1]), {1, 2}, 5, 3),
    (np.array([0.5, 0.5, 0.1, 0.1, 0.1]), set(), 2, 2),
]


def test_rank_of_target_exact():
    ranks = [rank_of_target(s, t, m, CATALOG) for s, m, t, _ in CASES]
    assert ranks == [exp for *_, exp in CASES]


def test_aggregate_metrics_exact():
    ranks = [rank_of_target(s, t, m, CATALOG) for s, m, t, _ in CASES]
    n = len(ranks)

    recall2 = sum(recall_at_k(r, 2) for r in ranks) / n
    recall3 = sum(recall_at_k(r, 3) for r in ranks) / n
    ndcg2 = sum(ndcg_at_k(r, 2) for r in ranks) / n
    ndcg3 = sum(ndcg_at_k(r, 3) for r in ranks) / n
    mrr_v = sum(mrr(r) for r in ranks) / n

    assert recall2 == pytest.approx(0.75, abs=1e-6)
    assert recall3 == pytest.approx(1.0, abs=1e-6)
    assert ndcg2 == pytest.approx(0.565465, abs=1e-6)
    assert ndcg3 == pytest.approx(0.690465, abs=1e-6)
    assert mrr_v == pytest.approx(0.583333, abs=1e-6)


def test_negative_target_in_mask_raises():
    """Таргет не может одновременно находиться в маске."""
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    with pytest.raises(AssertionError):
        rank_of_target(scores, target_item=3, mask_items={3}, catalog_ids=CATALOG)


def test_mask_equivalent_to_neg_inf():
    """Исключение товара эквивалентно присвоению ему -inf."""
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    masked = {1}
    r_exclude = rank_of_target(scores, 3, masked, CATALOG)
    s2 = scores.copy()
    s2[0] = -np.inf
    r_neginf = rank_of_target(s2, 3, set(), CATALOG)
    assert r_exclude == r_neginf == 2
