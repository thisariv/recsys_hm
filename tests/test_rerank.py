"""Тесты кандидатов и признаков реранкера."""
from __future__ import annotations

import numpy as np

from src.rerank.features import feat_names, metrics_from_order, user_candidate_features

CATALOG = np.array([10, 20, 30, 40, 50, 60])
CAT_POS = {int(c): i for i, c in enumerate(CATALOG.tolist())}
V = set(int(c) for c in CATALOG.tolist())
N = len(CATALOG)


class ConstScorer:
    """Скорер с фиксированным вектором."""
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float64)

    def score(self, uid, prefix):
        return self.vec.copy()


def _kw(**over):
    base = dict(
        ease=ConstScorer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        als=ConstScorer(np.zeros(N)),
        knn=ConstScorer(np.zeros(N)),
        pop_vec=np.arange(N, dtype=np.float32),
        dec_vec=np.arange(N, dtype=np.float32),
        recency_vec=np.zeros(N, dtype=np.float32),
        logpop_vec=np.zeros(N, dtype=np.float32),
        emb=np.eye(N, dtype=np.float32),
        cat_codes={"g": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)},
        cat_cols=["g"],
        catalog=CATALOG,
        cat_pos=CAT_POS,
        v_train=V,
        topk=4,
    )
    base.update(over)
    return base


def test_candidates_order_mask_and_label():
    F, lab, hit = user_candidate_features([10], target=30, **_kw())
    assert F.shape == (4, len(feat_names(["g"])))
    pop_col = feat_names(["g"]).index("pop_score")
    cand_idx = F[:, pop_col].astype(int).tolist()
    assert cand_idx == [5, 4, 3, 2]
    assert 0 not in cand_idx
    assert F[:, 1].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert lab.tolist() == [0, 0, 0, 1] and hit


def test_hit_false_when_target_not_in_candidates():
    F, lab, hit = user_candidate_features([10], target=20, **_kw())
    assert lab.sum() == 0 and not hit


def test_empty_prefix_returns_none():
    assert user_candidate_features([999], target=30, **_kw()) is None


def test_content_and_overlap_features():
    kw = _kw()
    F, _, _ = user_candidate_features([10, 20], target=30, **kw)
    cc = feat_names(["g"]).index("content_cos")
    assert np.allclose(F[:, cc], 0.0)
    ov = feat_names(["g"]).index("ov_g")
    assert F[:, ov].max() <= 1.0 and F[:, ov].min() >= 0.0


def test_metrics_from_order():
    groups = np.array([4, 4])
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    scores = np.array([4, 3, 2, 1, 1, 2, 3, 4], dtype=float)
    rec, ndcg = metrics_from_order(scores, y, groups, k=20)
    assert rec == 1.0
    assert abs(ndcg - (0.5 + 1 / np.log2(5)) / 2) < 1e-9
    rec2, _ = metrics_from_order(scores, y, groups, k=2)
    assert rec2 == 0.0
