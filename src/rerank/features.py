"""Построение кандидатов и признаков для LightGBM-реранкера."""
from __future__ import annotations

from typing import Sequence

import numpy as np

NEG_INF = -1e30

# Порядок должен совпадать при обучении и инференсе.
BASE_FEATS = ["ease_score", "ease_rank", "als_score", "knn_score", "pop_score",
              "decay_score", "recency", "logpop", "content_cos", "hist_len", "hist_days"]


def feat_names(cat_cols: Sequence[str]) -> list[str]:
    return list(BASE_FEATS) + [f"ov_{c}" for c in cat_cols]


def user_candidate_features(
    prefix_raw: Sequence[int],
    target: int,
    *,
    ease, als, knn,
    pop_vec: np.ndarray,
    dec_vec: np.ndarray,
    recency_vec: np.ndarray,
    logpop_vec: np.ndarray,
    emb: np.ndarray,
    cat_codes: dict[str, np.ndarray],
    cat_cols: Sequence[str],
    catalog: np.ndarray,
    cat_pos: dict[int, int],
    v_train: set[int],
    topk: int,
    hist_days: int = 0,
):
    """Вернуть признаки, метки и флаг попадания таргета в набор кандидатов."""
    prefix = [it for it in prefix_raw if it in v_train]
    if not prefix:
        return None
    pidx = np.array([cat_pos[it] for it in prefix], dtype=np.int64)

    ease_s = np.asarray(ease.score(0, prefix), dtype=np.float32)
    ease_s[pidx] = NEG_INF
    if topk >= ease_s.size:
        cand = np.argsort(-ease_s)
    else:
        cand = np.argpartition(ease_s, -topk)[-topk:]
    cand = cand[np.argsort(-ease_s[cand])]
    k = len(cand)

    als_s = np.asarray(als.score(0, prefix), dtype=np.float32)
    knn_s = np.asarray(knn.score(0, prefix), dtype=np.float32)
    profile = emb[pidx].mean(axis=0)
    content = (emb[cand] @ profile).astype(np.float32)

    names = feat_names(cat_cols)
    F = np.empty((k, len(names)), dtype=np.float32)
    F[:, 0] = ease_s[cand]
    F[:, 1] = np.arange(k, dtype=np.float32)
    F[:, 2] = als_s[cand]
    F[:, 3] = knn_s[cand]
    F[:, 4] = pop_vec[cand]
    F[:, 5] = dec_vec[cand]
    F[:, 6] = recency_vec[cand]
    F[:, 7] = logpop_vec[cand]
    F[:, 8] = content
    F[:, 9] = len(prefix)
    F[:, 10] = hist_days
    for j, c in enumerate(cat_cols):
        pc = cat_codes[c][pidx]
        cc = cat_codes[c][cand]
        F[:, 11 + j] = (pc[None, :] == cc[:, None]).mean(axis=1).astype(np.float32)

    labels = (catalog[cand] == target).astype(np.int8)
    return F, labels, bool(labels.sum() > 0)


def metrics_from_order(order_scores: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 20):
    """Посчитать Recall@K и NDCG@K для сгруппированных кандидатов."""
    recall = ndcg = 0.0
    i = 0
    for g in groups:
        ys = y[i:i + g]; ss = order_scores[i:i + g]
        rank = np.argsort(-ss)
        pos = np.where(ys[rank] == 1)[0]
        if len(pos) and pos[0] < k:
            recall += 1.0
            ndcg += 1.0 / np.log2(pos[0] + 2)
        i += g
    n = len(groups)
    return recall / n, ndcg / n
