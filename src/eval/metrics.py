"""Ранг таргета и метрики для одного релевантного товара."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rank_of_target(
    scores: np.ndarray,
    target_item: int,
    mask_items: Iterable[int],
    catalog_ids: np.ndarray,
    cat_pos: dict[int, int] | None = None,
) -> int:
    """Вернуть 1-индексный ранг таргета после маскирования истории."""
    scores = np.asarray(scores)
    mask_set = mask_items if isinstance(mask_items, (set, frozenset)) else set(mask_items)

    if target_item in mask_set:
        raise AssertionError(
            f"target {target_item} попал в mask; проверь split"
        )

    if cat_pos is None:
        cat_pos = {int(it): i for i, it in enumerate(catalog_ids.tolist())}
    t_idx = cat_pos[target_item]
    s_t = scores[t_idx]

    # При равных скорах меньший item_id идёт раньше.
    better_all = (scores > s_t) | ((scores == s_t) & (catalog_ids < target_item))
    n_better = int(better_all.sum())

    for it in mask_set:
        p = cat_pos.get(int(it))
        if p is None:
            continue
        s = scores[p]
        if s > s_t or (s == s_t and it < target_item):
            n_better -= 1

    return 1 + n_better


def recall_at_k(rank: int, k: int) -> float:
    """Recall@K для одного таргета."""
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG@K для одного таргета."""
    return 1.0 / math.log2(1 + rank) if rank <= k else 0.0


def mrr(rank: int) -> float:
    """Reciprocal rank по полному каталогу."""
    return 1.0 / rank
