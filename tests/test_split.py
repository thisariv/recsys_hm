"""Тесты temporal leave-last-out split на ручном примере."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import make_split

D = {i: pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(1, 4)}


def _toy_df() -> pd.DataFrame:
    rows = [
        # u6 добавляет товары в train-каталог.
        (6, 5, D[1]), (6, 7, D[1]), (6, 8, D[1]), (6, 9, D[1]),
        (6, 100, D[2]), (6, 101, D[2]), (6, 102, D[2]), (6, 103, D[2]),
        (0, 100, D[1]), (0, 101, D[2]),
        # u1 имеет допустимые validation и test таргеты.
        (1, 100, D[1]),
        (1, 5, D[2]), (1, 50, D[2]),
        (1, 7, D[3]), (1, 70, D[3]),
        # u2 повторяет test-товар из train.
        (2, 8, D[1]), (2, 100, D[1]),
        (2, 9, D[2]),
        (2, 8, D[3]),
        # u3 имеет одинаковые validation и test таргеты.
        (3, 100, D[1]),
        (3, 9, D[2]),
        (3, 9, D[3]),
        # u4 содержит validation-товар вне train-каталога.
        (4, 100, D[1]),
        (4, 999, D[2]),
        (4, 7, D[3]),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "t_dat"])


def test_split_manual_answer():
    train, holdouts, catalog, stats = make_split(_toy_df(), min_days=3)

    h = holdouts.set_index("user_id")
    assert set(h.index) == {1, 2, 3, 4}
    assert 0 not in h.index and 6 not in h.index

    assert (h.loc[1, "val_item"], h.loc[1, "test_item"]) == (5, 7)
    assert bool(h.loc[1, "is_val_eval"]) and bool(h.loc[1, "is_test_eval"])

    assert h.loc[2, "test_item"] == 8
    assert bool(h.loc[2, "is_val_eval"]) and not bool(h.loc[2, "is_test_eval"])

    assert h.loc[3, "val_item"] == 9 and h.loc[3, "test_item"] == 9
    assert bool(h.loc[3, "is_val_eval"]) and not bool(h.loc[3, "is_test_eval"])

    assert h.loc[4, "val_item"] == 999
    assert not bool(h.loc[4, "is_val_eval"]) and bool(h.loc[4, "is_test_eval"])

    tr_pairs = set(map(tuple, train[["user_id", "item_id"]].values.tolist()))
    assert (1, 50) not in tr_pairs and (1, 70) not in tr_pairs
    assert (1, 5) not in tr_pairs and (1, 7) not in tr_pairs

    assert set(catalog.tolist()) == {5, 7, 8, 9, 100, 101, 102, 103}
    assert 999 not in set(catalog.tolist())

    f = stats["funnel"]
    assert f["n_users_total"] == 6
    assert f["n_users_ge3_days"] == 4
    assert f["test"]["n_dropped_cold"] == 0
    assert f["test"]["n_dropped_repeat"] == 2
    assert f["test"]["n_test_eval"] == 2
    assert f["val"]["n_dropped_cold"] == 1
    assert f["val"]["n_dropped_repeat"] == 0
    assert f["val"]["n_val_eval"] == 3

    dd = stats["distinct_days_per_user"]
    assert (dd["min"], dd["median"], dd["max"]) == (2, 3.0, 3)
    assert dd["mean"] == pytest.approx(16 / 6)
    tp = stats["train_prefix_len"]
    assert (tp["min"], tp["median"], tp["max"]) == (1, 1.0, 2)


def test_split_invariants_hold_on_toy():
    """Повторный split даёт те же данные и catalog_hash."""
    df = _toy_df()
    _, h1, c1, s1 = make_split(df, min_days=3)
    _, h2, c2, s2 = make_split(df, min_days=3)
    assert s1["catalog_hash"] == s2["catalog_hash"]
    assert s1["n_v_train"] == s2["n_v_train"]
    assert s1["funnel"]["test"]["n_test_eval"] == s2["funnel"]["test"]["n_test_eval"]
    assert h1.equals(h2)
    assert list(c1) == list(c2)
