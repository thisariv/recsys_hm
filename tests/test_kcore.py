"""Тесты итеративной k-core фильтрации."""
from __future__ import annotations

import pandas as pd

from src.data.prepare import iterative_kcore

MIN_USER = 2
MIN_ITEM = 2


def _toy_df() -> pd.DataFrame:
    """Граф, в котором удаление редкого товара требует второго прохода."""
    rows = [
        ("u1", "i1"), ("u1", "i2"),
        ("u2", "i1"), ("u2", "i2"),
        ("u3", "i1"), ("u3", "i2"),
        ("u8", "i1"), ("u8", "i9"),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "article_id"])


def test_iterative_kcore_invariant_and_trap():
    out = iterative_kcore(_toy_df(), MIN_USER, MIN_ITEM)

    assert out["customer_id"].value_counts().min() >= MIN_USER
    assert out["article_id"].value_counts().min() >= MIN_ITEM

    assert "u8" not in set(out["customer_id"])
    assert "i9" not in set(out["article_id"])

    assert set(out["customer_id"]) == {"u1", "u2", "u3"}
    assert set(out["article_id"]) == {"i1", "i2"}
    assert len(out) == 6


def test_iterative_kcore_empty_when_no_core():
    """Граф без подходящего ядра становится пустым."""
    df = pd.DataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")],
        columns=["customer_id", "article_id"],
    )
    out = iterative_kcore(df, MIN_USER, MIN_ITEM)
    assert out.empty
