"""Глобальная популярность товаров с опциональным временным затуханием."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

VARIANTS = ("popularity", "popularity_decay")


class PopularityScorer:
    """Popularity baseline, одинаковый для всех пользователей."""

    def __init__(self, variant: str = "popularity", half_life_days: float = 30.0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant должен быть из {VARIANTS}, получено {variant!r}")
        self.variant = variant
        self.half_life_days = half_life_days
        self._scores: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, catalog: np.ndarray) -> "PopularityScorer":
        """Посчитать скоры по train и выровнять их по каталогу."""
        catalog = np.asarray(catalog)

        if self.variant == "popularity":
            weight = train_df.groupby("item_id").size()
        else:
            t_ref = train_df["t_dat"].max()
            age_days = (t_ref - train_df["t_dat"]).dt.days.to_numpy()
            w = 0.5 ** (age_days / self.half_life_days)
            weight = pd.Series(w, index=train_df["item_id"].to_numpy()).groupby(level=0).sum()

        self._scores = weight.reindex(catalog, fill_value=0.0).to_numpy(dtype=float)
        return self

    def score(self, user_id: int, prefix_items: Sequence[int]) -> np.ndarray:  # noqa: ARG002
        """Вернуть предвычисленный вектор популярности."""
        if self._scores is None:
            raise RuntimeError("PopularityScorer.score вызван до fit()")
        return self._scores
