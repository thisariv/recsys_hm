"""Простые скореры для evaluation pipeline."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class RandomScorer:
    """Детерминированный случайный baseline."""

    def __init__(self, n_items: int, seed: int = 42) -> None:
        self.n_items = n_items
        self.seed = seed

    def score(self, user_id: int, prefix_items: Sequence[int]) -> np.ndarray:  # noqa: ARG002
        rng = np.random.default_rng(self.seed + int(user_id))
        return rng.random(self.n_items)
