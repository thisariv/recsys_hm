"""Небольшая случайная sequence-модель для проверки адаптера."""
from __future__ import annotations

import numpy as np


class StubSASRec:
    def __init__(self, n_items: int, d: int = 64, seed: int = 2026, decay: float = 0.9) -> None:
        rng = np.random.default_rng(seed)
        self.item_emb = rng.standard_normal((n_items + 1, d))
        self.item_emb[0] = 0.0
        self.w_seq = rng.standard_normal((d, d))
        self.decay = decay
        self.n_items = n_items

    def full_sort_predict(self, seqs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Вернуть скоры для всех RecBole item ID, включая PAD."""
        seqs = np.asarray(seqs)
        lengths = np.asarray(lengths)
        b, length = seqs.shape

        emb = self.item_emb[seqs]
        pos = np.arange(length)
        mask = pos[None, :] < lengths[:, None]
        w = (self.decay ** (lengths[:, None] - 1 - pos[None, :])) * mask
        h = np.einsum("bl,bld->bd", w, emb)
        last_hidden = h @ self.w_seq
        return last_hidden @ self.item_emb.T
