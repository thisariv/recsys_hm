"""Подготовка данных и скорер для LightGCN.

Модель обучается через RecBole, а сохранённые эмбеддинги приводятся к порядку
локального каталога. Скорер предназначен для validation: test потребовал бы
обновить пользовательский эмбеддинг с учётом validation-события.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTER_FIELDS = ["user_id:token", "item_id:token"]
MISSING_SCORE = -1e9


def gen_interactions(train_df: pd.DataFrame) -> pd.DataFrame:
    """Уникальные (user_id, item_id) из train — рёбра двудольного графа."""
    return train_df[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)


def gen_valid(holdouts_df: pd.DataFrame) -> pd.DataFrame:
    """Собрать validation-пары для RecBole."""
    elig = holdouts_df[holdouts_df["is_val_eval"]]
    return pd.DataFrame({"user_id": elig["user_id"].to_numpy(),
                         "item_id": elig["val_item"].to_numpy()})


def write_inter(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(INTER_FIELDS) + "\n")
        for r in df.itertuples(index=False):
            f.write(f"{r.user_id}\t{r.item_id}\n")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class LightGCNScorer:
    """Скоринг по сохранённым пропагированным эмбеддингам LightGCN."""

    def __init__(self, user_emb: np.ndarray, item_emb: np.ndarray,
                 user_token2id: dict[str, int], item_token2id: dict[str, int],
                 catalog: np.ndarray) -> None:
        self.user_emb = np.asarray(user_emb, dtype=np.float32)
        self.item_emb = np.asarray(item_emb, dtype=np.float32)
        self.user_token2id = user_token2id
        catalog = np.asarray(catalog)
        self.n_items = len(catalog)

        # RecBole и локальный каталог используют разные индексы товаров.
        perm = np.zeros(self.n_items, dtype=np.int64)
        missing = np.zeros(self.n_items, dtype=bool)
        for i, cid in enumerate(catalog.tolist()):
            tid = item_token2id.get(str(int(cid)))
            if tid is None:
                missing[i] = True
            else:
                perm[i] = tid
        self.perm = perm
        self.missing = missing
        self.n_missing = int(missing.sum())
        self._item_v = self.item_emb[self.perm]
        self.n_cold_user = 0

    def score(self, user_id: int, prefix_items=None) -> np.ndarray:  # noqa: ARG002
        tid = self.user_token2id.get(str(int(user_id)))
        if tid is None:
            self.n_cold_user += 1
            return np.zeros(self.n_items, dtype=np.float64)
        scores = self._item_v @ self.user_emb[tid]
        if self.n_missing:
            scores[self.missing] = MISSING_SCORE
        return scores.astype(np.float64)

    def score_batch(self, batch):
        return [self.score(u, p) for u, p in batch]


def load_real_lightgcn(ckpt_dir, catalog, expected_hash: str | None = None) -> "LightGCNScorer":
    """Загрузить user_emb/item_emb + token-maps, сохранённые на Kaggle."""
    ckpt_dir = Path(ckpt_dir)
    user_emb = np.load(ckpt_dir / "user_emb.npy")
    item_emb = np.load(ckpt_dir / "item_emb.npy")
    with (ckpt_dir / "user_token_map.json").open(encoding="utf-8") as f:
        user_t = {str(k): int(v) for k, v in json.load(f).items()}
    with (ckpt_dir / "item_token_map.json").open(encoding="utf-8") as f:
        item_t = {str(k): int(v) for k, v in json.load(f).items()}
    meta_path = ckpt_dir / "meta.json"
    if expected_hash is not None and meta_path.exists():
        meta = json.load(meta_path.open(encoding="utf-8"))
        assert meta.get("catalog_hash") == expected_hash, (
            "catalog_hash чекпойнта LightGCN != текущего V_train"
        )
    return LightGCNScorer(user_emb, item_emb, user_t, item_t, catalog)
