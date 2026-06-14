"""Сравнение SASRec и ItemKNN при разной длине входной истории."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import load_config
from src.data.split import catalog_hash
from src.sasrec.scorer import load_real_sasrec
from src.models.itemknn import ItemKNNScorer
from src.eval.metrics import rank_of_target

KS = [1, 2, 4, 8, 16, 50]
MIN_LEN = 16


def main():
    cfg = load_config("configs/base.yaml")
    catalog = pd.read_parquet("artifacts/split/catalog.parquet")["item_id"].to_numpy()
    cat_pos = {int(c): i for i, c in enumerate(catalog.tolist())}
    sas = load_real_sasrec("artifacts/sasrec/checkpoint", catalog, cfg, expected_hash=catalog_hash(catalog))
    knn = ItemKNNScorer.load(Path("artifacts/models/itemknn"), catalog, 200, "sum")
    train = pd.read_parquet("artifacts/split/train.parquet")
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")

    items = train.sort_values(["user_id", "t_dat", "item_id"]).groupby("user_id")["item_id"].apply(list)
    val = hold[hold["is_val_eval"]]
    long_users = [(u, t) for u, t in zip(val["user_id"].to_numpy(), val["val_item"].to_numpy())
                  if len(items.get(u, [])) >= MIN_LEN]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(long_users), min(5000, len(long_users)), replace=False)
    sample = [long_users[i] for i in idx]
    print(f"длинноисторийных юзеров (>= {MIN_LEN} событий): {len(long_users)}, выборка {len(sample)}")

    res_sas = {k: 0 for k in KS}
    res_knn = {k: 0 for k in KS}
    n = 0
    for u, tgt in sample:
        full = items[u]
        mask = set(int(x) for x in full)
        if int(tgt) in mask:
            continue
        n += 1
        for k in KS:
            pre = full[-k:]
            res_sas[k] += rank_of_target(sas.score(int(u), pre), int(tgt), mask, catalog, cat_pos) <= 20
            res_knn[k] += rank_of_target(knn.score(int(u), pre), int(tgt), mask, catalog, cat_pos) <= 20

    print(f"\nRecall@20 vs длина ПОДАННОГО префикса (n={n}, маска/таргет фиксированы):")
    print(f"{'k айтемов':<10}" + "".join(f"{k:>8}" for k in KS))
    print(f"{'SASRec':<10}" + "".join(f"{res_sas[k]/n:>8.4f}" for k in KS))
    print(f"{'ItemKNN':<10}" + "".join(f"{res_knn[k]/n:>8.4f}" for k in KS))
    print(f"\nSASRec прирост 1->50: {res_sas[50]/n - res_sas[1]/n:+.4f} "
          f"({(res_sas[50]/res_sas[1]-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
