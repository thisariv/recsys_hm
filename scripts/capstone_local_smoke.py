"""Быстрая локальная проверка пайплайна реранкера на небольшой выборке."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.als import ALSScorer
from src.models.ease import load_real_ease
from src.models.itemknn import ItemKNNScorer
from src.models.popularity import PopularityScorer
from src.rerank.features import feat_names, metrics_from_order, user_candidate_features

SEED, NU, TOPK = 42, 60, 100


def main():
    t0 = time.time()
    catalog = pd.read_parquet("artifacts/split/catalog.parquet")["item_id"].to_numpy()
    cat_pos = {int(c): i for i, c in enumerate(catalog.tolist())}
    v_train = set(int(c) for c in catalog.tolist())
    n_items = len(catalog)
    train = pd.read_parquet("artifacts/split/train.parquet")
    train["t_dat"] = pd.to_datetime(train["t_dat"])
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")
    items = pd.read_parquet("artifacts/processed/items.parquet").drop_duplicates("item_id").set_index("item_id")

    pop = PopularityScorer("popularity").fit(train, catalog)
    dec = PopularityScorer("popularity_decay", 30.0).fit(train, catalog)
    pop_vec, dec_vec = pop.score(0, []).astype(np.float32), dec.score(0, []).astype(np.float32)
    knn = ItemKNNScorer.load(Path("artifacts/models/itemknn"), catalog, 200, "sum")
    als = ALSScorer.load(Path("artifacts/models/als"), catalog, factors=128, regularization=0.01,
                         iterations=20, alpha=40.0, seed=SEED)
    ease = load_real_ease("artifacts/ease/checkpoint", catalog)
    print(f"[smoke] модели готовы {time.time()-t0:.0f}s")

    maxd = train["t_dat"].max()
    recency_vec = np.full(n_items, 9999.0, np.float32); logpop_vec = np.zeros(n_items, np.float32)
    for it, d in (maxd - train.groupby("item_id")["t_dat"].max()).dt.days.items():
        if it in cat_pos: recency_vec[cat_pos[it]] = d
    for it, c in train.groupby("item_id")["user_id"].count().items():
        if it in cat_pos: logpop_vec[cat_pos[it]] = np.log1p(c)

    # Для smoke-теста достаточно случайных нормированных эмбеддингов.
    rng = np.random.default_rng(SEED)
    emb = rng.standard_normal((n_items, 16)).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    CAT_COLS = ["product_group_name", "department_name", "section_name"]
    cat_codes = {c: items[c].reindex(catalog).astype("category").cat.codes.to_numpy().astype(np.int32)
                 for c in CAT_COLS}

    val_u = rng.choice(hold[hold["is_val_eval"]]["user_id"].to_numpy(), NU, replace=False)
    test_u = rng.choice(hold[hold["is_test_eval"]]["user_id"].to_numpy(), NU, replace=False)
    sub = train[train["user_id"].isin(set(val_u.tolist()) | set(test_u.tolist()))]
    train_seq = sub.sort_values(["user_id", "t_dat", "item_id"]).groupby("user_id")["item_id"].apply(list)
    ndays = sub.groupby("user_id")["t_dat"].nunique()
    vmap = dict(zip(hold["user_id"].to_numpy(), hold["val_item"].to_numpy()))

    def build(users, is_test, tgt):
        Xs, ys, gs, ceil = [], [], [], 0
        for u in users:
            pre = list(train_seq.get(u, []))
            if is_test and u in vmap:
                pre.append(int(vmap[u]))
            r = user_candidate_features(pre, int(tgt[u]), ease=ease, als=als, knn=knn,
                                        pop_vec=pop_vec, dec_vec=dec_vec, recency_vec=recency_vec,
                                        logpop_vec=logpop_vec, emb=emb, cat_codes=cat_codes,
                                        cat_cols=CAT_COLS, catalog=catalog, cat_pos=cat_pos,
                                        v_train=v_train, topk=TOPK,
                                        hist_days=int(ndays.get(u, 1)) + (1 if is_test else 0))
            if r is None:
                continue
            F, lab, hit = r
            Xs.append(F); ys.append(lab); gs.append(len(lab)); ceil += hit
        X, y = np.concatenate(Xs), np.concatenate(ys)
        return X, y, np.array(gs), ceil / len(gs)

    vtgt = dict(zip(hold["user_id"].to_numpy(), hold["val_item"].to_numpy()))
    ttgt = dict(zip(hold["user_id"].to_numpy(), hold["test_item"].to_numpy()))
    Xtr, ytr, gtr, _ = build(val_u, False, vtgt)
    Xte, yte, gte, ceil = build(test_u, True, ttgt)
    F = feat_names(CAT_COLS)
    print(f"[smoke] Xtr={Xtr.shape} Xte={Xte.shape} feats={len(F)} ceiling={ceil:.3f}")
    assert Xtr.shape[1] == len(F) and not np.isnan(Xtr).any(), "битые фичи"
    assert set(np.unique(ytr)).issubset({0, 1}), "лейблы не бинарные"

    r_e, n_e = metrics_from_order(-Xte[:, 1], yte, gte)
    assert r_e <= ceil + 1e-9, "EASE-alone recall выше потолка — баг"
    print(f"[smoke] EASE-alone R@20={r_e:.4f} NDCG@20={n_e:.4f} (потолок={ceil:.4f})")

    try:
        import lightgbm as lgb
    except ModuleNotFoundError:
        print("[smoke] lightgbm не установлен локально (есть на Kaggle) — путь LGBM пропущен")
        print(f"[smoke] OK — интеграция реальных скореров + фичи рабочие ({time.time()-t0:.0f}s)")
        return
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=50, learning_rate=0.1,
                        num_leaves=15, label_gain=[0, 1], random_state=SEED, n_jobs=1, verbose=-1)
    rk.fit(Xtr, ytr, group=gtr)
    r_r, n_r = metrics_from_order(rk.predict(Xte), yte, gte)
    print(f"[smoke] reranker  R@20={r_r:.4f} NDCG@20={n_r:.4f}")
    assert r_r <= ceil + 1e-9, "reranker recall выше потолка — баг"
    print(f"[smoke] OK — интеграция + LightGBM-путь рабочие ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
