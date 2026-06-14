"""Обучение LightGBM-реранкера в Kaggle.

EASE формирует top-100 кандидатов, затем LambdaMART использует скоры нескольких
моделей и признаки товаров. Реранкер обучается на validation и проверяется на test.
Скрипт разбит на секции, которые удобно переносить в ячейки ноутбука.
"""

# Cell 0: зависимости
import subprocess, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "implicit", "sentence-transformers"], check=True)


# Cell 1: загрузка данных
import os, sys, glob, json, time, shutil, pathlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Kaggle иногда пропускает пустые __init__.py при загрузке датасета, поэтому
# исходники копируются в working и package-маркеры создаются заново.
hits = glob.glob("/kaggle/input/**/src/models/ease.py", recursive=True)
assert hits, "src не найден в /kaggle/input — прикрепи датасет recsys-hm-capstone (Add Input)"
DS = hits[0][:hits[0].index("/src/models/ease.py")]
print("нашёл датасет в:", DS)
PKG = "/kaggle/working/_pkg"
shutil.rmtree(PKG, ignore_errors=True); os.makedirs(PKG)
shutil.copytree(f"{DS}/src", f"{PKG}/src")
for d, _, files in os.walk(f"{PKG}/src"):
    if any(f.endswith(".py") for f in files):
        pathlib.Path(d, "__init__.py").touch()
sys.path.insert(0, PKG)
# Очищаем кеш импортов при повторном запуске ячейки.
import importlib
for m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
    del sys.modules[m]
importlib.invalidate_caches()
from src.models.ease import load_real_ease
from src.models.als import ALSScorer
from src.models.itemknn import ItemKNNScorer
from src.models.popularity import PopularityScorer
from src.rerank.features import user_candidate_features, feat_names, metrics_from_order

N_TRAIN_USERS = 150_000
N_EVAL_USERS = 150_000
TOPK = 100
SEED = 42

train = pd.read_parquet(f"{DS}/data/train.parquet")
hold  = pd.read_parquet(f"{DS}/data/holdouts.parquet")
catalog = pd.read_parquet(f"{DS}/data/catalog.parquet")["item_id"].to_numpy()
items = pd.read_parquet(f"{DS}/data/items.parquet").drop_duplicates("item_id").set_index("item_id")
train["t_dat"] = pd.to_datetime(train["t_dat"])

v_train = set(int(x) for x in catalog.tolist())
cat_pos = {int(c): i for i, c in enumerate(catalog.tolist())}
n_items = len(catalog)
print(f"train={train.shape} hold={hold.shape} |V|={n_items}")


# Cell 2: retrieval-модели
t0 = time.time()
pop = PopularityScorer("popularity").fit(train, catalog)
dec = PopularityScorer("popularity_decay", 30.0).fit(train, catalog)
pop_vec = pop.score(0, []).astype(np.float32)
dec_vec = dec.score(0, []).astype(np.float32)
print(f"[fit] pop/decay {time.time()-t0:.0f}s")
knn = ItemKNNScorer(200, "sum").fit(train, catalog)
print(f"[fit] itemknn {time.time()-t0:.0f}s")
als = ALSScorer(factors=128, regularization=0.01, iterations=20, alpha=40.0, seed=SEED).fit(train, catalog)
print(f"[fit] als {time.time()-t0:.0f}s")
ease = load_real_ease(f"{DS}/ease", catalog)
print(f"[load] ease {time.time()-t0:.0f}s — все модели готовы")

# Статические признаки товаров.
maxdate = train["t_dat"].max()
last = train.groupby("item_id")["t_dat"].max()
cnt = train.groupby("item_id")["user_id"].count()
recency_vec = np.full(n_items, 9999.0, dtype=np.float32)
logpop_vec = np.zeros(n_items, dtype=np.float32)
for it, d in (maxdate - last).dt.days.items():
    if it in cat_pos: recency_vec[cat_pos[it]] = d
for it, c in cnt.items():
    if it in cat_pos: logpop_vec[cat_pos[it]] = np.log1p(c)


# Cell 3: контентные признаки
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
texts = items["text"].reindex(catalog).fillna("").astype(str).tolist()
emb = st.encode(texts, batch_size=512, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
print(f"[content] emb={emb.shape}")

CAT_COLS = ["product_group_name", "product_type_name", "department_name",
            "section_name", "colour_group_name", "index_group_name"]
cat_codes = {c: items[c].reindex(catalog).astype("category").cat.codes.to_numpy().astype(np.int32)
             for c in CAT_COLS}


# Cell 4: кандидаты и признаки
# Векторизованная сборка историй заметно быстрее groupby.apply(list).
_ts = train.sort_values(["user_id", "t_dat", "item_id"])
_u = _ts["user_id"].to_numpy()
_i = _ts["item_id"].to_numpy()
_b = np.flatnonzero(np.r_[True, _u[1:] != _u[:-1]])
train_seq = {int(_u[s]): _i[s:e].tolist() for s, e in zip(_b, np.r_[_b[1:], len(_u)])}
ndays = train.groupby("user_id")["t_dat"].nunique()
val_item_by_user = dict(zip(hold["user_id"].to_numpy(), hold["val_item"].to_numpy()))

FEATS = feat_names(CAT_COLS)


def build_set(user_ids, is_test):
    """Собрать признаки и метки для списка пользователей."""
    Xs, ys, groups, ceil = [], [], [], 0
    t = time.time()
    for n, u in enumerate(user_ids):
        prefix_raw = list(train_seq.get(u, []))
        if is_test and u in val_item_by_user:
            prefix_raw.append(int(val_item_by_user[u]))
        r = user_candidate_features(
            prefix_raw, int(tgt_by_user[u]),
            ease=ease, als=als, knn=knn, pop_vec=pop_vec, dec_vec=dec_vec,
            recency_vec=recency_vec, logpop_vec=logpop_vec, emb=emb,
            cat_codes=cat_codes, cat_cols=CAT_COLS, catalog=catalog, cat_pos=cat_pos,
            v_train=v_train, topk=TOPK,
            hist_days=int(ndays.get(u, 1)) + (1 if is_test else 0))
        if r is None:
            continue
        F, lab, hit = r
        Xs.append(F); ys.append(lab); groups.append(len(lab)); ceil += hit
        if (n + 1) % 5000 == 0:
            print(f"  ...{n+1}/{len(user_ids)} ({time.time()-t:.0f}s)", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"[build] users={len(groups)} rows={len(y)} ceiling(target∈top{TOPK})={ceil/len(groups):.4f}", flush=True)
    return X, y, np.array(groups), ceil / len(groups)


rng = np.random.default_rng(SEED)
val_users = hold[hold["is_val_eval"]]["user_id"].to_numpy()
test_users = hold[hold["is_test_eval"]]["user_id"].to_numpy()
if N_TRAIN_USERS: val_users = rng.choice(val_users, min(N_TRAIN_USERS, len(val_users)), replace=False)
if N_EVAL_USERS:  test_users = rng.choice(test_users, min(N_EVAL_USERS, len(test_users)), replace=False)

tgt_by_user = dict(zip(hold["user_id"].to_numpy(), hold["val_item"].to_numpy()))
print("=== build TRAIN (val) ===", flush=True); Xtr, ytr, gtr, _ = build_set(val_users, is_test=False)
tgt_by_user = dict(zip(hold["user_id"].to_numpy(), hold["test_item"].to_numpy()))
print("=== build EVAL (test) ===", flush=True); Xte, yte, gte, ceiling = build_set(test_users, is_test=True)


# Cell 5: обучение реранкера
import lightgbm as lgb
ranker = lgb.LGBMRanker(
    objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    label_gain=[0, 1], importance_type="gain", random_state=SEED, n_jobs=-1,
)
ranker.fit(Xtr, ytr, group=gtr)
print("[train] LambdaMART обучен")


# Cell 6: оценка
ease_alone = -Xte[:, 1]
rer_scores = ranker.predict(Xte)
r_ease, n_ease = metrics_from_order(ease_alone, yte, gte)
r_rer,  n_rer  = metrics_from_order(rer_scores, yte, gte)

print(f"\n=== TEST (n_users={len(gte)}, candidate ceiling={ceiling:.4f}) ===")
print(f"EASE-alone : Recall@20={r_ease:.4f}  NDCG@20={n_ease:.4f}")
print(f"RERANKER   : Recall@20={r_rer:.4f}  NDCG@20={n_rer:.4f}")
print(f"Δ Recall@20 = {(r_rer-r_ease)/r_ease*100:+.1f}%   (потолок ретрива = {ceiling:.4f})")
print(f"доля потолка реализована: EASE {r_ease/ceiling:.1%} -> reranker {r_rer/ceiling:.1%}")


# Cell 7: сохранение результатов
imp = sorted(zip(FEATS, ranker.feature_importances_), key=lambda x: -x[1])
print("\n[feature importances (gain)]")
for f, v in imp:
    print(f"  {f:14s} {v:12.1f}")

ranker.booster_.save_model("/kaggle/working/rerank_lgbm.txt")
metrics = {"ceiling": ceiling, "n_eval_users": int(len(gte)),
           "ease_alone": {"recall@20": r_ease, "ndcg@20": n_ease},
           "reranker":   {"recall@20": r_rer,  "ndcg@20": n_rer},
           "delta_recall@20_pct": (r_rer - r_ease) / r_ease * 100,
           "feature_importance": {f: float(v) for f, v in imp},
           "config": {"topk": TOPK, "n_train_users": int(len(gtr)),
                      "n_estimators": 600, "feats": FEATS}}
json.dump(metrics, open("/kaggle/working/capstone_metrics.json", "w"), indent=2, ensure_ascii=False)
print("\n-> /kaggle/working/rerank_lgbm.txt + capstone_metrics.json")
