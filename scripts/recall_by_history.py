"""Recall@K по группам пользователей с разной длиной истории."""
from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.split import catalog_hash
from src.eval.metrics import rank_of_target

SAMPLE = 30000
BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-7", 5, 7), ("8+", 8, 10**9)]


def bucket_name(d):
    for name, lo, hi in BUCKETS:
        if lo <= d <= hi:
            return name
    return "8+"


def model_loaders(cfg, catalog):
    """Создать ленивые загрузчики доступных моделей."""
    md = Path(cfg["paths"]["metrics_dir"]).parent / "models"
    ch = catalog_hash(catalog)
    L = {}

    def _train():
        return pd.read_parquet(Path(cfg["paths"]["split_dir"]) / "train.parquet")

    L["popularity"] = lambda: __import__("src.models.popularity", fromlist=["PopularityScorer"]).PopularityScorer("popularity").fit(_train(), catalog)
    L["popularity_decay"] = lambda: __import__("src.models.popularity", fromlist=["PopularityScorer"]).PopularityScorer("popularity_decay", cfg["models"]["popularity"]["half_life_days"]).fit(_train(), catalog)
    L["itemknn"] = lambda: __import__("src.models.itemknn", fromlist=["ItemKNNScorer"]).ItemKNNScorer.load(md / "itemknn", catalog, cfg["models"]["itemknn"]["k_neighbors"], cfg["models"]["itemknn"]["aggregation"])
    L["als"] = lambda: __import__("src.models.als", fromlist=["ALSScorer"]).ALSScorer.load(md / "als", catalog, factors=cfg["models"]["als"]["factors"], regularization=cfg["models"]["als"]["regularization"], iterations=cfg["models"]["als"]["iterations"], alpha=cfg["models"]["als"]["alpha"], seed=cfg["seed"])
    if (Path("artifacts/ease/checkpoint/ease_B_topk.npz").exists()):
        L["ease"] = lambda: __import__("src.models.ease", fromlist=["load_real_ease"]).load_real_ease("artifacts/ease/checkpoint", catalog, expected_hash=ch)
    if (Path("artifacts/sasrec/checkpoint/token_map.json").exists()):
        L["sasrec"] = lambda: __import__("src.sasrec.scorer", fromlist=["load_real_sasrec"]).load_real_sasrec("artifacts/sasrec/checkpoint", catalog, cfg, expected_hash=ch)
    if (Path("artifacts/gru4rec/checkpoint/token_map.json").exists()):
        L["gru4rec"] = lambda: __import__("src.sasrec.scorer", fromlist=["load_real_sasrec"]).load_real_sasrec("artifacts/gru4rec/checkpoint", catalog, cfg, expected_hash=ch)
    if (Path("artifacts/multivae/checkpoint/token_map.json").exists()):
        L["multivae"] = lambda: __import__("src.models.multivae", fromlist=["load_real_multivae"]).load_real_multivae("artifacts/multivae/checkpoint", catalog, expected_hash=ch)
    if (Path("artifacts/lightgcn/checkpoint/user_emb.npy").exists()):
        L["lightgcn"] = lambda: __import__("src.models.lightgcn", fromlist=["load_real_lightgcn"]).load_real_lightgcn("artifacts/lightgcn/checkpoint", catalog, expected_hash=ch)
    return L


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default=None,
        help="список моделей через запятую; результаты добавятся в существующий JSON",
    )
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    cfg = load_config("configs/base.yaml")
    catalog = pd.read_parquet("artifacts/split/catalog.parquet")["item_id"].to_numpy()
    cat_pos = {int(c): i for i, c in enumerate(catalog.tolist())}
    train = pd.read_parquet("artifacts/split/train.parquet")
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")

    items = train.sort_values(["user_id", "t_dat", "item_id"]).groupby("user_id")["item_id"].apply(list)
    ndays = train.groupby("user_id")["t_dat"].nunique()

    val = hold[hold["is_val_eval"]].sample(SAMPLE, random_state=42)
    cases = []
    for u, tgt in zip(val["user_id"].to_numpy(), val["val_item"].to_numpy()):
        pre = items.get(u, [])
        if not pre:
            continue
        mask = set(int(x) for x in pre)
        if int(tgt) in mask:
            continue
        cases.append((int(u), int(tgt), pre, mask, bucket_name(int(ndays.get(u, 1)))))
    del items, ndays, train, hold
    gc.collect()
    print(f"кейсов: {len(cases)}")

    loaders = model_loaders(cfg, catalog)
    bucket_order = [b[0] for b in BUCKETS]
    out_path = Path("artifacts/metrics/recall_by_history.json")
    if only and out_path.exists():
        out = json.loads(out_path.read_text())
        loaders = {k: v for k, v in loaders.items() if k in only}
        print(f"[merge] считаю только {sorted(loaders)} -> доклею к {sorted(out['models'])}")
    else:
        out = {"sample": SAMPLE, "buckets_days": bucket_order, "models": {}}

    for name, load in loaders.items():
        sc = load()
        agg = defaultdict(lambda: [0, 0, 0])
        for u, tgt, pre, mask, b in cases:
            r = rank_of_target(sc.score(u, pre), tgt, mask, catalog, cat_pos)
            a = agg[b]; a[0] += 1; a[1] += r <= 10; a[2] += r <= 20
        out["models"][name] = {bb: {"n": agg[bb][0],
                                    "recall@10": agg[bb][1] / agg[bb][0] if agg[bb][0] else 0.0,
                                    "recall@20": agg[bb][2] / agg[bb][0] if agg[bb][0] else 0.0}
                               for bb in bucket_order}
        r20 = " ".join(f"{out['models'][name][bb]['recall@20']:.4f}" for bb in bucket_order)
        print(f"[{name:16s}] R@20 по бакетам: {r20}", flush=True)
        del sc; gc.collect()

    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    with open("artifacts/metrics/recall_by_history.json", "w") as f:
        json.dump(out, f, indent=2)
    ns = {bb: next(iter(out["models"].values()))[bb]["n"] for bb in bucket_order}
    print("\n### Recall@20 vs дни истории (val, n=%d)\n" % len(cases))
    print("| модель | " + " | ".join(f"{bb} (n={ns[bb]})" for bb in bucket_order) + " |")
    print("|" + "---|" * (len(bucket_order) + 1))
    for name in out["models"]:
        print(f"| {name} | " + " | ".join(f"{out['models'][name][bb]['recall@20']:.4f}" for bb in bucket_order) + " |")
    print("\n-> artifacts/metrics/recall_by_history.json")


if __name__ == "__main__":
    main()
