"""Локальное сравнение режимов next_day и next_item для SASRec."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.sasrec.atomic import gen_train_inter, gen_valid_inter, write_inter
from src.models.popularity import PopularityScorer
from src.models.itemknn import ItemKNNScorer
from src.sasrec.scorer import SASRecScorer
from src.eval.metrics import rank_of_target

SEED = 2026
N_USERS = 30000
EPOCHS = 12
TMP = Path("artifacts/sasrec/_validate")


def build_subsample():
    train = pd.read_parquet("artifacts/split/train.parquet")
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")
    ndays = train.groupby("user_id")["t_dat"].nunique()
    ge2 = ndays.index[ndays >= 2].to_numpy()
    rng = np.random.default_rng(SEED)
    users = set(rng.choice(ge2, size=min(N_USERS, len(ge2)), replace=False).tolist())
    tr = train[train["user_id"].isin(users)].copy()
    ho = hold[hold["user_id"].isin(users)].copy()
    cat = np.sort(tr["item_id"].unique())
    print(f"[val] subsample users={len(users)} train_rows={len(tr)} catalog={len(cat)} "
          f"val_eval={int(ho['is_val_eval'].sum())}")
    return tr, ho, cat


def write_atomic(tr, ho, mode, ds_dir):
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True)
    write_inter(gen_train_inter(tr, mode=mode), ds_dir / "hm.train.inter")
    v = gen_valid_inter(tr, ho)
    write_inter(v, ds_dir / "hm.valid.inter")
    write_inter(v, ds_dir / "hm.test.inter")


def train_recbole(ds_parent):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender.sasrec import SASRec
    from recbole.trainer import Trainer
    from recbole.utils import init_seed
    from threadpoolctl import threadpool_limits

    cfg = Config(model="SASRec", dataset="hm", config_dict={
        "data_path": str(ds_parent),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list", "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": 50, "benchmark_filename": ["train", "valid", "test"],
        "load_col": {"inter": ["user_id", "item_id_list", "item_id", "item_length"]},
        "alias_of_item_id": ["item_id_list"],
        "epochs": EPOCHS, "train_batch_size": 2048, "eval_batch_size": 4096,
        "hidden_size": 64, "inner_size": 128, "n_layers": 2, "n_heads": 2,
        "hidden_dropout_prob": 0.2, "attn_dropout_prob": 0.2, "loss_type": "CE",
        "train_neg_sample_args": None,
        "learning_rate": 0.001, "device": "cpu", "use_gpu": False,
        "seed": SEED, "reproducibility": True, "show_progress": False,
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full"},
        "metrics": ["Recall"], "topk": [20], "valid_metric": "Recall@20",
        "stopping_step": 100, "checkpoint_dir": str(ds_parent / "ckpt"),
    })
    init_seed(cfg["seed"], cfg["reproducibility"])
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    model = SASRec(cfg, train_data.dataset).to("cpu")
    trainer = Trainer(cfg, model)
    with threadpool_limits(limits=1, user_api="blas"):
        trainer.fit(train_data, valid_data, saved=False, show_progress=False)
    token2id = {str(k): int(v) for k, v in dataset.field2token_id[cfg["ITEM_ID_FIELD"]].items()}
    model.eval()
    return model, token2id


class _Wrap:
    def __init__(self, model):
        import torch
        self.t = torch; self.m = model
    def full_sort_predict(self, seqs, lengths):
        from recbole.data.interaction import Interaction
        t = self.t
        inter = Interaction({self.m.ITEM_SEQ: t.as_tensor(np.asarray(seqs), dtype=t.long),
                             self.m.ITEM_SEQ_LEN: t.as_tensor(np.asarray(lengths), dtype=t.long)})
        with t.no_grad():
            s = self.m.full_sort_predict(inter)
        return s.view(len(lengths), -1).cpu().numpy()


def evaluate(mode, tr, ho, cat):
    ds_dir = TMP / mode / "hm"
    write_atomic(tr, ho, mode, ds_dir)
    print(f"[val:{mode}] обучаю SASRec {EPOCHS} эпох на CPU ...")
    model, token2id = train_recbole(TMP / mode)
    scorer = SASRecScorer(_Wrap(model), token2id, cat, max_seq_len=50)
    cat_pos = {int(c): i for i, c in enumerate(cat.tolist())}

    pop = PopularityScorer("popularity").fit(tr, cat).score(0, [])
    knn = ItemKNNScorer(200, "sum").fit(tr, cat)

    seqs = tr.sort_values(["user_id", "t_dat", "item_id"]).groupby("user_id")["item_id"].apply(list)
    elig = ho[ho["is_val_eval"]]
    rng = np.random.default_rng(0)
    rows = elig.sample(min(4000, len(elig)), random_state=SEED)

    hit_s = hit_p = hit_k = n = 0
    for u, tgt in zip(rows["user_id"].to_numpy(), rows["val_item"].to_numpy()):
        pre = seqs.get(u, [])
        if not pre or str(int(tgt)) not in scorer.token2id or int(tgt) not in cat_pos:
            continue
        mask = set(int(x) for x in pre)
        if int(tgt) in mask:
            continue
        rs = rank_of_target(scorer.score(int(u), pre), int(tgt), mask, cat, cat_pos)
        rp = rank_of_target(pop, int(tgt), mask, cat, cat_pos)
        rk = rank_of_target(knn.score(int(u), pre), int(tgt), mask, cat, cat_pos)
        hit_s += rs <= 20; hit_p += rp <= 20; hit_k += rk <= 20; n += 1
    print(f"\n[val:{mode}] next-new-item Recall@20 на сабсэмпле (n={n}):")
    print(f"        SASRec({mode})={hit_s/n:.4f}   popularity={hit_p/n:.4f}   itemknn={hit_k/n:.4f}")

    # Сходство рекомендаций SASRec и ItemKNN для короткого префикса.
    probes = tr["item_id"].value_counts().index.to_numpy()[:200]
    probes = rng.choice(probes, 60, replace=False)
    jac = []
    for it in probes:
        if str(int(it)) not in scorer.token2id:
            continue
        a = set(np.argsort(-scorer.score(0, [int(it)]))[:20])
        b = set(np.argsort(-knn.score(0, [int(it)]))[:20])
        jac.append(len(a & b) / len(a | b))
    print(f"[val:{mode}] Jaccard(SASRec, ItemKNN) single-item = {np.mean(jac):.3f}")
    return hit_s / n


def main():
    tr, ho, cat = build_subsample()
    try:
        r_new = evaluate("next_day", tr, ho, cat)
        r_old = evaluate("next_item", tr, ho, cat)
    except Exception as e:
        print(f"[val] FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(3)
    print(f"\n[val] сравнение режимов:")
    print(f"      next_day  Recall@20 = {r_new:.4f}")
    print(f"      next_item Recall@20 = {r_old:.4f}")
    print(f"      -> {'next_day лучше' if r_new > r_old*1.5 else 'разница небольшая'}")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
