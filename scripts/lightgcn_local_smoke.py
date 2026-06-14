"""Локальная проверка обучения и скоринга LightGCN на небольшой выборке."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.lightgcn import LightGCNScorer, gen_interactions, gen_valid, write_inter

# Compatibility patch для RecBole 1.2.0 и новых версий SciPy.
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    def _dok_update(self, data):
        self._dict.update(data)
    _sp.dok_matrix._update = _dok_update

TMP = Path("artifacts/lightgcn/_smoke")
SEED = 2026


def main():
    train = pd.read_parquet("artifacts/split/train.parquet")
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")
    rng = np.random.default_rng(SEED)
    users = set(rng.choice(train["user_id"].unique(), size=3000, replace=False).tolist())
    tr = train[train["user_id"].isin(users)]
    ho = hold[hold["user_id"].isin(users)]
    cat = np.sort(tr["item_id"].unique())
    ds = TMP / "hm"
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    write_inter(gen_interactions(tr), ds / "hm.train.inter")
    v = gen_valid(ho)
    write_inter(v, ds / "hm.valid.inter")
    write_inter(v, ds / "hm.test.inter")
    print(f"[smoke] users={len(users)} edges={len(gen_interactions(tr))} catalog={len(cat)} valid={len(v)}")

    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.model.general_recommender.lightgcn import LightGCN
        from recbole.trainer import Trainer
        from recbole.utils import init_seed
        from threadpoolctl import threadpool_limits

        cfg = Config(model="LightGCN", config_dict={
            "data_path": str(TMP), "dataset": "hm",
            "benchmark_filename": ["train", "valid", "test"],
            "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
            "load_col": {"inter": ["user_id", "item_id"]},
            "embedding_size": 64, "n_layers": 2, "reg_weight": 1e-4,
            "epochs": 2, "train_batch_size": 2048, "eval_batch_size": 4096,
            "learning_rate": 0.001,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1,
                                      "alpha": 1.0, "dynamic": False, "candidate_num": 0},
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "RO", "mode": "full"},
            "metrics": ["Recall"], "topk": [20], "valid_metric": "Recall@20",
            "device": "cpu", "use_gpu": False, "seed": SEED, "reproducibility": True,
            "show_progress": False,
        })
        init_seed(cfg["seed"], cfg["reproducibility"])
        dataset = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, dataset)
        model = LightGCN(cfg, train_data.dataset).to("cpu")
        trainer = Trainer(cfg, model)
        with threadpool_limits(limits=1, user_api="blas"):
            trainer.fit(train_data, valid_data, saved=False, show_progress=False)

        import torch
        model.eval()
        with torch.no_grad():
            user_e, item_e = model.forward()
        user_e = user_e.cpu().numpy(); item_e = item_e.cpu().numpy()
        user_t = {str(k): int(v) for k, v in dataset.field2token_id["user_id"].items()}
        item_t = {str(k): int(v) for k, v in dataset.field2token_id["item_id"].items()}
        print(f"[smoke] forward(): user_e={user_e.shape} item_e={item_e.shape} "
              f"user_vocab={len(user_t)} item_vocab={len(item_t)}")

        scorer = LightGCNScorer(user_e, item_e, user_t, item_t, cat)
        some_user = int(tr["user_id"].iloc[0])
        sc = scorer.score(some_user, [])
        print(f"[smoke] scorer ok: shape={sc.shape} n_missing={scorer.n_missing} "
              f"top1_item={cat[int(np.argmax(sc))]} score_range=[{sc.min():.3f},{sc.max():.3f}]")
        assert sc.shape == (len(cat),) and not np.all(sc == 0)
        print("[smoke] LightGCN pipeline works")
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        shutil.rmtree(TMP, ignore_errors=True)
        sys.exit(3)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
