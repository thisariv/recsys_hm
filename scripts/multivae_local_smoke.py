"""Сравнение локального NumPy-инференса MultiVAE с forward из RecBole."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.lightgcn import gen_interactions, gen_valid, write_inter
from src.models.multivae import MultiVAEScorer, _extract_mlp

TMP = Path("artifacts/multivae/_smoke")
SEED = 2026


def main():
    train = pd.read_parquet("artifacts/split/train.parquet")
    hold = pd.read_parquet("artifacts/split/holdouts.parquet")
    rng = np.random.default_rng(SEED)
    users = set(rng.choice(train["user_id"].unique(), 2000, replace=False).tolist())
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
    print(f"[smoke] users={len(users)} edges={len(gen_interactions(tr))} catalog={len(cat)}")

    try:
        import torch
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.model.general_recommender.multivae import MultiVAE
        from recbole.trainer import Trainer
        from recbole.utils import init_seed
        from threadpoolctl import threadpool_limits

        cfg = Config(model="MultiVAE", config_dict={
            "data_path": str(TMP), "dataset": "hm",
            "benchmark_filename": ["train", "valid", "test"],
            "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
            "load_col": {"inter": ["user_id", "item_id"]},
            "mlp_hidden_size": [200], "latent_dimension": 64, "dropout_prob": 0.5,
            "anneal_cap": 0.2, "total_anneal_steps": 200000,
            "epochs": 2, "train_batch_size": 2048, "eval_batch_size": 4096,
            "learning_rate": 0.001, "train_neg_sample_args": None,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "RO", "mode": "full"},
            "metrics": ["Recall"], "topk": [20], "valid_metric": "Recall@20",
            "device": "cpu", "use_gpu": False, "seed": SEED, "reproducibility": True,
            "show_progress": False,
        })
        init_seed(cfg["seed"], cfg["reproducibility"])
        dataset = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, dataset)
        model = MultiVAE(cfg, train_data.dataset).to("cpu")
        trainer = Trainer(cfg, model)
        with threadpool_limits(limits=1, user_api="blas"):
            trainer.fit(train_data, valid_data, saved=False, show_progress=False)
        model.eval()

        n_rb = model.n_items
        x = np.zeros(n_rb, dtype=np.float64)
        x[[3, 7, 15, 22]] = 1.0
        with torch.no_grad():
            scores_rb, _, _ = model.forward(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        scores_rb = scores_rb[0].numpy()

        state = {k: v.cpu() for k, v in model.state_dict().items()}
        enc = _extract_mlp(state, "encoder"); dec = _extract_mlp(state, "decoder")
        token2id = {str(k): int(v) for k, v in dataset.field2token_id["item_id"].items()}
        scorer = MultiVAEScorer(enc, dec, token2id, cat)
        scores_mine = scorer._forward(x)

        print(f"[smoke] n_items_rb={n_rb} enc-слоёв={len(enc)} dec-слоёв={len(dec)} lat_dim={enc[-1][0].shape[0]}")
        print(f"[smoke] max|RecBole - наш| = {np.max(np.abs(scores_rb - scores_mine)):.2e}")
        assert np.allclose(scores_rb, scores_mine, atol=1e-4), "РАССИНХРОН forward!"
        some_u = int(tr["user_id"].iloc[0]); pre = tr[tr.user_id == some_u]["item_id"].tolist()
        sc = scorer.score(some_u, pre)
        print(f"[smoke] scorer.score shape={sc.shape} n_missing={scorer.n_missing}")
        assert sc.shape == (len(cat),)
        print("[smoke] NumPy inference matches RecBole forward")
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        shutil.rmtree(TMP, ignore_errors=True)
        sys.exit(3)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
