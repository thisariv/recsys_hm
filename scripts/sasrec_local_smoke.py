"""Локальная проверка RecBole SASRec на небольшой выборке."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import catalog_hash
from src.sasrec.atomic import gen_train_inter, gen_valid_inter, write_inter

SPLIT = Path("artifacts/split")
SMOKE = Path("artifacts/sasrec/smoke")
DATASET = "hm"
SEED = 2026
USER_FRAC = 0.01


def build_subsample_atomic():
    train = pd.read_parquet(SPLIT / "train.parquet")
    holdouts = pd.read_parquet(SPLIT / "holdouts.parquet")
    rng = np.random.default_rng(SEED)
    users = train["user_id"].unique()
    keep = set(rng.choice(users, size=max(1, int(len(users) * USER_FRAC)), replace=False))
    tr = train[train["user_id"].isin(keep)].copy()
    ho = holdouts[holdouts["user_id"].isin(keep)].copy()

    ds_dir = SMOKE / DATASET
    write_inter(gen_train_inter(tr), ds_dir / f"{DATASET}.train.inter")
    write_inter(gen_valid_inter(tr, ho), ds_dir / f"{DATASET}.valid.inter")
    write_inter(gen_valid_inter(tr, ho), ds_dir / f"{DATASET}.test.inter")
    print(f"[smoke] 1% atomic: {len(keep)} users -> {ds_dir}/")
    return ds_dir


def run_recbole_smoke(ds_dir: Path):
    # quick_start импортирует необязательные зависимости для hyperparameter tuning.
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    # Прямой импорт не загружает остальные sequential-модели RecBole.
    from recbole.model.sequential_recommender.sasrec import SASRec
    from recbole.trainer import Trainer
    from recbole.utils import init_seed

    config_dict = {
        "data_path": str(SMOKE),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": 50,
        "benchmark_filename": ["train", "valid", "test"],
        "load_col": {"inter": ["user_id", "item_id_list", "item_id", "item_length"]},
        "alias_of_item_id": ["item_id_list"],
        "epochs": 1,
        "train_batch_size": 256,
        "eval_batch_size": 256,
        "hidden_size": 64,
        "n_layers": 2,
        "n_heads": 2,
        "loss_type": "CE",
        "device": "cpu",
        "use_gpu": False,
        "seed": SEED,
        "reproducibility": True,
        "eval_args": {"order": "TO", "mode": "full", "split": {"LS": "valid_and_test"}},
        "metrics": ["Recall"],
        "topk": [20],
        "valid_metric": "Recall@20",
        "show_progress": False,
        "checkpoint_dir": str(SMOKE / "ckpt"),
    }
    print("[smoke] Config/create_dataset/Trainer (SASRec, hm, 1 epoch CPU) ...")
    config = Config(model="SASRec", dataset=DATASET, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    print(f"[smoke] dataset: {dataset}")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = SASRec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    best_valid_score, _ = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    print(f"[smoke] обучение прошло. checkpoint={trainer.saved_model_file} "
          f"(RecBole valid={best_valid_score})")
    return trainer.saved_model_file


def main():
    ds_dir = build_subsample_atomic()
    try:
        run_recbole_smoke(ds_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}")
        print("[smoke] локальный запуск RecBole не удался; подробности выше.")
        sys.exit(3)
    print("[smoke] OK: пайплайн прошёл локально на 1%.")
    print(f"[smoke] catalog_hash={catalog_hash(pd.read_parquet(SPLIT / 'catalog.parquet')['item_id'].to_numpy())}")


if __name__ == "__main__":
    main()
