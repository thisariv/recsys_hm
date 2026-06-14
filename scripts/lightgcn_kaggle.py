"""Обучение LightGCN в Kaggle и сохранение эмбеддингов для локальной оценки."""

# Cell 0: зависимости и compatibility patches
import subprocess, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "recbole==1.2.0", "kmeans_pytorch"], check=True)
# RecBole 1.2.0 обращается к алиасам, удалённым в NumPy 2.
import numpy as np
np.float_ = getattr(np, "float_", np.float64)
np.complex_ = getattr(np, "complex_", np.complex128)
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
# LightGCN в RecBole использует удалённый метод scipy.dok_matrix._update.
import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = lambda self, d: self._dict.update(d)
print("патчи применены (numpy + scipy)")


# Cell 1: данные и обучение
import os, glob, json
import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed

SEED = 2026
rng = np.random.default_rng(SEED)
# Validation ограничен 20 тысячами пользователей, чтобы полный ranking не
# занимал большую часть времени обучения.
cands = glob.glob("/kaggle/input/**/*.train.inter", recursive=True)
assert cands, "не найден *.train.inter — прикреплён ли датасет recsys-hm-lightgcn?"
src_dir = os.path.dirname(cands[0])
DATA_PATH = "/kaggle/working/rb_data"
os.makedirs(f"{DATA_PATH}/hm", exist_ok=True)
for split in ["train", "test"]:
    src = f"{src_dir}/hm.{split}.inter"; dst = f"{DATA_PATH}/hm/hm.{split}.inter"
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
import pandas as pd
vdf = pd.read_csv(f"{src_dir}/hm.valid.inter", sep="\t")
ucol = [c for c in vdf.columns if c.startswith("user_id")][0]
keep = set(rng.choice(vdf[ucol].unique(), min(20000, vdf[ucol].nunique()), replace=False).tolist())
vdf[vdf[ucol].isin(keep)].to_csv(f"{DATA_PATH}/hm/hm.valid.inter", sep="\t", index=False)
print(f"valid подсэмплен до {len(keep)} юзеров; файлы:", os.listdir(f"{DATA_PATH}/hm"))
cfg = Config(model="LightGCN", config_dict={
    "data_path": DATA_PATH, "dataset": "hm",
    "benchmark_filename": ["train", "valid", "test"],
    "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
    "load_col": {"inter": ["user_id", "item_id"]},
    "embedding_size": 64, "n_layers": 2, "reg_weight": 1e-4,
    "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
    "epochs": 80, "train_batch_size": 4096, "eval_batch_size": 8192,
    "learning_rate": 0.001,
    "eval_args": {"split": {"LS": "valid_and_test"}, "order": "RO", "mode": "full"},
    "metrics": ["Recall", "NDCG", "MRR"], "topk": [10, 20], "valid_metric": "Recall@20",
    # Валидация выполняется один раз после последней эпохи.
    "eval_step": 80, "stopping_step": 99,
    "device": "cuda", "use_gpu": True, "seed": SEED, "reproducibility": True,
    "show_progress": False,
})
init_seed(cfg["seed"], cfg["reproducibility"])
dataset = create_dataset(cfg)
train_data, valid_data, test_data = data_preparation(cfg, dataset)
model = LightGCN(cfg, train_data.dataset).to("cuda")
trainer = Trainer(cfg, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
print("\n=== BEST VALID ===")
print(best_valid_result)


# Cell 2: сохранение эмбеддингов
model.eval()
with torch.no_grad():
    user_e, item_e = model.forward()
np.save("/kaggle/working/user_emb.npy", user_e.cpu().numpy())
np.save("/kaggle/working/item_emb.npy", item_e.cpu().numpy())
tok = {"user_id": {str(k): int(v) for k, v in dataset.field2token_id["user_id"].items()},
       "item_id": {str(k): int(v) for k, v in dataset.field2token_id["item_id"].items()}}
json.dump(tok, open("/kaggle/working/token_map.json", "w"))
json.dump({"model": "LightGCN", "best_valid_result": {k: float(v) for k, v in best_valid_result.items()}},
          open("/kaggle/working/meta.json", "w"), indent=2)
print("сохранено: user_emb.npy, item_emb.npy, token_map.json, meta.json")
print("best_valid Recall@20 =", best_valid_result.get("recall@20"))
