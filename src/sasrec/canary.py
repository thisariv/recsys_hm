"""Проверка sequence-адаптера на случайной модели."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.harness import evaluate
from src.sasrec.scorer import SASRecScorer, build_token2id
from src.sasrec.stub import StubSASRec

# Случайный адаптер не должен заметно превосходить random baseline.
CANARY_MAX_RATIO = 2.0


def run_canary(config: dict[str, Any], split: str = "val") -> dict[str, Any]:
    catalog = pd.read_parquet(
        Path(config["paths"]["split_dir"]) / "catalog.parquet"
    )["item_id"].to_numpy()
    n_items = len(catalog)

    sasrec_cfg = config.get("models", {}).get("sasrec", {})
    d = sasrec_cfg.get("d_model", 64)
    seed = sasrec_cfg.get("seed", 2026)

    stub = StubSASRec(n_items=n_items, d=d, seed=seed)
    token2id = build_token2id(catalog)
    scorer = SASRecScorer(stub, token2id, catalog,
                          max_seq_len=sasrec_cfg.get("max_seq_len", 50))

    result = evaluate(scorer, split, config, log_mlflow=True, model_name="canary")

    random_r20 = 20.0 / n_items
    ratio = result["recall@20"] / random_r20 if random_r20 else float("inf")
    result["random_recall@20"] = random_r20
    result["canary_ratio"] = ratio

    print(
        f"[canary] Recall@20={result['recall@20']:.6f} vs random≈{random_r20:.6f} "
        f"(ratio {ratio:.2f}x)"
    )
    if ratio > CANARY_MAX_RATIO:
        print(
            f"[canary] подозрительно высокий результат случайной модели: {ratio:.2f}x "
            f"(порог {CANARY_MAX_RATIO}x). Проверь отображение ID и маскирование."
        )
        result["bridge_clean"] = False
    else:
        print(f"[canary] результат близок к random (ratio {ratio:.2f}x)")
        result["bridge_clean"] = True

    return result
