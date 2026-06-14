"""Общий evaluation pipeline для всех моделей."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from src.eval.metrics import mrr, ndcg_at_k, rank_of_target, recall_at_k

K_VALUES = (10, 20)


class Scorer(Protocol):
    """Модель возвращает скоры для всего train-каталога."""

    def score(self, user_id: int, prefix_items: Sequence[int]) -> np.ndarray: ...


def build_prefix_mask(
    prefix_raw: list[int], v_train: set[int]
) -> tuple[list[int], set[int]]:
    """Убрать OOV-товары из истории и построить маску просмотренного."""
    prefix_in_v = [it for it in prefix_raw if it in v_train]
    return prefix_in_v, set(prefix_in_v)


def _load_split(split_dir: Path):
    train = pd.read_parquet(split_dir / "train.parquet")
    holdouts = pd.read_parquet(split_dir / "holdouts.parquet")
    catalog = pd.read_parquet(split_dir / "catalog.parquet")["item_id"].to_numpy()
    return train, holdouts, catalog


def _train_sequences(train: pd.DataFrame, users: set[int]) -> dict[int, list[int]]:
    """Собрать истории пользователей в хронологическом порядке."""
    sub = train[train["user_id"].isin(users)].sort_values(
        ["user_id", "t_dat", "item_id"], kind="mergesort"
    )
    return {int(u): g.tolist() for u, g in sub.groupby("user_id")["item_id"]}


def evaluate(
    scorer: Scorer,
    split: str,
    config: dict[str, Any],
    *,
    log_mlflow: bool = True,
    model_name: str = "unknown",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Оценить модель на validation или test.

    Для test к train-истории добавляется validation-событие. Ранжирование
    выполняется по полному каталогу, уже купленные товары исключаются.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split должен быть val|test, получено {split!r}")

    split_dir = Path(config["paths"]["split_dir"])
    train, holdouts, catalog = _load_split(split_dir)

    v_train = set(int(x) for x in catalog.tolist())
    cat_pos = {int(it): i for i, it in enumerate(catalog.tolist())}

    flag_col = "is_val_eval" if split == "val" else "is_test_eval"
    target_col = "val_item" if split == "val" else "test_item"
    elig = holdouts[holdouts[flag_col]].copy()

    if split == "test":
        print(
            "[eval] test split: гиперпараметры лучше подбирать только по validation."
        )

    eval_users = set(int(u) for u in elig["user_id"].tolist())
    train_seq = _train_sequences(train, eval_users)
    val_item_by_user = dict(zip(holdouts["user_id"], holdouts["val_item"]))

    n = len(elig)
    sums = {f"recall@{k}": 0.0 for k in K_VALUES}
    sums.update({f"ndcg@{k}": 0.0 for k in K_VALUES})
    sums["mrr"] = 0.0

    for user_id, target in zip(elig["user_id"].to_numpy(), elig[target_col].to_numpy()):
        user_id = int(user_id)
        prefix_raw = list(train_seq.get(user_id, []))
        if split == "test":
            prefix_raw.append(int(val_item_by_user[user_id]))

        prefix_items, mask = build_prefix_mask(prefix_raw, v_train)
        scores = scorer.score(user_id, prefix_items)
        rank = rank_of_target(scores, int(target), mask, catalog, cat_pos)

        for k in K_VALUES:
            sums[f"recall@{k}"] += recall_at_k(rank, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(rank, k)
        sums["mrr"] += mrr(rank)

    metrics = {key: (val / n if n else 0.0) for key, val in sums.items()}
    result = {
        "split": split,
        "model": model_name,
        "n_cases": n,
        "catalog_size": len(catalog),
        **metrics,
    }

    print(
        f"[eval] split={split} model={model_name} n_cases={n:,} | "
        + " ".join(f"{k}={metrics[k]:.6f}" for k in
                   [f"recall@{K_VALUES[0]}", f"recall@{K_VALUES[1]}",
                    f"ndcg@{K_VALUES[0]}", f"ndcg@{K_VALUES[1]}", "mrr"])
    )

    if log_mlflow:
        _log_mlflow(result, config, extra_params or {})

    return result


def _log_mlflow(result: dict[str, Any], config: dict[str, Any], extra_params: dict[str, Any]) -> None:
    """Записать параметры и метрики запуска в MLflow."""
    import mlflow

    from src.utils.tracking import init_mlflow

    init_mlflow(config)
    with mlflow.start_run(run_name=f"eval-{result['model']}-{result['split']}"):
        mlflow.set_tag("stage", "eval")
        mlflow.set_tag("split", result["split"])
        mlflow.set_tag("model", result["model"])
        mlflow.log_params(
            {
                "seed": config["seed"],
                "split": result["split"],
                "model": result["model"],
                "catalog_size": result["catalog_size"],
                **{f"hp_{k}": v for k, v in extra_params.items()},
            }
        )
        # В именах метрик MLflow нельзя использовать @.
        mlflow.log_metrics(
            {k.replace("@", "_at_"): float(v) for k, v in result.items()
             if isinstance(v, (int, float)) and k != "catalog_size"}
        )
