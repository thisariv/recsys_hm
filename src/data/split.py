"""Temporal leave-last-out split по дням.

Последний день используется для test, предпоследний для validation, а более
ранние события попадают в train. Таргеты должны присутствовать в train-каталоге
и не встречаться в истории пользователя.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def _pct_stats(s: pd.Series) -> dict[str, float]:
    """Краткая статистика распределения."""
    if len(s) == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0}
    return {
        "min": int(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p90": float(s.quantile(0.90)),
        "max": int(s.max()),
    }


def catalog_hash(item_ids: np.ndarray) -> str:
    """Стабильный хеш отсортированного каталога."""
    ordered = np.sort(np.asarray(item_ids, dtype=np.int64))
    payload = ",".join(map(str, ordered.tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_split(
    df: pd.DataFrame, min_days: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Построить train, holdouts, каталог и статистику сплита."""
    df = df.sort_values(["user_id", "t_dat", "item_id"], kind="mergesort").reset_index(drop=True)

    # Один таргет на день; item_id сохраняет порядок исходного article_id.
    day_df = df.groupby(["user_id", "t_dat"], sort=True, as_index=False)["item_id"].first()
    day_df["rev"] = day_df.groupby("user_id").cumcount(ascending=False)
    m_per_user = day_df.groupby("user_id").size().rename("m")

    ge3_users = m_per_user.index[m_per_user >= min_days]

    test_days = day_df[(day_df["rev"] == 0) & day_df["user_id"].isin(ge3_users)]
    val_days = day_df[(day_df["rev"] == 1) & day_df["user_id"].isin(ge3_users)]

    holdouts = (
        test_days[["user_id", "item_id", "t_dat"]]
        .rename(columns={"item_id": "test_item", "t_dat": "test_date"})
        .merge(
            val_days[["user_id", "item_id", "t_dat"]].rename(
                columns={"item_id": "val_item", "t_dat": "val_date"}
            ),
            on="user_id",
        )
        .reset_index(drop=True)
    )

    val_date_map = holdouts[["user_id", "val_date"]]
    merged = df.merge(val_date_map, on="user_id", how="left")
    train_mask = merged["val_date"].isna() | (merged["t_dat"] < merged["val_date"])
    train = merged.loc[train_mask, ["user_id", "item_id", "t_dat"]].reset_index(drop=True)

    catalog = np.sort(train["item_id"].unique())
    v_train = set(catalog.tolist())

    holdouts["val_cold_ok"] = holdouts["val_item"].isin(v_train)
    holdouts["test_cold_ok"] = holdouts["test_item"].isin(v_train)

    train_pairs = train[["user_id", "item_id"]].drop_duplicates()

    def _in_user_train(item_col: str) -> pd.Series:
        probe = holdouts[["user_id", item_col]].rename(columns={item_col: "item_id"})
        hit = probe.merge(train_pairs, on=["user_id", "item_id"], how="left", indicator=True)
        return (hit["_merge"] == "both").to_numpy()

    val_in_train = _in_user_train("val_item")
    test_in_train = _in_user_train("test_item")

    holdouts["val_repeat_ok"] = ~val_in_train
    # Для test val_item уже считается частью истории.
    holdouts["test_repeat_ok"] = (~test_in_train) & (
        holdouts["test_item"] != holdouts["val_item"]
    )

    holdouts["is_val_eval"] = holdouts["val_cold_ok"] & holdouts["val_repeat_ok"]
    holdouts["is_test_eval"] = holdouts["test_cold_ok"] & holdouts["test_repeat_ok"]

    _assert_invariants(train, holdouts, v_train)

    stats = _build_stats(df, train, holdouts, m_per_user, catalog)

    holdouts_out = holdouts[
        ["user_id", "val_item", "val_date", "test_item", "test_date",
         "is_val_eval", "is_test_eval"]
    ].sort_values("user_id").reset_index(drop=True)

    return train, holdouts_out, catalog, stats


def _assert_invariants(train: pd.DataFrame, holdouts: pd.DataFrame, v_train: set) -> None:
    """Проверить временной порядок и допустимость таргетов."""
    if holdouts.empty:
        return

    train_max = train.groupby("user_id")["t_dat"].max().rename("train_max")
    chk = holdouts.merge(train_max, on="user_id", how="left")
    assert chk["train_max"].notna().all(), "у m>=3 юзера нет train-событий"
    assert (chk["train_max"] < chk["val_date"]).all(), "train_max >= val_date"
    assert (chk["val_date"] < chk["test_date"]).all(), "val_date >= test_date"

    bad = train.merge(
        holdouts[["user_id", "val_date"]], on="user_id", how="inner"
    )
    assert (bad["t_dat"] < bad["val_date"]).all(), "train содержит событие >= val_date"

    elig_val = holdouts.loc[holdouts["is_val_eval"], "val_item"]
    elig_test = holdouts.loc[holdouts["is_test_eval"], "test_item"]
    assert elig_val.isin(v_train).all(), "is_val_eval, но val_item ∉ V_train"
    assert elig_test.isin(v_train).all(), "is_test_eval, но test_item ∉ V_train"


def _build_stats(
    df: pd.DataFrame,
    train: pd.DataFrame,
    holdouts: pd.DataFrame,
    m_per_user: pd.Series,
    catalog: np.ndarray,
    red_flag_frac: float = 0.40,
) -> dict[str, Any]:
    """Собрать статистику по отфильтрованным eval-кейсам."""
    n_users_total = int(df["user_id"].nunique())
    n_ge3 = int(len(holdouts))

    n_drop_cold_test = int((~holdouts["test_cold_ok"]).sum())
    n_drop_repeat_test = int((holdouts["test_cold_ok"] & ~holdouts["test_repeat_ok"]).sum())
    n_test_eval = int(holdouts["is_test_eval"].sum())

    n_drop_cold_val = int((~holdouts["val_cold_ok"]).sum())
    n_drop_repeat_val = int((holdouts["val_cold_ok"] & ~holdouts["val_repeat_ok"]).sum())
    n_val_eval = int(holdouts["is_val_eval"].sum())

    cold_survivors_test = n_ge3 - n_drop_cold_test
    repeat_drop_frac_test = (
        n_drop_repeat_test / cold_survivors_test if cold_survivors_test else 0.0
    )
    red_flag = repeat_drop_frac_test > red_flag_frac

    train_prefix_len = train[train["user_id"].isin(holdouts["user_id"])].groupby("user_id").size()

    def _frac(n: int) -> float:
        return n / n_ge3 if n_ge3 else 0.0

    return {
        "funnel": {
            "n_users_total": n_users_total,
            "n_users_ge3_days": n_ge3,
            "test": {
                "n_ge3": n_ge3,
                "n_dropped_cold": n_drop_cold_test,
                "frac_dropped_cold": _frac(n_drop_cold_test),
                "n_dropped_repeat": n_drop_repeat_test,
                "frac_dropped_repeat": _frac(n_drop_repeat_test),
                "n_test_eval": n_test_eval,
                "frac_test_eval": _frac(n_test_eval),
            },
            "val": {
                "n_ge3": n_ge3,
                "n_dropped_cold": n_drop_cold_val,
                "frac_dropped_cold": _frac(n_drop_cold_val),
                "n_dropped_repeat": n_drop_repeat_val,
                "frac_dropped_repeat": _frac(n_drop_repeat_val),
                "n_val_eval": n_val_eval,
                "frac_val_eval": _frac(n_val_eval),
            },
            "n_dropped_cold_test": n_drop_cold_test,
            "n_dropped_repeat_test": n_drop_repeat_test,
            "n_dropped_cold_val": n_drop_cold_val,
            "n_dropped_repeat_val": n_drop_repeat_val,
        },
        "repeat_drop_frac_test": repeat_drop_frac_test,
        "red_flag_repeat_drop": bool(red_flag),
        "distinct_days_per_user": _pct_stats(m_per_user),
        "train_prefix_len": _pct_stats(train_prefix_len),
        "n_v_train": int(len(catalog)),
        "n_interactions_train": int(len(train)),
        "catalog_hash": catalog_hash(catalog),
    }
