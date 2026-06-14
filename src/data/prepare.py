"""Предобработка транзакций и описаний товаров H&M."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Поля, из которых собирается текстовое описание товара.
TEXT_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "section_name",
    "detail_desc",
]


def filter_last_n_months(
    df: pd.DataFrame, last_n_months: int, date_col: str = "t_dat"
) -> pd.DataFrame:
    """Оставить последние `last_n_months` относительно максимальной даты в данных."""
    max_date = df[date_col].max()
    start = max_date - pd.DateOffset(months=last_n_months)
    return df[df[date_col] > start].copy()


def dedup_interactions(df: pd.DataFrame, grain: list[str]) -> pd.DataFrame:
    """Удалить повторные покупки одного товара в один день."""
    raw_grain = [_GRAIN_TO_RAW.get(c, c) for c in grain]
    return df.drop_duplicates(subset=raw_grain).copy()


# В конфиге используются итоговые имена колонок, а здесь данные ещё сырые.
_GRAIN_TO_RAW = {"user_id": "customer_id", "item_id": "article_id"}


def subsample_users(
    df: pd.DataFrame, max_users: int | None, seed: int, user_col: str = "customer_id"
) -> pd.DataFrame:
    """Оставить случайную выборку пользователей, если задан лимит."""
    if max_users is None:
        return df
    unique_users = df[user_col].unique()
    if len(unique_users) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(unique_users, size=max_users, replace=False)
    return df[df[user_col].isin(keep)].copy()


def iterative_kcore(
    df: pd.DataFrame,
    min_user: int,
    min_item: int,
    user_col: str = "customer_id",
    item_col: str = "article_id",
) -> pd.DataFrame:
    """Итеративно удалить редких пользователей и товары до стабилизации."""
    df = df.copy()
    while True:
        n_before = len(df)

        user_counts = df[user_col].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        df = df[df[user_col].isin(keep_users)]

        item_counts = df[item_col].value_counts()
        keep_items = item_counts[item_counts >= min_item].index
        df = df[df[item_col].isin(keep_items)]

        if len(df) == n_before:
            break

    assert_kcore_invariant(df, min_user, min_item, user_col, item_col)
    return df.copy()


def assert_kcore_invariant(
    df: pd.DataFrame,
    min_user: int,
    min_item: int,
    user_col: str = "customer_id",
    item_col: str = "article_id",
) -> None:
    """Проверить минимальные степени после k-core фильтрации."""
    if df.empty:
        return
    min_u = df[user_col].value_counts().min()
    min_i = df[item_col].value_counts().min()
    assert min_u >= min_user, f"k-core нарушен: юзер с {min_u} < {min_user}"
    assert min_i >= min_item, f"k-core нарушен: товар с {min_i} < {min_item}"


def build_dense_ids(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Заменить исходные идентификаторы плотными целыми ID."""
    unique_users = np.sort(df["customer_id"].unique())
    unique_items = np.sort(df["article_id"].unique())

    user_map = pd.DataFrame(
        {"customer_id": unique_users, "user_id": np.arange(len(unique_users), dtype=np.int64)}
    )
    item_map = pd.DataFrame(
        {"article_id": unique_items, "item_id": np.arange(len(unique_items), dtype=np.int64)}
    )

    u_lookup = dict(zip(user_map["customer_id"], user_map["user_id"]))
    i_lookup = dict(zip(item_map["article_id"], item_map["item_id"]))

    out = pd.DataFrame(
        {
            "user_id": df["customer_id"].map(u_lookup).astype(np.int64),
            "item_id": df["article_id"].map(i_lookup).astype(np.int64),
            "t_dat": df["t_dat"].values,
        }
    )
    return out, user_map, item_map


def sort_interactions(df: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Отсортировать события по пользователю, дате и исходному article_id."""
    i_to_article = dict(zip(item_map["item_id"], item_map["article_id"]))
    df = df.copy()
    df["_tie"] = df["item_id"].map(i_to_article)
    df = df.sort_values(["user_id", "t_dat", "_tie"], kind="mergesort").reset_index(drop=True)
    return df.drop(columns="_tie")


def build_items_text(
    articles_path: Path, kept_item_map: pd.DataFrame
) -> pd.DataFrame:
    """Собрать текст и категориальные признаки для оставшихся товаров."""
    kept_articles = set(kept_item_map["article_id"].tolist())
    articles = pd.read_csv(
        articles_path,
        usecols=["article_id"] + TEXT_COLUMNS,
        dtype={"article_id": np.int64},
    )
    articles = articles[articles["article_id"].isin(kept_articles)].copy()

    for col in TEXT_COLUMNS:
        articles[col] = articles[col].fillna("").astype(str)
    articles["text"] = articles[TEXT_COLUMNS].agg(" ".join, axis=1).str.strip()

    merged = kept_item_map.merge(articles, on="article_id", how="left")
    cols = ["item_id", "article_id", "text"] + TEXT_COLUMNS
    return merged[cols].sort_values("item_id").reset_index(drop=True)


def compute_stats(
    interactions: pd.DataFrame,
    counts_before: dict[str, int],
    last_n_months: int,
    seed: int,
) -> dict[str, Any]:
    """Посчитать базовую статистику подготовленного датасета."""
    n_users = int(interactions["user_id"].nunique())
    n_items = int(interactions["item_id"].nunique())
    n_interactions = int(len(interactions))

    seq_lens = interactions.groupby("user_id").size()
    density = n_interactions / (n_users * n_items) if n_users and n_items else 0.0

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density": density,
        "seq_len": {
            "min": int(seq_lens.min()),
            "median": float(seq_lens.median()),
            "mean": float(seq_lens.mean()),
            "p90": float(seq_lens.quantile(0.90)),
            "max": int(seq_lens.max()),
        },
        "date_range": {
            "min": str(interactions["t_dat"].min().date()),
            "max": str(interactions["t_dat"].max().date()),
        },
        "before_kcore": counts_before,
        "after_kcore": {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
        },
        "last_n_months": last_n_months,
        "seed": seed,
    }


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Подготовить данные и записать parquet-файлы со статистикой."""
    seed = config["seed"]
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_cfg = data_cfg["sample"]
    last_n_months = sample_cfg["last_n_months"]
    max_users = sample_cfg.get("max_users")
    kcore = data_cfg["kcore"]

    transactions_path = raw_dir / "transactions_train.csv"
    articles_path = raw_dir / "articles.csv"

    print(f"[prep] reading {transactions_path} ...")
    df = pd.read_csv(
        transactions_path,
        usecols=["customer_id", "article_id", "t_dat"],
        dtype={"article_id": np.int32},
        parse_dates=["t_dat"],
    )
    print(f"[prep] raw rows: {len(df):,}")

    df = filter_last_n_months(df, last_n_months)
    print(f"[prep] after {last_n_months}m window: {len(df):,}")

    df = dedup_interactions(df, data_cfg["dedup_grain"])
    print(f"[prep] after dedup: {len(df):,}")

    counts_before = {
        "n_users": int(df["customer_id"].nunique()),
        "n_items": int(df["article_id"].nunique()),
        "n_interactions": int(len(df)),
    }

    df = subsample_users(df, max_users, seed)
    if max_users is not None:
        print(f"[prep] after user subsample (max={max_users}): {len(df):,}")

    df = iterative_kcore(df, kcore["min_user_interactions"], kcore["min_item_interactions"])
    print(f"[prep] after iterative k-core: {len(df):,}")

    interactions, user_map, item_map = build_dense_ids(df)
    interactions = sort_interactions(interactions, item_map)
    items = build_items_text(articles_path, item_map)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    user_map.to_parquet(out_dir / "user_id_map.parquet", index=False)
    item_map.to_parquet(out_dir / "item_id_map.parquet", index=False)

    stats = compute_stats(interactions, counts_before, last_n_months, seed)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[prep] wrote artifacts to {out_dir}/")
    return stats
