"""Подготовка sequence-файлов для обучения моделей в RecBole."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

# Генератор и скорер должны использовать одинаковую длину истории.
MAX_SEQ_LEN = 50

INTER_FIELDS = ["user_id:token", "item_id_list:token_seq", "item_id:token", "item_length:token"]


def truncate(seq: Sequence, length: int = MAX_SEQ_LEN) -> list:
    """Оставить последние `length` событий."""
    return list(seq[-length:])


def _user_sequences(df: pd.DataFrame) -> "pd.Series":
    """Собрать последовательности товаров по пользователям."""
    return df.groupby("user_id", sort=True)["item_id"].apply(list)


def _user_day_groups(df: pd.DataFrame) -> "pd.Series":
    """Сгруппировать историю пользователя по дням."""
    out = {}
    for (uid, day), items in df.sort_values(["user_id", "t_dat", "item_id"]).groupby(
        ["user_id", "t_dat"], sort=True
    )["item_id"]:
        out.setdefault(uid, []).append(items.tolist())
    return pd.Series(out)


def gen_train_inter(train_df: pd.DataFrame, mode: str = "next_day") -> pd.DataFrame:
    """Собрать обучающие пары из train.

    `next_day` предсказывает товары следующего дня по предыдущим дням.
    `next_item` оставлен для сравнительного эксперимента.
    """
    if mode not in ("next_day", "next_item"):
        raise ValueError(f"mode должен быть next_day|next_item, получено {mode!r}")

    rows = []
    if mode == "next_item":
        for uid, events in _user_sequences(train_df).items():
            if len(events) < 2:
                continue
            for k in range(1, len(events)):
                lst = truncate(events[:k])
                rows.append((uid, " ".join(map(str, lst)), str(events[k]), len(lst)))
        return pd.DataFrame(rows, columns=["user_id", "item_id_list", "item_id", "item_length"])

    for uid, days in _user_day_groups(train_df).items():
        if len(days) < 2:
            continue
        history: list[int] = []
        for j in range(len(days) - 1):
            history += days[j]
            prefix = truncate(history)
            prefix_str = " ".join(map(str, prefix))
            plen = len(prefix)
            for target in days[j + 1]:
                rows.append((uid, prefix_str, str(target), plen))
    return pd.DataFrame(rows, columns=["user_id", "item_id_list", "item_id", "item_length"])


def gen_valid_inter(train_df: pd.DataFrame, holdouts_df: pd.DataFrame) -> pd.DataFrame:
    """Собрать validation-пары для RecBole."""
    seqs = _user_sequences(train_df)
    elig = holdouts_df[holdouts_df["is_val_eval"]]
    rows = []
    for uid, val_item in zip(elig["user_id"].to_numpy(), elig["val_item"].to_numpy()):
        events = seqs.get(uid, [])
        lst = truncate(events)
        rows.append((uid, " ".join(map(str, lst)), str(val_item), len(lst)))
    return pd.DataFrame(rows, columns=["user_id", "item_id_list", "item_id", "item_length"])


def gen_test_inter(train_df: pd.DataFrame, holdouts_df: pd.DataFrame) -> pd.DataFrame:
    """Вернуть копию validation-файла, которую ожидает RecBole."""
    return gen_valid_inter(train_df, holdouts_df)


def write_inter(rows: pd.DataFrame, path: Path) -> str:
    """Записать .inter-файл и вернуть его sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(INTER_FIELDS) + "\n")
        for r in rows.itertuples(index=False):
            f.write(f"{r.user_id}\t{r.item_id_list}\t{r.item_id}\t{r.item_length}\n")
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_leak(
    train_inter: pd.DataFrame, train_df: pd.DataFrame, holdouts_df: pd.DataFrame,
    mode: str = "next_day",
) -> dict[str, int]:
    """Проверить число обучающих пар и отсутствие holdout-таргетов в train."""
    if mode == "next_item":
        lens = train_df.groupby("user_id").size()
        expected_rows = int((lens[lens >= 2] - 1).sum())
    else:
        per_day = train_df.groupby(["user_id", "t_dat"]).size()
        n_days = per_day.groupby(level=0).size()
        total_ev = per_day.groupby(level=0).sum()
        first_day_ev = per_day.groupby(level=0).first()
        ge2 = n_days >= 2
        expected_rows = int((total_ev[ge2] - first_day_ev[ge2]).sum())
    assert len(train_inter) == expected_rows, (
        f"train.inter содержит {len(train_inter)} строк вместо {expected_rows} (mode={mode})"
    )

    train_pairs = train_df[["user_id", "item_id"]].drop_duplicates()

    def _in_user_train(elig: pd.DataFrame, item_col: str) -> int:
        probe = elig[["user_id", item_col]].rename(columns={item_col: "item_id"})
        hit = probe.merge(train_pairs, on=["user_id", "item_id"], how="left", indicator=True)
        return int((hit["_merge"] == "both").sum())

    val_elig = holdouts_df[holdouts_df["is_val_eval"]]
    n_val_leak = _in_user_train(val_elig, "val_item")
    assert n_val_leak == 0, f"val_item встретился в train у {n_val_leak} пользователей"

    test_elig = holdouts_df[holdouts_df["is_test_eval"]]
    n_test_leak = _in_user_train(test_elig, "test_item")
    assert n_test_leak == 0, f"test_item встретился в train у {n_test_leak} пользователей"
    n_test_eq_val = int((test_elig["test_item"] == test_elig["val_item"]).sum())
    assert n_test_eq_val == 0, f"test_item == val_item у {n_test_eq_val} пользователей"

    return {
        "n_train_inter_rows": len(train_inter),
        "expected_rows": expected_rows,
        "mode": mode,
        "n_val_eval": int(len(val_elig)),
        "n_test_eval": int(len(test_elig)),
    }
