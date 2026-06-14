"""Тесты подготовки sequence-файлов для RecBole."""
from __future__ import annotations

import pandas as pd

from src.sasrec.atomic import (
    MAX_SEQ_LEN,
    assert_no_leak,
    file_sha256,
    gen_train_inter,
    gen_valid_inter,
    truncate,
    write_inter,
)

T = pd.Timestamp("2020-01-01")


def _train(rows):
    return pd.DataFrame(
        [(u, i, T + pd.Timedelta(days=d)) for u, i, d in rows],
        columns=["user_id", "item_id", "t_dat"],
    )


def test_next_day_transitions_exact():
    """next_day создаёт переходы только между днями."""
    train = _train([(1, 10, 0), (1, 20, 0), (1, 30, 1), (1, 40, 2), (1, 50, 2)])
    inter = gen_train_inter(train, mode="next_day").sort_values(
        ["item_length", "item_id"]).reset_index(drop=True)

    assert inter["item_id"].tolist() == ["30", "40", "50"]
    assert inter["item_id_list"].tolist() == ["10 20", "10 20 30", "10 20 30"]
    assert inter["item_length"].tolist() == [2, 3, 3]
    assert not ((inter["item_id"] == "20") & (inter["item_id_list"] == "10")).any()


def test_next_day_single_day_user_no_rows():
    """Один день не образует обучающих переходов."""
    train = _train([(1, 10, 0), (1, 20, 0), (1, 30, 0)])
    assert len(gen_train_inter(train, mode="next_day")) == 0


def test_next_item_mode_still_available():
    """Режим next_item остаётся доступен для сравнения."""
    train = _train([(1, 10, 0), (1, 20, 1), (1, 30, 2), (2, 99, 0)])
    inter = gen_train_inter(train, mode="next_item")
    assert len(inter) == 2


def test_truncate_last_50_shared():
    seq = list(range(60))
    assert truncate(seq) == list(range(10, 60))
    assert len(truncate(seq)) == MAX_SEQ_LEN
    train = _train([(1, i, i) for i in range(60)])
    hold = pd.DataFrame([(1, 999, T, 888, T, True, True)],
                        columns=["user_id", "val_item", "val_date",
                                 "test_item", "test_date", "is_val_eval", "is_test_eval"])
    toks = gen_valid_inter(train, hold).iloc[0]["item_id_list"].split()
    assert len(toks) == 50 and toks == [str(i) for i in range(10, 60)]


def test_anti_leak_pass_next_day():
    train = _train([(1, 10, 0), (1, 20, 1), (1, 30, 2)])
    hold = pd.DataFrame([(1, 50, T, 60, T, True, True)],
                        columns=["user_id", "val_item", "val_date",
                                 "test_item", "test_date", "is_val_eval", "is_test_eval"])
    inter = gen_train_inter(train, mode="next_day")
    info = assert_no_leak(inter, train, hold, mode="next_day")
    assert info["n_train_inter_rows"] == 2 == info["expected_rows"]


def test_anti_leak_detects_target_in_train():
    train = _train([(1, 10, 0), (1, 20, 1), (1, 30, 2)])
    hold = pd.DataFrame([(1, 20, T, 99, T, True, False)],
                        columns=["user_id", "val_item", "val_date",
                                 "test_item", "test_date", "is_val_eval", "is_test_eval"])
    inter = gen_train_inter(train, mode="next_day")
    try:
        assert_no_leak(inter, train, hold, mode="next_day")
        raise AssertionError("должно было упасть на лике val_item∈train")
    except AssertionError as e:
        assert "val_item встретился в train" in str(e)


def test_sha_deterministic(tmp_path):
    train = _train([(1, 10, 0), (1, 20, 1), (1, 30, 2), (3, 5, 0), (3, 6, 1)])
    inter = gen_train_inter(train, mode="next_day")
    h1 = write_inter(inter, tmp_path / "a.inter")
    h2 = write_inter(inter, tmp_path / "b.inter")
    assert h1 == h2 == file_sha256(tmp_path / "a.inter")
