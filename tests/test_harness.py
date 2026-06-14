"""Интеграционные тесты evaluation pipeline на маленьком сплите."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.harness import evaluate

CATALOG = [10, 20, 30, 40, 50]
SCORE_VEC = np.array([0.5, 0.4, 0.3, 0.2, 0.1])


class MockScorer:
    """Скорер с фиксированным порядком товаров."""

    def score(self, user_id, prefix_items):  # noqa: ARG002
        return SCORE_VEC.copy()


def _write_split(tmp_path, train_rows, holdout_rows):
    d = tmp_path / "split"
    d.mkdir()
    pd.DataFrame(train_rows, columns=["user_id", "item_id", "t_dat"]).to_parquet(
        d / "train.parquet", index=False
    )
    pd.DataFrame(
        holdout_rows,
        columns=["user_id", "val_item", "val_date", "test_item", "test_date",
                 "is_val_eval", "is_test_eval"],
    ).to_parquet(d / "holdouts.parquet", index=False)
    pd.DataFrame({"item_id": CATALOG}).to_parquet(d / "catalog.parquet", index=False)
    return {"paths": {"split_dir": str(d)}, "seed": 42}


def test_evaluate_val_exact(tmp_path):
    t = pd.Timestamp("2020-01-01")
    train = [
        (1, 10, t), (1, 20, t),
        (2, 30, t), (2, 999, t),
    ]
    hold = [
        (1, 40, t, 50, t, True, True),
        (2, 10, t, 20, t, True, True),
    ]
    cfg = _write_split(tmp_path, train, hold)
    res = evaluate(MockScorer(), "val", cfg, log_mlflow=False, model_name="mock")

    assert res["n_cases"] == 2
    assert res["recall@10"] == pytest.approx(1.0)
    assert res["mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert res["ndcg@10"] == pytest.approx((1 / np.log2(3) + 1) / 2)


def test_evaluate_test_prefix_and_oov(tmp_path):
    """OOV validation-событие не попадает в test-prefix."""
    t = pd.Timestamp("2020-01-01")
    train = [(3, 20, t)]
    hold = [(3, 777, t, 50, t, False, True)]
    cfg = _write_split(tmp_path, train, hold)
    res = evaluate(MockScorer(), "test", cfg, log_mlflow=False, model_name="mock")
    assert res["n_cases"] == 1
    assert res["mrr"] == pytest.approx(1 / 4)


def test_evaluate_raises_when_target_in_mask(tmp_path):
    """Таргет в истории считается ошибкой сплита."""
    t = pd.Timestamp("2020-01-01")
    train = [(5, 30, t)]
    hold = [(5, 30, t, 40, t, True, False)]
    cfg = _write_split(tmp_path, train, hold)
    with pytest.raises(AssertionError):
        evaluate(MockScorer(), "val", cfg, log_mlflow=False, model_name="mock")
