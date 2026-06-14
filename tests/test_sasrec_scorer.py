"""Тесты адаптера sequence-моделей без зависимости от RecBole."""
from __future__ import annotations

import numpy as np

from src.sasrec.atomic import MAX_SEQ_LEN, truncate
from src.sasrec.scorer import SASRecScorer, build_token2id
from src.sasrec.stub import StubSASRec

CATALOG = np.arange(100, 130)


def _scorer():
    stub = StubSASRec(n_items=len(CATALOG), d=16, seed=2026)
    token2id = build_token2id(CATALOG)
    return SASRecScorer(stub, token2id, CATALOG, max_seq_len=MAX_SEQ_LEN)


def test_output_shape_full_catalog():
    s = _scorer()
    out = s.score(0, [100, 101, 102])
    assert out.shape == (len(CATALOG),)


def test_different_prefixes_different_scores():
    """Разные истории дают разные скоры."""
    s = _scorer()
    a = s.score(0, [100, 101])
    b = s.score(0, [120, 121, 122])
    assert not np.allclose(a, b)


def test_pad_does_not_change_short_prefix():
    """PAD-токены не меняют результат короткой истории."""
    s = _scorer()
    one = s.score(0, [105])
    one_again = s.score(0, [105])
    assert np.array_equal(one, one_again)
    two = s.score(0, [105, 106])
    assert not np.allclose(one, two)


def test_history_60_uses_last_50():
    """Длинная история обрезается до последних MAX_SEQ_LEN событий."""
    s = _scorer()
    long_prefix = (list(CATALOG) + list(CATALOG))[:60]
    full = s.score(0, long_prefix)
    last50 = s.score(0, truncate(long_prefix))
    assert np.array_equal(full, last50)


def test_perm_is_injective():
    """Каждый товар каталога отображается в отдельный token ID."""
    s = _scorer()
    assert len(set(s.perm.tolist())) == len(s.perm)
    assert s.n_missing == 0


def test_empty_prefix_abstains():
    """Полностью OOV-история даёт нулевой вектор."""
    s = _scorer()
    out = s.score(0, [99999])
    assert out.shape == (len(CATALOG),)
    assert np.all(out == 0.0)
    assert s.n_empty_prefix == 1
