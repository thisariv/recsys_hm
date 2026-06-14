"""Адаптер sequence-моделей RecBole к общему интерфейсу скоринга."""
from __future__ import annotations

import numpy as np

from src.sasrec.atomic import MAX_SEQ_LEN, truncate

MISSING_SCORE = -1e9


class SASRecScorer:
    def __init__(self, model, token2id: dict[str, int], catalog: np.ndarray,
                 max_seq_len: int = MAX_SEQ_LEN, batch_size: int = 1024) -> None:
        self.model = model
        self.token2id = token2id
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        catalog = np.asarray(catalog)
        self.n_items = len(catalog)
        # Перестановка из локального каталога в словарь RecBole.
        perm = np.zeros(self.n_items, dtype=np.int64)
        missing = np.zeros(self.n_items, dtype=bool)
        for i, cid in enumerate(catalog.tolist()):
            tid = token2id.get(str(cid))
            if tid is None:
                missing[i] = True
            else:
                perm[i] = tid
        self.perm = perm
        self.missing = missing
        self.n_missing = int(missing.sum())
        self.n_empty_prefix = 0

    def _encode(self, prefix_items):
        """Закодировать историю и дополнить её PAD-токенами справа."""
        seq = [self.token2id[str(c)] for c in truncate(prefix_items, self.max_seq_len)
               if str(c) in self.token2id]
        if not seq:
            return None
        padded = seq + [0] * (self.max_seq_len - len(seq))
        return padded, len(seq)

    def _permute(self, raw: np.ndarray) -> np.ndarray:
        """Привести скоры RecBole к порядку локального каталога."""
        out = raw[:, self.perm]
        if self.n_missing:
            out[:, self.missing] = MISSING_SCORE
        return out

    def score_batch(self, batch):
        """Посчитать скоры батчами; пустая история получает нулевой вектор."""
        results = []
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start:start + self.batch_size]
            enc = [self._encode(p) for _, p in chunk]
            ok_idx = [i for i, e in enumerate(enc) if e is not None]
            out = np.zeros((len(chunk), self.n_items), dtype=np.float64)
            if ok_idx:
                seqs = np.array([enc[i][0] for i in ok_idx])
                lengths = np.array([enc[i][1] for i in ok_idx])
                raw = np.asarray(self.model.full_sort_predict(seqs, lengths))
                out[ok_idx] = self._permute(raw)
            self.n_empty_prefix += len(chunk) - len(ok_idx)
            results.extend(out[i] for i in range(len(chunk)))
        return results

    def score(self, user_id: int, prefix_items) -> np.ndarray:
        return self.score_batch([(user_id, prefix_items)])[0]


def build_token2id(catalog: np.ndarray) -> dict[str, int]:
    """Построить простой словарь токенов для тестовой модели."""
    return {str(int(c)): i + 1 for i, c in enumerate(np.asarray(catalog).tolist())}


def verify_bridge(scorer, catalog, train_df, current_hash, atomic_meta=None) -> dict:
    """Проверить совместимость словаря RecBole с локальным каталогом."""
    catalog = np.asarray(catalog)
    n_v = len(catalog)
    info = {"n_v_train": n_v, "vocab_size": len(scorer.token2id), "n_missing": scorer.n_missing}

    if atomic_meta and "catalog_hash" in atomic_meta:
        assert atomic_meta["catalog_hash"] == current_hash, (
            "catalog_hash сплита не совпадает с atomic_meta"
        )
        info["hash_check"] = "ok"
    else:
        info["hash_check"] = "skipped"

    map_tokens = {k for k in scorer.token2id if k != "[PAD]"}
    v_train_tokens = {str(int(c)) for c in catalog.tolist()}
    foreign = map_tokens - v_train_tokens
    assert not foreign, (
        f"в token_map найдено {len(foreign)} токенов вне V_train"
    )
    assert scorer.n_missing == n_v - len(map_tokens), (
        f"n_missing={scorer.n_missing}, ожидалось {n_v - len(map_tokens)}"
    )
    frac_missing = scorer.n_missing / n_v
    assert frac_missing < 0.01, (
        f"слишком много missing-айтемов: {scorer.n_missing} ({frac_missing:.3%}) "
        f"-> словарь не соответствует каталогу"
    )
    info["frac_missing"] = round(frac_missing, 5)
    info["missing_item_ids"] = catalog[scorer.missing].tolist()

    perm_present = scorer.perm[~scorer.missing]
    assert len(set(perm_present.tolist())) == len(perm_present), "perm не инъективен"
    assert len(scorer.token2id) == n_v + 1 - scorer.n_missing, (
        f"|vocab| {len(scorer.token2id)} != |V_train|+1−n_missing "
        f"{n_v + 1 - scorer.n_missing}"
    )
    id2token = {v: k for k, v in scorer.token2id.items()}
    rng = np.random.default_rng(0)
    sample = catalog[rng.choice(n_v, size=min(1000, n_v), replace=False)]
    for cid in sample.tolist():
        tid = scorer.token2id.get(str(int(cid)))
        if tid is not None:
            assert id2token[tid] == str(int(cid)), f"round-trip сломан на {cid}"

    probe = next((int(c) for c in catalog.tolist() if str(int(c)) in scorer.token2id), None)
    assert probe is not None
    out = scorer.score(0, [probe])
    assert out.shape == (n_v,), f"форма {out.shape} != ({n_v},)"
    info["score_shape"] = list(out.shape)
    return info


class RecBoleModelWrapper:
    """Обёртка RecBole-модели с интерфейсом для NumPy-батчей."""

    def __init__(self, model, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def full_sort_predict(self, seqs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        from recbole.data.interaction import Interaction

        torch = self._torch
        inter = Interaction({
            self.model.ITEM_SEQ: torch.as_tensor(np.asarray(seqs), dtype=torch.long),
            self.model.ITEM_SEQ_LEN: torch.as_tensor(np.asarray(lengths), dtype=torch.long),
        }).to(self.device)
        with torch.no_grad():
            scores = self.model.full_sort_predict(inter)
        return scores.view(len(lengths), -1).cpu().numpy()


def _load_token_map(ckpt_dir) -> dict[str, int]:
    import json
    from pathlib import Path

    with (Path(ckpt_dir) / "token_map.json").open(encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_real_sasrec(ckpt_dir, catalog, config, expected_hash: str | None = None) -> SASRecScorer:
    """Загрузить sequence-модель RecBole из сохранённого чекпойнта."""
    import importlib
    from pathlib import Path

    import torch

    ckpt_dir = Path(ckpt_dir)
    pths = sorted(ckpt_dir.glob("*.pth"))
    assert pths, f"в {ckpt_dir} нет .pth чекпойнта"
    token2id = _load_token_map(ckpt_dir)
    n_items = len(token2id)

    ckpt = torch.load(pths[0], map_location="cpu", weights_only=False)
    rb_config = ckpt["config"]
    rb_config["device"] = torch.device("cpu")

    model_name = rb_config["model"]
    mod = importlib.import_module(f"recbole.model.sequential_recommender.{model_name.lower()}")
    ModelClass = getattr(mod, model_name)

    class _StubDataset:
        def num(self, field):  # noqa: ARG002
            return n_items

    model = ModelClass(rb_config, _StubDataset())
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    wrapper = RecBoleModelWrapper(model, device="cpu")
    max_seq_len = config["models"]["sasrec"]["max_seq_len"]
    return SASRecScorer(wrapper, token2id, catalog, max_seq_len=max_seq_len)
