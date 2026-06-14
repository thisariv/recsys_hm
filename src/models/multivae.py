"""NumPy-инференс для обученной в RecBole модели Mult-VAE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MISSING_SCORE = -1e9


def _extract_mlp(state, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Извлечь параметры линейных слоёв из state_dict."""
    layers, i = [], 0
    while f"{prefix}.{i}.weight" in state:
        W = np.asarray(state[f"{prefix}.{i}.weight"], dtype=np.float64)
        b = np.asarray(state[f"{prefix}.{i}.bias"], dtype=np.float64)
        layers.append((W, b))
        i += 2
    return layers


class MultiVAEScorer:
    def __init__(self, enc, dec, token2id: dict[str, int], catalog: np.ndarray) -> None:
        self.enc = enc
        self.dec = dec
        self.token2id = token2id
        self.n_recbole = enc[0][0].shape[1]
        catalog = np.asarray(catalog)
        self.n_items = len(catalog)
        perm = np.zeros(self.n_items, dtype=np.int64)
        missing = np.zeros(self.n_items, dtype=bool)
        for i, cid in enumerate(catalog.tolist()):
            tid = token2id.get(str(int(cid)))
            if tid is None:
                missing[i] = True
            else:
                perm[i] = tid
        self.perm, self.missing = perm, missing
        self.n_missing = int(missing.sum())

    @staticmethod
    def _mlp(h: np.ndarray, layers) -> np.ndarray:
        for j, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if j < len(layers) - 1:
                h = np.tanh(h)
        return h

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x / (np.linalg.norm(x) + 1e-12)
        h = self._mlp(h, self.enc)
        mu = h[: h.shape[0] // 2]
        return self._mlp(mu, self.dec)

    def score(self, user_id: int, prefix_items) -> np.ndarray:  # noqa: ARG002
        idx = [self.token2id[str(int(c))] for c in prefix_items if str(int(c)) in self.token2id]
        if not idx:
            return np.zeros(self.n_items, dtype=np.float64)
        x = np.zeros(self.n_recbole, dtype=np.float64)
        x[idx] = 1.0
        raw = self._forward(x)
        out = raw[self.perm]
        if self.n_missing:
            out[self.missing] = MISSING_SCORE
        return out

    def score_batch(self, batch):
        return [self.score(u, p) for u, p in batch]


def load_real_multivae(ckpt_dir, catalog, expected_hash: str | None = None) -> "MultiVAEScorer":
    import torch

    ckpt_dir = Path(ckpt_dir)
    with (ckpt_dir / "token_map.json").open(encoding="utf-8") as f:
        token2id = {str(k): int(v) for k, v in json.load(f).items()}
    meta_path = ckpt_dir / "meta.json"
    if expected_hash is not None and meta_path.exists():
        meta = json.load(meta_path.open(encoding="utf-8"))
        assert meta.get("catalog_hash") == expected_hash, "catalog_hash MultiVAE != V_train"

    pth = sorted(ckpt_dir.glob("*.pth"))
    assert pth, f"в {ckpt_dir} нет .pth"
    ckpt = torch.load(pth[0], map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    enc = _extract_mlp(state, "encoder")
    dec = _extract_mlp(state, "decoder")
    assert enc and dec, "не нашёл encoder/decoder в state_dict"
    return MultiVAEScorer(enc, dec, token2id, catalog)
