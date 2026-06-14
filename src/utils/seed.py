"""Настройка генераторов случайных чисел."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Настроить воспроизводимость для Python, NumPy и PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Не загружаем PyTorch в командах, где он не нужен.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Не все операции и бэкенды поддерживают строгий детерминизм.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
