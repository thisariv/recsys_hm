"""Тесты воспроизводимости генераторов случайных чисел."""
import numpy as np
import torch

from src.utils.seed import set_seed


def test_numpy_determinism():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_torch_determinism():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_combined_determinism():
    """NumPy и PyTorch повторяют последовательности после set_seed."""
    set_seed(42)
    seq_a = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_seed(42)
    seq_b = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert seq_a == seq_b
