from unittest.mock import patch

import numpy as np
import torch

from colors_of_meaning.shared.determinism import seed_everything


def test_should_return_torch_generator_when_seeded() -> None:
    generator = seed_everything(42)

    assert isinstance(generator, torch.Generator)


def test_should_reproduce_global_torch_values_when_seed_is_repeated() -> None:
    seed_everything(123)
    first = torch.rand(5)
    seed_everything(123)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_should_reproduce_numpy_values_when_seed_is_repeated() -> None:
    seed_everything(7)
    first = np.random.rand(5)
    seed_everything(7)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_should_seed_returned_generator_deterministically_when_seed_is_repeated() -> None:
    first = torch.rand(5, generator=seed_everything(99))
    second = torch.rand(5, generator=seed_everything(99))

    assert torch.equal(first, second)


def test_should_enable_deterministic_algorithms_when_flag_is_true() -> None:
    with patch("torch.use_deterministic_algorithms") as mock_use_deterministic:
        seed_everything(42, deterministic=True)

    mock_use_deterministic.assert_called_once_with(True)


def test_should_not_enable_deterministic_algorithms_when_flag_is_false() -> None:
    with patch("torch.use_deterministic_algorithms") as mock_use_deterministic:
        seed_everything(42, deterministic=False)

    mock_use_deterministic.assert_not_called()
