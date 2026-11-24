"""Uniform sampling with replacement from indexed offline experience."""

from collections.abc import Callable
from typing import Generic, TypeVar

import torch

Batch = TypeVar("Batch")


class UniformReplaySampler(Generic[Batch]):
    """Draw CPU indices with a dedicated RNG; get_batch materializes those rows."""

    def __init__(self, size: int, get_batch: Callable[[torch.Tensor], Batch], *, seed: int):
        if type(size) is not int or size < 1:
            raise ValueError("Replay size must be a positive integer")
        self.size = size
        self.get_batch = get_batch
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, batch_size: int) -> Batch:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        indices = torch.randint(self.size, (batch_size,), generator=self._rng, device="cpu")
        return self.get_batch(indices)

    def get_rng_state(self) -> torch.Tensor:
        return self._rng.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self._rng.set_state(state)
