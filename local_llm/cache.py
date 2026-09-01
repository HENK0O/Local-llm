from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig


@dataclass
class LayerKV:
    keys: NDArray[np.floating]
    values: NDArray[np.floating]


class KVCache:
    """Preallocated per-layer KV storage in [sequence, kv_heads, head_dim] order."""

    def __init__(self, config: ModelConfig, capacity: int, dtype: np.dtype = np.float32) -> None:
        if capacity <= 0 or capacity > config.max_position_embeddings:
            raise ValueError("cache capacity must be within max_position_embeddings")
        shape = (capacity, config.num_key_value_heads, config.head_dim)
        self.layers: List[LayerKV] = [
            LayerKV(np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype))
            for _ in range(config.num_hidden_layers)
        ]
        self.capacity = capacity
        self.length = 0

    @property
    def nbytes(self) -> int:
        return sum(layer.keys.nbytes + layer.values.nbytes for layer in self.layers)

    def reset(self) -> None:
        self.length = 0

