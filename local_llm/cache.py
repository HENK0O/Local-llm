from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class PrefixCache:
    """Retain one bounded prefix between serialized requests, without saving text.

    Token IDs must match exactly. The last prompt token is always evaluated to
    produce fresh logits, including when a whole prompt matches an earlier one.
    The caller owns the returned cache until ``store``; interrupted work cannot
    leave a partially overwritten prefix available to the next request.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self.clear()

    def clear(self) -> None:
        self.model = None
        self.tokens: List[int] = []
        self.cache: Optional[KVCache] = None

    @property
    def nbytes(self) -> int:
        return self.cache.nbytes if self.cache is not None else 0

    def prepare(self, model, prompt: List[int], capacity: int) -> Tuple[KVCache, int]:
        previous = self.cache if self.model is model else None
        reused = 0
        if previous is not None:
            for old, new in zip(self.tokens, prompt[:-1]):
                if old != new:
                    break
                reused += 1
        self.clear()
        if previous is not None and previous.capacity >= capacity:
            cache = previous
        else:
            cache = model.new_cache(capacity)
            if reused:
                for source, target in zip(previous.layers, cache.layers):
                    target.keys[:reused] = source.keys[:reused]
                    target.values[:reused] = source.values[:reused]
        cache.length = reused
        return cache, reused

    def store(self, model, tokens: List[int], cache: KVCache) -> None:
        if cache.nbytes <= self.max_bytes and cache.length == len(tokens):
            self.model = model
            self.tokens = list(tokens)
            self.cache = cache
        else:
            self.clear()
