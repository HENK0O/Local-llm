from __future__ import annotations

from collections import OrderedDict
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


class PrefixCache:
    """Bounded LRU of exact conversation prefixes; serialized callers own leases.

    Taking an entry removes it until a completed generation stores it again.
    Interrupted work therefore cannot expose partially overwritten KV data.
    Other conversations remain intact. No text or cache is written to disk.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024, max_entries: int = 8) -> None:
        if max_bytes < 0 or max_entries < 1:
            raise ValueError("Invalid prefix cache budget")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.clear()

    def clear(self) -> None:
        self.entries = OrderedDict()
        self.model = self.cache = None
        self.tokens: List[int] = []

    @property
    def nbytes(self) -> int:
        return sum(entry[2].nbytes for entry in self.entries.values())

    def prepare(self, model, prompt: List[int], capacity: int, key: str = "default") -> Tuple[KVCache, int]:
        entry = self.entries.pop(key, None)
        previous = entry[2] if entry is not None and entry[0] is model else None
        reused = 0
        if previous is not None:
            for old, new in zip(entry[1], prompt[:-1]):
                if old != new:
                    break
                reused += 1
        self.model = self.cache = None
        self.tokens = []
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

    def store(self, model, tokens: List[int], cache: KVCache, key: str = "default") -> None:
        self.entries.pop(key, None)
        if cache.nbytes <= self.max_bytes and cache.length == len(tokens):
            self.entries[key] = (model, list(tokens), cache)
        while len(self.entries) > self.max_entries or self.nbytes > self.max_bytes:
            self.entries.popitem(last=False)
        last = next(reversed(self.entries.values()), None) if self.entries else None
        self.model, self.tokens, self.cache = last if last else (None, [], None)
