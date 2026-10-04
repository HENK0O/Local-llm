"""Bounded, RAM-only MLX conversation state; no import of MLX in the app venv.

An entry is leased out while generation mutates it. Reuse always compares token
IDs, never rendered strings. Recurrent state is reused only as an entire exact
prefix: it cannot be rolled back like attention KV.
"""
import copy
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    tokens: tuple
    layers: list
    nbytes: int


class ConversationCache:
    def __init__(self, factory, can_trim, trim, max_bytes, max_entries=4):
        self.factory, self.can_trim, self.trim = factory, can_trim, trim
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max_entries
        self.entries = OrderedDict()

    @property
    def nbytes(self):
        return sum(entry.nbytes for entry in self.entries.values())

    @staticmethod
    def size_bytes(layers):
        try:
            sizes = [layer.nbytes for layer in layers]
            return sum(sizes) if sizes and all(type(n) is int and n >= 0 for n in sizes) else None
        except (AttributeError, NotImplementedError, TypeError):
            return None

    @staticmethod
    def position(layers):
        """Only a unanimous explicit position can certify generated KV state."""
        try:
            positions = [layer.size() for layer in layers]
            if positions and all(type(n) is int and n >= 0 and n == positions[0] for n in positions):
                return positions[0]
        except (AttributeError, NotImplementedError, TypeError):
            pass
        return None

    def _evict(self, incoming=0):
        while self.entries and (self.nbytes + incoming > self.max_bytes or len(self.entries) >= self.max_entries):
            self.entries.popitem(last=False)

    def prepare(self, key, prompt):
        entry = self.entries.pop(key, None)
        reused = 0
        if entry is not None:
            for old, new in zip(entry.tokens, prompt[:-1]):
                if old != new:
                    break
                reused += 1
            if reused:
                if self.can_trim(entry.layers):
                    amount = len(entry.tokens) - reused
                    if self.position(entry.layers) == len(entry.tokens):
                        trimmed = self.trim(entry.layers, amount) if amount else 0
                        if trimmed == amount and self.position(entry.layers) == reused:
                            return entry.layers, reused
                elif reused == len(entry.tokens):
                    return entry.layers, reused
        return self.factory(), 0

    def checkpoint(self, layers, tokens):
        """Save pre-decode recurrent state, before a generated token mutates it."""
        size = self.size_bytes(layers)
        if not tokens or size is None or size > self.max_bytes or not self.max_bytes:
            return None
        self._evict(size)
        snapshot = copy.deepcopy(layers)
        actual = self.size_bytes(snapshot)
        if actual is None or actual > self.max_bytes:
            return None
        return CacheEntry(tuple(tokens), snapshot, actual)

    def store(self, key, tokens, layers):
        size = self.size_bytes(layers)
        self.entries.pop(key, None)
        if not tokens or size is None or size > self.max_bytes or not self.max_bytes:
            return False
        self._evict(size)
        self.entries[key] = CacheEntry(tuple(tokens), layers, size)
        return True

    def summary(self):
        return {'supported': True, 'entries': len(self.entries), 'bytes': self.nbytes,
                'budget_bytes': self.max_bytes, 'max_entries': self.max_entries, 'storage': 'memory'}

    def resize(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        while self.entries and (self.nbytes > self.max_bytes or len(self.entries) > self.max_entries):
            self.entries.popitem(last=False)
