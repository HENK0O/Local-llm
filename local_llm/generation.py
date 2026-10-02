from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .model import LlamaModel
from .cache import PrefixCache


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    cache_bytes: int
    reused_prompt_tokens: int = 0

    @property
    def prefill_tokens_per_second(self) -> float:
        processed = self.prompt_tokens - self.reused_prompt_tokens
        return processed / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        decoded = max(0, self.generated_tokens - 1)
        return decoded / self.decode_seconds if self.decode_seconds else 0.0


@dataclass(frozen=True)
class GenerationResult:
    token_ids: List[int]
    stats: GenerationStats


def sample_token(
    logits: NDArray[np.floating],
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    logits = np.asarray(logits)
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return int(np.argmax(logits))
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    scaled = logits.astype(np.float64) / temperature
    keep = np.ones(logits.size, dtype=bool)
    if top_k is not None and top_k < logits.size:
        indices = np.argpartition(scaled, -top_k)[-top_k:]
        keep[:] = False
        keep[indices] = True

    filtered = np.where(keep, scaled, -np.inf)
    shifted = filtered - np.max(filtered)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    if top_p is not None and top_p < 1:
        order = np.argsort(probabilities)[::-1]
        ordered = probabilities[order]
        remove = np.cumsum(ordered) - ordered >= top_p
        probabilities[order[remove]] = 0.0
        probabilities /= probabilities.sum()

    return int((rng or np.random.default_rng()).choice(logits.size, p=probabilities))


def generate_tokens(
    model: LlamaModel,
    prompt_tokens: List[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    *,
    prefix_cache: Optional[PrefixCache] = None,
    cache_key: str = "default",
) -> Iterator[Tuple[int, Optional[GenerationStats]]]:
    """Yield ``(token_id, stats)``; stats is populated only on the final item."""
    if not prompt_tokens:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    maximum = model.config.max_position_embeddings - len(prompt_tokens) + 1
    if max_new_tokens > maximum:
        raise ValueError(
            f"requested {max_new_tokens} new tokens, but only {max(0, maximum)} fit in the context"
        )
    capacity = min(model.config.max_position_embeddings, len(prompt_tokens) + max_new_tokens)
    if len(prompt_tokens) > capacity:
        raise ValueError("prompt exceeds the model context length")

    if max_new_tokens == 0:
        return

    prefill_start = time.perf_counter()
    cache, reused = (prefix_cache.prepare(model, prompt_tokens, max(capacity, 1), cache_key)
                     if prefix_cache is not None else (model.new_cache(max(capacity, 1)), 0))
    logits = model.forward(np.asarray(prompt_tokens[reused:], dtype=np.int64), cache=cache,
                           last_logits_only=True)
    prefill_seconds = time.perf_counter() - prefill_start
    rng = np.random.default_rng(seed)
    emitted: List[int] = []
    decode_seconds = 0.0

    for _ in range(max_new_tokens):
        token = sample_token(logits[-1], temperature, top_k, top_p, rng)
        emitted.append(token)
        is_eos = model.config.eos_token_id is not None and token == model.config.eos_token_id
        if is_eos or len(emitted) == max_new_tokens:
            stats = GenerationStats(
                prompt_tokens=len(prompt_tokens),
                generated_tokens=len(emitted),
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                cache_bytes=cache.nbytes,
                reused_prompt_tokens=reused,
            )
            if prefix_cache is not None:
                # The final emitted token (EOS or length limit) has not yet
                # passed through forward; only retain positions with KV data.
                prefix_cache.store(model, prompt_tokens + emitted[:-1], cache, cache_key)
            yield token, stats
            return
        yield token, None
        decode_start = time.perf_counter()
        logits = model.forward(np.asarray([token], dtype=np.int64), cache=cache)
        decode_seconds += time.perf_counter() - decode_start


def generate(
    model: LlamaModel,
    prompt_tokens: List[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> GenerationResult:
    tokens: List[int] = []
    final_stats: Optional[GenerationStats] = None
    for token, stats in generate_tokens(
        model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
    ):
        tokens.append(token)
        if stats is not None:
            final_stats = stats
    if final_stats is None:
        final_stats = GenerationStats(len(prompt_tokens), 0, 0.0, 0.0, 0)
    return GenerationResult(tokens, final_stats)


def greedy_generate_without_cache(
    model: LlamaModel, prompt_tokens: List[int], max_new_tokens: int
) -> List[int]:
    """Slow reference path: recompute the entire sequence for every token."""
    sequence = list(prompt_tokens)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(np.asarray(sequence, dtype=np.int64))
        token = int(np.argmax(logits[-1]))
        generated.append(token)
        if model.config.eos_token_id is not None and token == model.config.eos_token_id:
            break
        sequence.append(token)
    return generated
