from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def linear(x: Array, weight: Array) -> Array:
    """Apply a bias-free linear layer; weights use PyTorch's [out, in] layout."""
    if hasattr(weight, "matmul"):
        return weight.matmul(x)
    # Apple's bundled BLAS can leave spurious floating-point status flags after
    # valid SGEMM calls (notably for small, oddly sized vocabularies). Non-finite
    # values still propagate normally and can be asserted by callers/tests.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.matmul(x, weight.T)


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    variance = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)
    normalized = x * (1.0 / np.sqrt(variance + eps))
    return normalized * weight


def silu(x: Array) -> Array:
    x32 = x.astype(np.float32)
    return x32 / (1.0 + np.exp(-x32))


def softmax(x: Array, axis: int = -1) -> Array:
    x32 = x.astype(np.float32)
    shifted = x32 - np.max(x32, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)


def apply_rope(x: Array, positions: NDArray[np.integer], theta: float,
               interleaved: bool = False) -> Array:
    """Apply split-half (HF) or adjacent-pair (GGUF) rotary embeddings."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    frequencies = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies[None, :]
    if interleaved:
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        result = np.empty_like(x)
        result[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        result[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return result
    embedding = np.concatenate((angles, angles), axis=-1)[:, None, :]
    cos, sin = np.cos(embedding), np.sin(embedding)
    half = head_dim // 2
    rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + rotated * sin
