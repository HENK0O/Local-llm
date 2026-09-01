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


def sigmoid(x: Array) -> Array:
    x32 = x.astype(np.float32)
    positive = x32 >= 0
    result = np.empty_like(x32)
    result[positive] = 1.0 / (1.0 + np.exp(-x32[positive]))
    exponential = np.exp(x32[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def softmax(x: Array, axis: int = -1) -> Array:
    x32 = x.astype(np.float32)
    shifted = x32 - np.max(x32, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)


def apply_rope(x: Array, positions: NDArray[np.integer], theta: float,
               interleaved: bool = False, dimension_count: int | None = None) -> Array:
    """Apply split-half (HF) or adjacent-pair (GGUF) rotary embeddings."""
    head_dim = x.shape[-1]
    rotary_dim = dimension_count or head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("RoPE dimension count must be even and within the head dimension")
    rotary = x[..., :rotary_dim]
    frequencies = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies[None, :]
    if interleaved:
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        rotated = np.empty_like(rotary)
        rotated[..., 0::2] = rotary[..., 0::2] * cos - rotary[..., 1::2] * sin
        rotated[..., 1::2] = rotary[..., 0::2] * sin + rotary[..., 1::2] * cos
    else:
        embedding = np.concatenate((angles, angles), axis=-1)[:, None, :]
        cos, sin = np.cos(embedding), np.sin(embedding)
        half = rotary_dim // 2
        rotated_half = np.concatenate((-rotary[..., half:], rotary[..., :half]), axis=-1)
        rotated = rotary * cos + rotated_half * sin
    if rotary_dim == head_dim:
        return rotated
    result = np.array(x, copy=True)
    result[..., :rotary_dim] = rotated
    return result
