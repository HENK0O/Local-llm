from __future__ import annotations

import os
import numpy as np
from numpy.typing import NDArray

try:
    from ._native import rms_norm as _native_rms_norm
except ImportError:
    _native_rms_norm = None

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


def linear_pair(x: Array, first: Array, second: Array) -> tuple[Array, Array]:
    """Apply two projections while allowing quantized backends to share dispatch."""
    pair = getattr(first, "matmul_pair", None)
    if pair is not None:
        result = pair(second, x)
        if result is not NotImplemented:
            return result
    return linear(x, first), linear(x, second)


def linear_qkv(x: Array, query: Array, key: Array, value: Array) -> tuple[Array, Array, Array]:
    fused = getattr(query, "matmul_qkv", None)
    if fused is not None:
        result = fused(key, value, x)
        if result is not NotImplemented:
            return result
    k, v = linear_pair(x, key, value)
    return linear(x, query), k, v


def linear_swiglu(x: Array, gate_weight: Array, up_weight: Array) -> Array:
    """Fuse quantized gate/up projections and SwiGLU when the backend supports it."""
    fused = getattr(gate_weight, "matmul_swiglu", None)
    if fused is not None:
        result = fused(up_weight, x)
        if result is not NotImplemented:
            return result
    return silu(linear(x, gate_weight)) * linear(x, up_weight)


def linear_add(x: Array, weight: Array, residual: Array) -> Array:
    """Fuse a quantized projection with its residual addition when possible."""
    fused = getattr(weight, "matmul_add", None)
    if fused is not None:
        return fused(x, residual)
    return residual + linear(x, weight)


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    if (_native_rms_norm is not None and os.environ.get("LOCAL_LLM_DISABLE_NATIVE") != "1"
            and x.dtype == np.float32 and weight.dtype == np.float32):
        return _native_rms_norm(x, weight, eps)
    variance = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)
    normalized = x * (1.0 / np.sqrt(variance + eps))
    return normalized * weight


def silu(x: Array) -> Array:
    x32 = x.astype(np.float32)
    positive = x32 >= 0
    result = np.empty_like(x32)
    result[positive] = x32[positive] / (1.0 + np.exp(-x32[positive]))
    exponential = np.exp(x32[~positive])
    result[~positive] = x32[~positive] * exponential / (1.0 + exponential)
    return result


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
               interleaved: bool = False, dimension_count: int | None = None,
               factors: tuple[Array, Array] | None = None) -> Array:
    """Apply split-half (HF) or adjacent-pair (GGUF) rotary embeddings."""
    head_dim = x.shape[-1]
    rotary_dim = dimension_count or head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("RoPE dimension count must be even and within the head dimension")
    rotary = x[..., :rotary_dim]
    cos, sin = factors if factors is not None else rope_factors(
        positions, theta, rotary_dim, interleaved
    )
    if interleaved:
        rotated = np.empty_like(rotary)
        rotated[..., 0::2] = rotary[..., 0::2] * cos - rotary[..., 1::2] * sin
        rotated[..., 1::2] = rotary[..., 0::2] * sin + rotary[..., 1::2] * cos
    else:
        half = rotary_dim // 2
        rotated_half = np.concatenate((-rotary[..., half:], rotary[..., :half]), axis=-1)
        rotated = rotary * cos + rotated_half * sin
    if rotary_dim == head_dim:
        return rotated
    result = np.array(x, copy=True)
    result[..., :rotary_dim] = rotated
    return result


def rope_factors(positions: NDArray[np.integer], theta: float, dimension: int,
                 interleaved: bool = False) -> tuple[Array, Array]:
    """Compute factors once per forward, shared by Q/K in every layer."""
    frequencies = 1.0 / (theta ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
    angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies[None, :]
    if not interleaved:
        angles = np.concatenate((angles, angles), axis=-1)
    return np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
