from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .cache import KVCache
from .config import ModelConfig
from .ops import (
    apply_rope, linear, linear_add, linear_qkv, linear_swiglu, rms_norm, rope_factors, sigmoid, softmax,
)
from .profiling import OperationProfiler
from .safetensors import load_directory as load_safetensors_directory

Array = NDArray[np.floating]


class LlamaModel:
    """A readable NumPy forward pass for decoder-only Llama architectures."""

    def __init__(self, config: ModelConfig, weights: Dict[str, Array]) -> None:
        config.validate()
        self.config = config
        self.weights = {
            name: value if hasattr(value, "matmul") else np.asarray(value)
            for name, value in weights.items()
        }
        self.profiler: Optional[OperationProfiler] = None
        self._validate_weights()

    def start_profiling(self) -> OperationProfiler:
        self.profiler = OperationProfiler()
        return self.profiler

    def stop_profiling(self) -> Optional[OperationProfiler]:
        profiler = self.profiler
        self.profiler = None
        return profiler

    @classmethod
    def from_directory(cls, model_dir: Path) -> "LlamaModel":
        model_dir = Path(model_dir)
        config = ModelConfig.load(model_dir / "config.json")
        weights_path = model_dir / "weights.npz"
        if weights_path.exists():
            with np.load(weights_path, allow_pickle=False) as archive:
                weights = {name: archive[name] for name in archive.files}
        else:
            weights = load_safetensors_directory(model_dir, np.float32)
        return cls(config, weights)

    def new_cache(self, capacity: Optional[int] = None) -> KVCache:
        return KVCache(self.config, capacity or self.config.max_position_embeddings)

    def _expected_shapes(self) -> Dict[str, Tuple[int, ...]]:
        c = self.config
        shapes: Dict[str, Tuple[int, ...]] = {
            "model.embed_tokens.weight": (c.vocab_size, c.hidden_size),
            "model.norm.weight": (c.hidden_size,),
        }
        if not c.tie_word_embeddings:
            shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size)
        for layer in range(c.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (c.hidden_size,),
                    f"{prefix}.self_attn.q_proj.weight": (c.num_attention_heads * c.head_dim, c.hidden_size),
                    f"{prefix}.self_attn.k_proj.weight": (c.num_key_value_heads * c.head_dim, c.hidden_size),
                    f"{prefix}.self_attn.v_proj.weight": (c.num_key_value_heads * c.head_dim, c.hidden_size),
                    f"{prefix}.self_attn.o_proj.weight": (c.hidden_size, c.num_attention_heads * c.head_dim),
                    f"{prefix}.post_attention_layernorm.weight": (c.hidden_size,),
                    f"{prefix}.mlp.gate_proj.weight": (c.intermediate_size, c.hidden_size),
                    f"{prefix}.mlp.up_proj.weight": (c.intermediate_size, c.hidden_size),
                    f"{prefix}.mlp.down_proj.weight": (c.hidden_size, c.intermediate_size),
                }
            )
            if c.qk_norm:
                shapes[f"{prefix}.self_attn.q_norm.weight"] = (c.head_dim,)
                shapes[f"{prefix}.self_attn.k_norm.weight"] = (c.head_dim,)
            if c.attention_gate:
                shapes[f"{prefix}.self_attn.gate_proj.weight"] = (
                    c.num_attention_heads * c.head_dim, c.hidden_size
                )
        return shapes

    def _validate_weights(self) -> None:
        errors = []
        for name, shape in self._expected_shapes().items():
            if name not in self.weights:
                errors.append(f"missing {name}")
            elif self.weights[name].shape != shape:
                errors.append(f"{name}: expected {shape}, got {self.weights[name].shape}")
        if errors:
            raise ValueError("invalid weights:\n  " + "\n  ".join(errors))

    def forward(self, token_ids: NDArray[np.integer], cache: Optional[KVCache] = None,
                *, last_logits_only: bool = False) -> Array:
        """Evaluate tokens; optionally project only the last position for generation."""
        logits, _ = self._forward(token_ids, cache, capture=False,
                                  last_logits_only=last_logits_only)
        return logits

    def forward_with_activations(self, token_ids: NDArray[np.integer]) -> Tuple[Array, Dict[str, Array]]:
        """Reference/debug path returning residual-stream checkpoints."""
        return self._forward(token_ids, cache=None, capture=True)

    def _forward(self, token_ids: NDArray[np.integer], cache: Optional[KVCache],
                 capture: bool, last_logits_only: bool = False) -> Tuple[Array, Dict[str, Array]]:
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty 1D array")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("token id outside vocabulary")

        start = cache.length if cache is not None else 0
        end = start + tokens.size
        if end > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if cache is not None and end > cache.capacity:
            raise ValueError("sequence exceeds KV cache capacity")

        profiler = self.profiler
        started = time.perf_counter() if profiler is not None else 0.0
        x = self.weights["model.embed_tokens.weight"][tokens]
        if profiler is not None:
            profiler.record("embeddings", time.perf_counter() - started)
        activations: Dict[str, Array] = {}
        if capture:
            activations["embeddings"] = np.asarray(x, dtype=np.float32).copy()
        positions = np.arange(start, end, dtype=np.int64)
        factors = rope_factors(positions, self.config.rope_theta,
                               self.config.rope_dimension_count or self.config.head_dim,
                               self.config.rope_interleaved)
        for layer_index in range(self.config.num_hidden_layers):
            x = self._layer(x, layer_index, positions, start, end, cache, factors)
            if capture:
                activations[f"layer.{layer_index}"] = np.asarray(x, dtype=np.float32).copy()

        if cache is not None:
            cache.length = end
        started = time.perf_counter() if profiler is not None else 0.0
        if last_logits_only:
            x = x[-1:]
        x = rms_norm(x, self.weights["model.norm.weight"], self.config.rms_norm_eps)
        if profiler is not None:
            profiler.record("final_norm", time.perf_counter() - started)
        output_weight = (
            self.weights["model.embed_tokens.weight"]
            if self.config.tie_word_embeddings
            else self.weights["lm_head.weight"]
        )
        if capture:
            activations["norm"] = np.asarray(x, dtype=np.float32).copy()
        started = time.perf_counter() if profiler is not None else 0.0
        logits = linear(x, output_weight).astype(np.float32, copy=False)
        if profiler is not None:
            profiler.record("vocab_projection", time.perf_counter() - started)
        if capture:
            activations["logits"] = logits.copy()
        return logits, activations

    def _layer(
        self,
        x: Array,
        layer_index: int,
        positions: NDArray[np.integer],
        start: int,
        end: int,
        cache: Optional[KVCache],
        factors: Tuple[Array, Array],
    ) -> Array:
        c = self.config
        prefix = f"model.layers.{layer_index}"
        profiler = self.profiler
        residual = x
        started = time.perf_counter() if profiler is not None else 0.0
        hidden = rms_norm(x, self.weights[f"{prefix}.input_layernorm.weight"], c.rms_norm_eps)
        if profiler is not None:
            profiler.record("attention_norm", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        query, key, value = linear_qkv(
            hidden,
            self.weights[f"{prefix}.self_attn.q_proj.weight"],
            self.weights[f"{prefix}.self_attn.k_proj.weight"],
            self.weights[f"{prefix}.self_attn.v_proj.weight"],
        )
        if profiler is not None:
            profiler.record("qkv_projections", time.perf_counter() - started)
        query = query.reshape(-1, c.num_attention_heads, c.head_dim)
        key = key.reshape(-1, c.num_key_value_heads, c.head_dim)
        value = value.reshape(-1, c.num_key_value_heads, c.head_dim)
        started = time.perf_counter() if profiler is not None else 0.0
        if c.qk_norm:
            query = rms_norm(query, self.weights[f"{prefix}.self_attn.q_norm.weight"],
                             c.rms_norm_eps)
            key = rms_norm(key, self.weights[f"{prefix}.self_attn.k_norm.weight"],
                           c.rms_norm_eps)
        query = apply_rope(query, positions, c.rope_theta, c.rope_interleaved,
                           c.rope_dimension_count, factors)
        key = apply_rope(key, positions, c.rope_theta, c.rope_interleaved,
                         c.rope_dimension_count, factors)
        if profiler is not None:
            profiler.record("qk_norm_rope", time.perf_counter() - started)

        if cache is None:
            all_key, all_value = key, value
            key_start = 0
        else:
            layer_cache = cache.layers[layer_index]
            layer_cache.keys[start:end] = key
            layer_cache.values[start:end] = value
            all_key = layer_cache.keys[:end]
            all_value = layer_cache.values[:end]
            key_start = start

        groups = c.num_attention_heads // c.num_key_value_heads
        # Keep GQA keys/values in their compact KV-head layout. Reshaping the
        # queries exposes each KV head's query groups without np.repeat(), which
        # otherwise copied the complete cache twice in every layer and token.
        grouped_query = query.reshape(
            query.shape[0], c.num_key_value_heads, groups, c.head_dim
        )
        started = time.perf_counter() if profiler is not None else 0.0
        # Batched matrix products use BLAS for prefill while broadcasting KV
        # heads across query groups without materializing repeated keys/values.
        # Accelerate can leave spurious FP flags, as in ops.linear. Actual
        # non-finite results still propagate and are checked by reference tests.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = np.matmul(
                grouped_query.transpose(1, 2, 0, 3),
                all_key.transpose(1, 2, 0)[:, None, :, :],
            ) / np.sqrt(c.head_dim)
        # A one-token cached decode is necessarily the last position, so every
        # key visible in the cache is causal. Avoid allocating a mask per layer.
        if query.shape[0] > 1 or key_start == 0:
            query_absolute = key_start + np.arange(query.shape[0])
            key_positions = np.arange(all_key.shape[0])
            causal_mask = key_positions[None, :] > query_absolute[:, None]
            if np.any(causal_mask):
                scores = np.where(causal_mask[None, None, :, :], -np.inf, scores)
        if profiler is not None:
            profiler.record("attention_scores", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        probabilities = softmax(scores, axis=-1)
        if profiler is not None:
            profiler.record("attention_softmax", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            attention = np.matmul(
                probabilities, all_value.transpose(1, 0, 2)[:, None, :, :]
            ).transpose(2, 0, 1, 3)
        attention = attention.reshape(-1, c.num_attention_heads * c.head_dim)
        if profiler is not None:
            profiler.record("attention_values", time.perf_counter() - started)
        if c.attention_gate:
            started = time.perf_counter() if profiler is not None else 0.0
            attention *= sigmoid(linear(
                hidden, self.weights[f"{prefix}.self_attn.gate_proj.weight"]
            ))
            if profiler is not None:
                profiler.record("attention_gate", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        x = linear_add(
            attention, self.weights[f"{prefix}.self_attn.o_proj.weight"], residual
        )
        if profiler is not None:
            profiler.record("attention_output", time.perf_counter() - started)

        residual = x
        started = time.perf_counter() if profiler is not None else 0.0
        hidden = rms_norm(x, self.weights[f"{prefix}.post_attention_layernorm.weight"], c.rms_norm_eps)
        if profiler is not None:
            profiler.record("ffn_norm", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        activated = linear_swiglu(
            hidden,
            self.weights[f"{prefix}.mlp.gate_proj.weight"],
            self.weights[f"{prefix}.mlp.up_proj.weight"],
        )
        if profiler is not None:
            profiler.record("ffn_gate_up", time.perf_counter() - started)
        started = time.perf_counter() if profiler is not None else 0.0
        result = linear_add(
            activated, self.weights[f"{prefix}.mlp.down_proj.weight"], residual
        )
        if profiler is not None:
            profiler.record("ffn_down", time.perf_counter() - started)
        return result
