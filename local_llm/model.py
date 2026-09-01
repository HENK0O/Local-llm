from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .cache import KVCache
from .config import ModelConfig
from .ops import apply_rope, linear, rms_norm, silu, softmax
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
        self._validate_weights()

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

    def forward(self, token_ids: NDArray[np.integer], cache: Optional[KVCache] = None) -> Array:
        logits, _ = self._forward(token_ids, cache, capture=False)
        return logits

    def forward_with_activations(self, token_ids: NDArray[np.integer]) -> Tuple[Array, Dict[str, Array]]:
        """Reference/debug path returning residual-stream checkpoints."""
        return self._forward(token_ids, cache=None, capture=True)

    def _forward(self, token_ids: NDArray[np.integer], cache: Optional[KVCache],
                 capture: bool) -> Tuple[Array, Dict[str, Array]]:
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

        x = self.weights["model.embed_tokens.weight"][tokens]
        activations: Dict[str, Array] = {}
        if capture:
            activations["embeddings"] = np.asarray(x, dtype=np.float32).copy()
        positions = np.arange(start, end, dtype=np.int64)
        for layer_index in range(self.config.num_hidden_layers):
            x = self._layer(x, layer_index, positions, start, end, cache)
            if capture:
                activations[f"layer.{layer_index}"] = np.asarray(x, dtype=np.float32).copy()

        if cache is not None:
            cache.length = end
        x = rms_norm(x, self.weights["model.norm.weight"], self.config.rms_norm_eps)
        output_weight = (
            self.weights["model.embed_tokens.weight"]
            if self.config.tie_word_embeddings
            else self.weights["lm_head.weight"]
        )
        if capture:
            activations["norm"] = np.asarray(x, dtype=np.float32).copy()
        logits = linear(x, output_weight).astype(np.float32, copy=False)
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
    ) -> Array:
        c = self.config
        prefix = f"model.layers.{layer_index}"
        residual = x
        hidden = rms_norm(x, self.weights[f"{prefix}.input_layernorm.weight"], c.rms_norm_eps)
        query = linear(hidden, self.weights[f"{prefix}.self_attn.q_proj.weight"])
        key = linear(hidden, self.weights[f"{prefix}.self_attn.k_proj.weight"])
        value = linear(hidden, self.weights[f"{prefix}.self_attn.v_proj.weight"])
        query = query.reshape(-1, c.num_attention_heads, c.head_dim)
        key = key.reshape(-1, c.num_key_value_heads, c.head_dim)
        value = value.reshape(-1, c.num_key_value_heads, c.head_dim)
        query = apply_rope(query, positions, c.rope_theta, c.rope_interleaved)
        key = apply_rope(key, positions, c.rope_theta, c.rope_interleaved)

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
        all_key = np.repeat(all_key, groups, axis=1)
        all_value = np.repeat(all_value, groups, axis=1)
        scores = np.einsum("thd,shd->hts", query, all_key, optimize=True) / np.sqrt(c.head_dim)
        query_absolute = key_start + np.arange(query.shape[0])
        key_positions = np.arange(all_key.shape[0])
        causal_mask = key_positions[None, :] > query_absolute[:, None]
        scores = np.where(causal_mask[None, :, :], -np.inf, scores)
        probabilities = softmax(scores, axis=-1)
        attention = np.einsum("hts,shd->thd", probabilities, all_value, optimize=True)
        attention = attention.reshape(-1, c.num_attention_heads * c.head_dim)
        x = residual + linear(attention, self.weights[f"{prefix}.self_attn.o_proj.weight"])

        residual = x
        hidden = rms_norm(x, self.weights[f"{prefix}.post_attention_layernorm.weight"], c.rms_norm_eps)
        gate = silu(linear(hidden, self.weights[f"{prefix}.mlp.gate_proj.weight"]))
        up = linear(hidden, self.weights[f"{prefix}.mlp.up_proj.weight"])
        return residual + linear(gate * up, self.weights[f"{prefix}.mlp.down_proj.weight"])
