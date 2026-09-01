from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .tokenizer import ByteTokenizer


def make_toy_weights(config: ModelConfig, seed: int = 42) -> Dict[str, NDArray[np.float32]]:
    """Create deterministic random weights for tests and runtime exploration."""
    rng = np.random.default_rng(seed)

    def matrix(shape: tuple) -> NDArray[np.float32]:
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    weights: Dict[str, NDArray[np.float32]] = {
        "model.embed_tokens.weight": matrix((config.vocab_size, config.hidden_size)),
        "model.norm.weight": np.ones(config.hidden_size, dtype=np.float32),
    }
    if not config.tie_word_embeddings:
        weights["lm_head.weight"] = matrix((config.vocab_size, config.hidden_size))
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": np.ones(config.hidden_size, dtype=np.float32),
                f"{prefix}.self_attn.q_proj.weight": matrix(
                    (config.num_attention_heads * config.head_dim, config.hidden_size)
                ),
                f"{prefix}.self_attn.k_proj.weight": matrix(
                    (config.num_key_value_heads * config.head_dim, config.hidden_size)
                ),
                f"{prefix}.self_attn.v_proj.weight": matrix(
                    (config.num_key_value_heads * config.head_dim, config.hidden_size)
                ),
                f"{prefix}.self_attn.o_proj.weight": matrix(
                    (config.hidden_size, config.num_attention_heads * config.head_dim)
                ),
                f"{prefix}.post_attention_layernorm.weight": np.ones(config.hidden_size, dtype=np.float32),
                f"{prefix}.mlp.gate_proj.weight": matrix((config.intermediate_size, config.hidden_size)),
                f"{prefix}.mlp.up_proj.weight": matrix((config.intermediate_size, config.hidden_size)),
                f"{prefix}.mlp.down_proj.weight": matrix((config.hidden_size, config.intermediate_size)),
            }
        )
    return weights


def create_toy_model(output: Path, seed: int = 42) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.save(output / "config.json")
    tokenizer.save(output / "tokenizer.json")
    np.savez(output / "weights.npz", **make_toy_weights(config, seed))
    return output

