from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from ..config import ModelConfig
from ..safetensors import save_file


BAGUETTE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + "
    "message['content']|trim + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
_SPECIAL_IDS = {
    "<|endoftext|>": 0,
    "<|im_start|>": 1,
    "<|im_end|>": 2,
    "<think>": 3,
    "</think>": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_config(raw: Mapping[str, object]) -> ModelConfig:
    if bool(raw.get("hybrid", False)):
        raise ValueError("hybrid Baguette checkpoints with DeltaNet are not supported")
    hidden = int(raw["d_model"])
    heads = int(raw["n_head"])
    head_dim = int(raw["head_dim"])
    if hidden != heads * head_dim:
        raise ValueError("Baguette head_dim must equal d_model / n_head")
    rope_dimensions = int(head_dim * float(raw.get("rope_frac", 1.0)))
    rope_dimensions = max(2, rope_dimensions - rope_dimensions % 2)
    return ModelConfig(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(raw["d_ff"]),
        num_hidden_layers=int(raw["n_layer"]),
        num_attention_heads=heads,
        num_key_value_heads=int(raw["n_kv_head"]),
        max_position_embeddings=int(raw["max_seq_len"]),
        rms_norm_eps=float(raw["rms_eps"]),
        rope_theta=float(raw["rope_theta"]),
        bos_token_id=int(raw.get("bos_id", 0)),
        eos_token_id=2,
        pad_token_id=int(raw.get("pad_id", 0)),
        tie_word_embeddings=bool(raw.get("tie_embeddings", True)),
        rope_dimension_count=rope_dimensions,
        qk_norm=True,
        attention_gate=bool(raw.get("attn_gate", True)),
    )


def _validate_tokenizer(path: Path, vocab_size: int) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tokenizer JSON: {path}") from exc
    model = raw.get("model", {})
    vocab = model.get("vocab", {}) if isinstance(model, dict) else {}
    if not isinstance(model, dict) or model.get("type") != "BPE" or not isinstance(vocab, dict):
        raise ValueError("Baguette tokenizer must use the Hugging Face BPE format")
    if len(vocab) != vocab_size:
        raise ValueError(f"tokenizer has {len(vocab)} entries, checkpoint expects {vocab_size}")
    wrong = {
        token: (expected, vocab.get(token))
        for token, expected in _SPECIAL_IDS.items()
        if vocab.get(token) != expected
    }
    if wrong:
        raise ValueError("Baguette tokenizer special token IDs do not match the checkpoint")


def _runtime_name(name: str, tied: bool) -> str | None:
    if name == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return None if tied else name
    if not name.startswith("layers."):
        raise ValueError(f"unsupported Baguette tensor {name!r}")
    name = "model." + name
    return name.replace(".mixer.", ".self_attn.")


def _is_norm(name: str) -> bool:
    return name.endswith((
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )) or name == "model.norm.weight"


def _expected_shapes(config: ModelConfig) -> Dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
    }
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.input_layernorm.weight": (config.hidden_size,),
            f"{prefix}.self_attn.q_proj.weight": (
                config.num_attention_heads * config.head_dim, config.hidden_size),
            f"{prefix}.self_attn.k_proj.weight": (
                config.num_key_value_heads * config.head_dim, config.hidden_size),
            f"{prefix}.self_attn.v_proj.weight": (
                config.num_key_value_heads * config.head_dim, config.hidden_size),
            f"{prefix}.self_attn.o_proj.weight": (
                config.hidden_size, config.num_attention_heads * config.head_dim),
            f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
            f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
            f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
            f"{prefix}.mlp.gate_proj.weight": (
                config.intermediate_size, config.hidden_size),
            f"{prefix}.mlp.up_proj.weight": (
                config.intermediate_size, config.hidden_size),
            f"{prefix}.mlp.down_proj.weight": (
                config.hidden_size, config.intermediate_size),
        })
        if config.attention_gate:
            shapes[f"{prefix}.self_attn.gate_proj.weight"] = (
                config.num_attention_heads * config.head_dim, config.hidden_size)
    return shapes


def _convert_weights(state: Mapping[str, object], config: ModelConfig,
                     zero_centered: bool) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {}
    for source_name, tensor in state.items():
        target_name = _runtime_name(source_name, config.tie_word_embeddings)
        if target_name is None:
            continue
        try:
            array = tensor.detach().cpu().numpy()
        except AttributeError as exc:
            raise ValueError(f"checkpoint value {source_name!r} is not a tensor") from exc
        if zero_centered and _is_norm(target_name):
            # Baguette stores an offset around one. Keep the addition in F32 so
            # conversion does not introduce a second F16 rounding step.
            array = array.astype(np.float32) + np.float32(1.0)
        else:
            array = array.astype(np.float16, copy=False)
        converted[target_name] = np.ascontiguousarray(array)
    expected = _expected_shapes(config)
    missing = set(expected) - set(converted)
    extra = set(converted) - set(expected)
    bad_shapes = {
        name: (expected[name], converted[name].shape)
        for name in set(expected) & set(converted)
        if converted[name].shape != expected[name]
    }
    if missing or extra or bad_shapes:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected: " + ", ".join(sorted(extra)))
        if bad_shapes:
            details.append("bad shapes: " + ", ".join(
                f"{name} expected {wanted}, got {actual}"
                for name, (wanted, actual) in sorted(bad_shapes.items())
            ))
        raise ValueError("converted checkpoint tensors do not match the runtime: " + "; ".join(details))
    return converted


def convert_baguette(checkpoint: Path, tokenizer: Path, output: Path) -> Path:
    """Convert a Baguette weights-only checkpoint to a local-llm directory."""
    checkpoint, tokenizer, output = Path(checkpoint), Path(tokenizer), Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer}")
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "converting .pt checkpoints requires PyTorch; inference remains PyTorch-free"
        ) from exc

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, dict):
        raise ValueError("Baguette checkpoint must contain a dictionary")
    raw_config = checkpoint_data.get("model_cfg")
    state = checkpoint_data.get("model")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain model_cfg and model dictionaries")
    config = _runtime_config(raw_config)
    _validate_tokenizer(tokenizer, config.vocab_size)
    weights = _convert_weights(state, config, bool(raw_config.get("zero_centered", True)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        config.save(staging / "config.json")
        shutil.copyfile(tokenizer, staging / "tokenizer.json")
        with (staging / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "bos_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": ["<|im_start|>", "<think>", "</think>"],
                "chat_template": BAGUETTE_CHAT_TEMPLATE,
            }, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        save_file(staging / "model.safetensors", weights, {
            "format": "pt",
            "source": "HENK0O/baguette",
        })
        with (staging / "conversion.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "source_checkpoint": checkpoint.name,
                "source_sha256": _sha256(checkpoint),
                "stage": checkpoint_data.get("stage"),
                "step": checkpoint_data.get("step"),
                "tokens_seen": checkpoint_data.get("tokens_seen"),
                "val_loss": checkpoint_data.get("val_loss"),
                "tensor_count": len(weights),
            }, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        staging.rename(output)
    return output
