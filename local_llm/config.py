from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    pad_token_id: Optional[int] = 0
    tie_word_embeddings: bool = False
    rope_interleaved: bool = False
    rope_dimension_count: Optional[int] = None
    qk_norm: bool = False
    attention_gate: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        rotary_dim = self.rope_dimension_count or self.head_dim
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("rope_dimension_count must be even and within the head dimension")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        if raw.get("model_type", "llama") != "llama":
            raise ValueError(f"unsupported architecture: {raw['model_type']!r}")
        if raw.get("rope_scaling"):
            raise ValueError("RoPE scaling is not supported by this runtime")
        if raw.get("attention_bias") or raw.get("mlp_bias"):
            raise ValueError("biased attention/MLP projections are not supported")
        fields = cls.__dataclass_fields__
        values = {name: raw[name] for name in fields if name in raw}
        if "num_key_value_heads" not in values and "num_attention_heads" in values:
            values["num_key_value_heads"] = values["num_attention_heads"]
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.__dict__, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
