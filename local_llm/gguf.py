"""Small GGUF v3 reader for unquantized Llama checkpoints."""

from __future__ import annotations

import struct
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .config import ModelConfig
from .tokenizer import BPETokenizer

try:
    from ._native import q8_matmul as _native_q8_matmul
except ImportError:
    _native_q8_matmul = None


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
F32, F16, Q8_0, BF16 = 0, 1, 8, 30

_SCALARS: Mapping[int, Tuple[str, int]] = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}
_TENSOR_DTYPES: Mapping[int, Tuple[np.dtype, int, str]] = {
    F32: (np.dtype("<f4"), 4, "F32"),
    F16: (np.dtype("<f2"), 2, "F16"),
    BF16: (np.dtype("<u2"), 2, "BF16"),
}
_TYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return _TYPE_NAMES.get(self.ggml_type, f"type-{self.ggml_type}")


class Q8Matrix:
    """GGML Q8_0 matrix: one FP16 scale and 32 signed bytes per block."""

    block_size = 32
    storage_bytes = 34
    _dtype = np.dtype([("scale", "<f2"), ("values", "i1", (block_size,))], align=False)

    def __init__(self, path: Path, offset: int, shape: Tuple[int, ...]) -> None:
        if len(shape) != 2 or shape[-1] % self.block_size:
            raise GGUFError(f"Q8_0 runtime supports 2D matrices with input size divisible by 32, got {shape}")
        self.path, self.shape = Path(path), shape
        self.blocks_per_row = shape[-1] // self.block_size
        self.blocks = np.memmap(path, mode="r", dtype=self._dtype, offset=offset,
                                shape=(shape[0], self.blocks_per_row))

    @property
    def nbytes(self) -> int:
        return self.blocks.nbytes

    def __getitem__(self, item: Any) -> np.ndarray:
        blocks = self.blocks[item]
        values = blocks["values"].astype(np.float32)
        result = values * blocks["scale"].astype(np.float32)[..., None]
        return result.reshape((*result.shape[:-2], self.shape[-1]))

    def matmul_numpy(self, x: np.ndarray, rows_per_chunk: int = 256) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.shape[-1] != self.shape[-1]:
            raise ValueError(f"Q8 matmul input size {values.shape[-1]} != {self.shape[-1]}")
        flat = values.reshape(-1, self.blocks_per_row, self.block_size)
        output = np.empty((flat.shape[0], self.shape[0]), dtype=np.float32)
        for start in range(0, self.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, self.shape[0])
            blocks = self.blocks[start:end]
            quantized = blocks["values"].astype(np.float32)
            scales = blocks["scale"].astype(np.float32)
            output[:, start:end] = np.einsum(
                "tbi,obi,ob->to", flat, quantized, scales, optimize=True
            )
        return output.reshape((*values.shape[:-1], self.shape[0]))

    def matmul(self, x: np.ndarray, rows_per_chunk: int = 256) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.shape[-1] != self.shape[-1]:
            raise ValueError(f"Q8 matmul input size {values.shape[-1]} != {self.shape[-1]}")
        if _native_q8_matmul is not None and os.environ.get("LOCAL_LLM_DISABLE_NATIVE") != "1":
            return _native_q8_matmul(self.blocks, np.ascontiguousarray(values))
        return self.matmul_numpy(values, rows_per_chunk)


def q8_backend_name() -> str:
    if _native_q8_matmul is not None and os.environ.get("LOCAL_LLM_DISABLE_NATIVE") != "1":
        return "native-cpp"
    return "numpy"


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of GGUF file")
    return data


def _read_scalar(handle: BinaryIO, value_type: int) -> Any:
    try:
        fmt, size = _SCALARS[value_type]
    except KeyError as exc:
        raise GGUFError(f"unsupported GGUF metadata type {value_type}") from exc
    return struct.unpack(fmt, _read_exact(handle, size))[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_scalar(handle, UINT64)
    if length > 256 * 1024 * 1024:
        raise GGUFError(f"unreasonable GGUF string length: {length}")
    try:
        return _read_exact(handle, length).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError("invalid UTF-8 string in GGUF metadata") from exc


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == STRING:
        return _read_string(handle)
    if value_type == BOOL:
        value = _read_scalar(handle, UINT8)
        if value not in (0, 1):
            raise GGUFError(f"invalid GGUF boolean value {value}")
        return bool(value)
    if value_type == ARRAY:
        element_type = _read_scalar(handle, UINT32)
        if element_type == ARRAY:
            raise GGUFError("nested GGUF arrays are not supported")
        length = _read_scalar(handle, UINT64)
        if length > 100_000_000:
            raise GGUFError(f"unreasonable GGUF array length: {length}")
        return [_read_value(handle, element_type) for _ in range(length)]
    return _read_scalar(handle, value_type)


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError(f"alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


class GGUFReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.version = 0
        self.alignment = DEFAULT_ALIGNMENT
        self.data_offset = 0
        self._parse()

    def _parse(self) -> None:
        file_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                raise GGUFError(f"{self.path}: invalid GGUF magic")
            self.version = _read_scalar(handle, UINT32)
            if self.version != GGUF_VERSION:
                raise GGUFError(f"unsupported GGUF version {self.version}; expected 3")
            tensor_count = _read_scalar(handle, UINT64)
            metadata_count = _read_scalar(handle, UINT64)
            if tensor_count > 1_000_000 or metadata_count > 1_000_000:
                raise GGUFError("unreasonable GGUF header counts")
            for _ in range(metadata_count):
                key = _read_string(handle)
                if key in self.metadata:
                    raise GGUFError(f"duplicate GGUF metadata key {key}")
                self.metadata[key] = _read_value(handle, _read_scalar(handle, UINT32))
            self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
            for _ in range(tensor_count):
                name = _read_string(handle)
                dimensions = _read_scalar(handle, UINT32)
                if dimensions > 8:
                    raise GGUFError(f"tensor {name} has too many dimensions")
                ggml_shape = tuple(_read_scalar(handle, UINT64) for _ in range(dimensions))
                info = TensorInfo(name, tuple(reversed(ggml_shape)), _read_scalar(handle, UINT32),
                                  _read_scalar(handle, UINT64))
                if name in self.tensors:
                    raise GGUFError(f"duplicate GGUF tensor {name}")
                self.tensors[name] = info
            self.data_offset = _align(handle.tell(), self.alignment)
        if self.data_offset > file_size:
            raise GGUFError("GGUF tensor data starts beyond end of file")
        for info in self.tensors.values():
            if info.ggml_type in _TENSOR_DTYPES or info.ggml_type == Q8_0:
                elements = int(np.prod(info.shape, dtype=np.int64))
                if info.ggml_type == Q8_0:
                    if elements % Q8Matrix.block_size:
                        raise GGUFError(f"Q8_0 tensor {info.name} has a partial block")
                    size = elements // Q8Matrix.block_size * Q8Matrix.storage_bytes
                else:
                    size = elements * _TENSOR_DTYPES[info.ggml_type][1]
                if info.offset % self.alignment or self.data_offset + info.offset + size > file_size:
                    raise GGUFError(f"invalid data range for tensor {info.name}")

    def tensor(self, name: str, float_dtype: Optional[np.dtype] = None) -> np.ndarray:
        try:
            info = self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor {name!r} not found") from exc
        if info.ggml_type == Q8_0:
            return Q8Matrix(self.path, self.data_offset + info.offset, info.shape)
        if info.ggml_type not in _TENSOR_DTYPES:
            raise GGUFError(f"tensor {name} uses unsupported quantization type {info.ggml_type}")
        dtype, _, _ = _TENSOR_DTYPES[info.ggml_type]
        if not info.shape or int(np.prod(info.shape, dtype=np.int64)) == 0:
            raw = np.empty(info.shape, dtype=dtype)
        else:
            raw = np.memmap(self.path, mode="r", dtype=dtype,
                            offset=self.data_offset + info.offset, shape=info.shape)
        if info.ggml_type == BF16:
            return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if info.ggml_type in (F16, F32) and float_dtype is not None and raw.dtype != np.dtype(float_dtype):
            return raw.astype(float_dtype)
        return raw

    def load_unquantized_tensors(self) -> Dict[str, np.ndarray]:
        unsupported = [info for info in self.tensors.values() if info.ggml_type not in _TENSOR_DTYPES]
        if unsupported:
            kinds = ", ".join(sorted({info.type_name for info in unsupported}))
            raise GGUFError(f"quantized tensors are not supported yet: {kinds}")
        return {name: self.tensor(name) for name in self.tensors}


_FIXED_TENSOR_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYER_TENSOR_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def model_config(reader: GGUFReader) -> ModelConfig:
    metadata = reader.metadata
    architecture = metadata.get("general.architecture")
    if architecture != "llama":
        raise GGUFError(f"only GGUF architecture 'llama' is supported, got {architecture!r}")
    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens) if isinstance(tokens, list) else reader.tensors["token_embd.weight"].shape[0]

    def required(key: str) -> Any:
        if key not in metadata:
            raise GGUFError(f"missing required GGUF metadata {key}")
        return metadata[key]

    head_count = int(required("llama.attention.head_count"))
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=int(required("llama.embedding_length")),
        intermediate_size=int(required("llama.feed_forward_length")),
        num_hidden_layers=int(required("llama.block_count")),
        num_attention_heads=head_count,
        num_key_value_heads=int(metadata.get("llama.attention.head_count_kv", head_count)),
        max_position_embeddings=int(required("llama.context_length")),
        rms_norm_eps=float(metadata.get("llama.attention.layer_norm_rms_epsilon", 1e-5)),
        rope_theta=float(metadata.get("llama.rope.freq_base", 10000.0)),
        bos_token_id=metadata.get("tokenizer.ggml.bos_token_id"),
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
        pad_token_id=metadata.get("tokenizer.ggml.padding_token_id"),
        tie_word_embeddings="output.weight" not in reader.tensors,
        rope_interleaved=True,
    )


def model_weights(reader: GGUFReader, float_dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for gguf_name, runtime_name in _FIXED_TENSOR_NAMES.items():
        if gguf_name in reader.tensors:
            result[runtime_name] = reader.tensor(gguf_name, float_dtype)
    for layer in range(int(reader.metadata.get("llama.block_count", 0))):
        for suffix, runtime_suffix in _LAYER_TENSOR_NAMES.items():
            gguf_name = f"blk.{layer}.{suffix}"
            if gguf_name in reader.tensors:
                result[f"model.layers.{layer}.{runtime_suffix}"] = reader.tensor(gguf_name, float_dtype)
    return result


def tokenizer(reader: GGUFReader) -> BPETokenizer:
    metadata = reader.metadata
    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise GGUFError("GGUF tokenizer.ggml.tokens is missing or invalid")
    if not isinstance(merges, list):
        raise GGUFError("only GGUF BPE tokenizers with tokenizer.ggml.merges are supported")
    token_types = metadata.get("tokenizer.ggml.token_type", [1] * len(tokens))
    special_ids = {
        value for key, value in metadata.items()
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id") and isinstance(value, int)
    }
    added_tokens = [
        {"id": index, "content": value, "special": index in special_ids or
         (index < len(token_types) and token_types[index] in (3, 4))}
        for index, value in enumerate(tokens)
    ]
    pre = str(metadata.get("tokenizer.ggml.pre", "")).lower()
    return BPETokenizer(
        {value: index for index, value in enumerate(tokens)}, merges, added_tokens,
        bos_token_id=metadata.get("tokenizer.ggml.bos_token_id"),
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
        pad_token_id=metadata.get("tokenizer.ggml.padding_token_id"),
        individual_digits="smollm" in pre,
    )
