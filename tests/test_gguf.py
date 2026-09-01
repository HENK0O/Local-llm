import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_llm.config import ModelConfig
from local_llm.gguf import (
    ARRAY, BOOL, FLOAT32, GGUFError, GGUFReader, Q8Matrix, STRING, UINT32,
    q8_backend_name,
)
from local_llm.loading import load_runtime
from local_llm.model import LlamaModel
from local_llm.toy import make_toy_weights


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _metadata_value(value):
    if isinstance(value, bool):
        return BOOL, struct.pack("<B", value)
    if isinstance(value, str):
        return STRING, _string(value)
    if isinstance(value, float):
        return FLOAT32, struct.pack("<f", value)
    if isinstance(value, int):
        return UINT32, struct.pack("<I", value)
    if isinstance(value, list):
        element_type = STRING if not value or isinstance(value[0], str) else UINT32
        payload = struct.pack("<IQ", element_type, len(value))
        for item in value:
            payload += _string(item) if element_type == STRING else struct.pack("<I", item)
        return ARRAY, payload
    raise TypeError(type(value))


def write_gguf(path: Path, metadata, tensors):
    metadata_blob = bytearray()
    for key, value in metadata.items():
        value_type, payload = _metadata_value(value)
        metadata_blob += _string(key) + struct.pack("<I", value_type) + payload

    data_blob = bytearray()
    infos = bytearray()
    type_ids = {np.dtype("float32"): 0, np.dtype("float16"): 1}
    for name, tensor in tensors.items():
        tensor = np.asarray(tensor)
        while len(data_blob) % 32:
            data_blob.append(0)
        offset = len(data_blob)
        data_blob += tensor.tobytes()
        infos += _string(name) + struct.pack("<I", tensor.ndim)
        infos += b"".join(struct.pack("<Q", dim) for dim in reversed(tensor.shape))
        infos += struct.pack("<IQ", type_ids[tensor.dtype], offset)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    prefix = header + metadata_blob + infos
    prefix += b"\0" * ((-len(prefix)) % 32)
    with path.open("wb") as handle:
        handle.write(prefix)
        handle.write(data_blob)


def gguf_name(runtime_name):
    fixed = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
    if runtime_name in fixed:
        return fixed[runtime_name]
    parts = runtime_name.split(".")
    layer, suffix = int(parts[2]), ".".join(parts[3:])
    reverse = {
        "input_layernorm.weight": "attn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
    }
    return f"blk.{layer}.{reverse[suffix]}"


def permute_rope_weight(weight, heads, head_dim):
    return weight.reshape(heads, 2, head_dim // 2, -1).transpose(0, 2, 1, 3).reshape(weight.shape)


class GGUFTests(unittest.TestCase):
    def test_full_toy_model_matches_npz_layout(self):
        config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        weights = make_toy_weights(config, seed=19)
        gguf_weights = {}
        for name, value in weights.items():
            if name.endswith("self_attn.q_proj.weight"):
                value = permute_rope_weight(value, config.num_attention_heads, config.head_dim)
            elif name.endswith("self_attn.k_proj.weight"):
                value = permute_rope_weight(value, config.num_key_value_heads, config.head_dim)
            gguf_weights[gguf_name(name)] = value
        metadata = {
            "general.architecture": "llama", "general.alignment": 32,
            "llama.context_length": 32, "llama.embedding_length": 16, "llama.feed_forward_length": 24,
            "llama.block_count": 1, "llama.attention.head_count": 4,
            "llama.attention.head_count_kv": 2, "llama.attention.layer_norm_rms_epsilon": 1e-5,
            "llama.rope.freq_base": 10000.0, "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": [chr(256 + i) for i in range(32)], "tokenizer.ggml.merges": [],
            "tokenizer.ggml.token_type": [1] * 32, "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2, "tokenizer.ggml.padding_token_id": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.gguf"
            write_gguf(path, metadata, gguf_weights)
            loaded, tokenizer = load_runtime(path)
            expected = LlamaModel(config, weights).forward(np.array([1, 5, 9, 3]))
            actual = loaded.forward(np.array([1, 5, 9, 3]))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            self.assertEqual(tokenizer.vocab_size, 32)

    def test_f16_tensor_is_memory_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.gguf"
            write_gguf(path, {"general.architecture": "llama"},
                       {"value": np.array([[1.5, -2.0]], dtype=np.float16)})
            tensor = GGUFReader(path).tensor("value")
            self.assertIsInstance(tensor, np.memmap)
            np.testing.assert_array_equal(tensor, [[1.5, -2.0]])

    def test_bad_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"NOPE" + b"\0" * 20)
            with self.assertRaises(GGUFError):
                GGUFReader(path)

    def test_q8_matrix_vector_kernel(self):
        rng = np.random.default_rng(3)
        weights = rng.normal(size=(7, 64)).astype(np.float32)
        blocks = weights.reshape(7, 2, 32)
        scales = np.max(np.abs(blocks), axis=-1) / 127.0
        quantized = np.rint(blocks / scales[..., None]).clip(-127, 127).astype(np.int8)
        storage = np.empty((7, 2), dtype=Q8Matrix._dtype)
        storage["scale"] = scales.astype(np.float16)
        storage["values"] = quantized
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q8.bin"
            with path.open("wb") as handle:
                handle.write(storage.tobytes())
            matrix = Q8Matrix(path, 0, weights.shape)
            x = rng.normal(size=(4, 64)).astype(np.float32)
            expected = x @ weights.T
            actual = matrix.matmul(x)
            np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.12)
            self.assertEqual(matrix[2].shape, (64,))

    @unittest.skipUnless(q8_backend_name() == "native-cpp", "native Q8 extension is not built")
    def test_native_q8_matches_numpy_kernel(self):
        rng = np.random.default_rng(17)
        storage = np.empty((513, 30), dtype=Q8Matrix._dtype)
        storage["scale"] = rng.uniform(0.001, 0.05, size=(513, 30)).astype(np.float16)
        storage["values"] = rng.integers(-127, 128, size=(513, 30, 32), dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native-q8.bin"
            path.write_bytes(storage.tobytes())
            matrix = Q8Matrix(path, 0, (513, 960))
            inputs = rng.normal(size=(4, 960)).astype(np.float32)
            np.testing.assert_allclose(
                matrix.matmul(inputs), matrix.matmul_numpy(inputs), rtol=2e-5, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()
