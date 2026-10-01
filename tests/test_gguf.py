import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_llm.config import ModelConfig
from local_llm.gguf import (
    ARRAY, BOOL, FLOAT32, GGUFError, GGUFReader, Q4Matrix, Q8Matrix, STRING, UINT32,
    q4_backend_name, q8_backend_name,
)
from local_llm.loading import load_runtime
from local_llm.model import LlamaModel
from local_llm.ops import silu
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


def write_q4_gguf(path: Path, name: str, shape, storage: np.ndarray):
    metadata_blob = _string("general.architecture") + struct.pack("<I", STRING) + _string("llama")
    infos = _string(name) + struct.pack("<I", len(shape))
    infos += b"".join(struct.pack("<Q", dim) for dim in reversed(shape))
    infos += struct.pack("<IQ", 2, 0)  # GGML_TYPE_Q4_0, offset zero.
    prefix = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + metadata_blob + infos
    prefix += b"\0" * ((-len(prefix)) % 32)
    with path.open("wb") as handle:
        handle.write(prefix)
        handle.write(storage.tobytes())


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


def q4_storage(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    packed = (
        (quantized[..., :16].astype(np.int16) + 8)
        | ((quantized[..., 16:].astype(np.int16) + 8) << 4)
    ).astype(np.uint8)
    storage = np.empty(quantized.shape[:-1], dtype=Q4Matrix._dtype)
    storage["scale"] = scales
    storage["values"] = packed
    return storage


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
            "tokenizer.chat_template": "{{ messages[0]['content'] }} default",
            "tokenizer.chat_templates": ["short"],
            "tokenizer.chat_template.short": "{{ messages[0]['content'] }} short",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.gguf"
            write_gguf(path, metadata, gguf_weights)
            loaded, tokenizer = load_runtime(path)
            expected = LlamaModel(config, weights).forward(np.array([1, 5, 9, 3]))
            actual = loaded.forward(np.array([1, 5, 9, 3]))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            self.assertEqual(tokenizer.vocab_size, 32)
            self.assertEqual(tokenizer.chat_template, {
                "default": "{{ messages[0]['content'] }} default",
                "short": "{{ messages[0]['content'] }} short",
            })

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
            residual = rng.normal(size=actual.shape).astype(np.float32)
            np.testing.assert_allclose(matrix.matmul_add(x, residual), actual + residual,
                                       rtol=2e-5, atol=1e-4)
            self.assertEqual(matrix[2].shape, (64,))

    def test_q4_matrix_unpacks_ggml_nibble_order_and_multiplies(self):
        rng = np.random.default_rng(29)
        quantized = rng.integers(-8, 8, size=(7, 2, 32), dtype=np.int8)
        scales = rng.uniform(0.01, 0.2, size=(7, 2)).astype(np.float16)
        storage = q4_storage(quantized, scales)
        dequantized = (
            quantized.astype(np.float32) * scales.astype(np.float32)[..., None]
        ).reshape(7, 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "q4.bin"
            raw_path.write_bytes(storage.tobytes())
            matrix = Q4Matrix(raw_path, 0, dequantized.shape)
            inputs = rng.normal(size=(4, 64)).astype(np.float32)
            np.testing.assert_allclose(matrix.matmul(inputs), inputs @ dequantized.T,
                                       rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(matrix[2], dequantized[2], rtol=0, atol=0)
            self.assertEqual(matrix.nbytes, 7 * 2 * 18)

            gguf_path = root / "q4.gguf"
            write_q4_gguf(gguf_path, "weight", dequantized.shape, storage)
            reader = GGUFReader(gguf_path)
            loaded = reader.tensor("weight")
            self.assertIsInstance(loaded, Q4Matrix)
            self.assertEqual(reader.tensors["weight"].type_name, "Q4_0")
            np.testing.assert_allclose(loaded[1], dequantized[1], rtol=0, atol=0)

    def test_q4_embedding_and_output_projection_run_inside_model(self):
        config = ModelConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        )
        weights = make_toy_weights(config, seed=37)
        source = weights["model.embed_tokens.weight"]
        blocks = source.reshape(32, 1, 32)
        scales = np.maximum(np.max(np.abs(blocks), axis=-1) / 7.0, 1e-8).astype(np.float16)
        quantized = np.rint(blocks / scales.astype(np.float32)[..., None]).clip(-8, 7).astype(np.int8)
        storage = q4_storage(quantized, scales)
        dequantized = (quantized.astype(np.float32) * scales.astype(np.float32)[..., None]).reshape(32, 32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding-q4.bin"
            path.write_bytes(storage.tobytes())
            q4_weights = dict(weights)
            q4_weights["model.embed_tokens.weight"] = Q4Matrix(path, 0, source.shape)
            expected_weights = dict(weights)
            expected_weights["model.embed_tokens.weight"] = dequantized
            tokens = np.array([1, 8, 3, 11], dtype=np.int64)
            actual = LlamaModel(config, q4_weights).forward(tokens)
            expected = LlamaModel(config, expected_weights).forward(tokens)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

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

    @unittest.skipUnless(q8_backend_name() == "native-cpp", "native Q8 extension is not built")
    def test_native_q8_pair_matches_two_numpy_kernels(self):
        rng = np.random.default_rng(41)
        first_storage = np.empty((513, 30), dtype=Q8Matrix._dtype)
        second_storage = np.empty((513, 30), dtype=Q8Matrix._dtype)
        for storage in (first_storage, second_storage):
            storage["scale"] = rng.uniform(0.001, 0.05, size=(513, 30)).astype(np.float16)
            storage["values"] = rng.integers(-127, 128, size=(513, 30, 32), dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, second_path = root / "first.bin", root / "second.bin"
            first_path.write_bytes(first_storage.tobytes())
            second_path.write_bytes(second_storage.tobytes())
            first = Q8Matrix(first_path, 0, (513, 960))
            second = Q8Matrix(second_path, 0, (513, 960))
            inputs = rng.normal(size=(4, 960)).astype(np.float32)
            actual_first, actual_second = first.matmul_pair(second, inputs)
            np.testing.assert_allclose(actual_first, first.matmul_numpy(inputs),
                                       rtol=2e-5, atol=1e-4)
            np.testing.assert_allclose(actual_second, second.matmul_numpy(inputs),
                                       rtol=2e-5, atol=1e-4)
            expected_swiglu = silu(first.matmul_numpy(inputs)) * second.matmul_numpy(inputs)
            np.testing.assert_allclose(first.matmul_swiglu(second, inputs), expected_swiglu,
                                       rtol=2e-4, atol=1e-3)

    @unittest.skipUnless(q8_backend_name() == "native-cpp", "native Q8 extension is not built")
    def test_native_qkv_supports_unequal_rows_and_batched_inputs(self):
        from local_llm._native import q8_matmul_qkv
        rng = np.random.default_rng(52)
        storages = []
        for rows in (65, 17, 17):
            storage = np.empty((rows, 3), dtype=Q8Matrix._dtype)
            storage["scale"] = rng.uniform(0.001, 0.05, size=(rows, 3)).astype(np.float16)
            storage["values"] = rng.integers(-127, 128, size=(rows, 3, 32), dtype=np.int8)
            storages.append(storage)
        for shape in [(96,), (1, 96), (2, 3, 96), (0, 96)]:
            x = rng.normal(size=shape).astype(np.float32)
            actual = q8_matmul_qkv(*storages, x)
            for storage, result in zip(storages, actual):
                weight = (storage["values"].astype(np.float32) *
                          storage["scale"].astype(np.float32)[..., None]).reshape(len(storage), 96)
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    expected = x @ weight.T
                np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            q8_matmul_qkv(*storages, np.ones(95, dtype=np.float32))

    @unittest.skipUnless(q8_backend_name() == "native-cpp", "native Q8 extension is not built")
    def test_scale_lookup_preserves_subnormals_signs_and_extremes(self):
        from local_llm._native import q8_matmul
        bits = np.array([0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff], dtype=np.uint16)
        storage = np.zeros((len(bits), 1), dtype=Q8Matrix._dtype)
        storage["scale"][:, 0] = bits.view(np.float16)
        storage["values"][:, 0, 0] = 1
        actual = q8_matmul(storage, np.ones(32, dtype=np.float32))
        np.testing.assert_array_equal(actual, bits.view(np.float16).astype(np.float32))

    @unittest.skipUnless(q4_backend_name() == "native-cpp", "native Q4 extension is not built")
    def test_native_q4_matches_numpy_kernel(self):
        rng = np.random.default_rng(31)
        storage = np.empty((513, 30), dtype=Q4Matrix._dtype)
        storage["scale"] = rng.uniform(0.001, 0.05, size=(513, 30)).astype(np.float16)
        storage["values"] = rng.integers(0, 256, size=(513, 30, 16), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native-q4.bin"
            path.write_bytes(storage.tobytes())
            matrix = Q4Matrix(path, 0, (513, 960))
            inputs = rng.normal(size=(4, 960)).astype(np.float32)
            np.testing.assert_allclose(
                matrix.matmul(inputs), matrix.matmul_numpy(inputs), rtol=2e-5, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()
