import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from local_llm.converters.baguette import convert_baguette
from local_llm.loading import load_runtime


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def fake_checkpoint():
    config = {
        "vocab_size": 8, "n_layer": 1, "n_head": 2, "n_kv_head": 1,
        "d_model": 8, "head_dim": 4, "d_ff": 12, "max_seq_len": 32,
        "rope_theta": 100000.0, "rms_eps": 1e-6, "tie_embeddings": True,
        "hybrid": False, "attn_gate": True, "rope_frac": 0.5,
        "zero_centered": True, "bos_id": 0, "eos_id": 0, "pad_id": 0,
    }
    rng = np.random.default_rng(4)

    def tensor(shape):
        return FakeTensor(rng.normal(0, 0.1, shape).astype(np.float16))

    state = {
        "embed_tokens.weight": tensor((8, 8)),
        "norm.weight": FakeTensor(np.zeros(8, dtype=np.float16)),
        "lm_head.weight": tensor((8, 8)),
        "layers.0.input_layernorm.weight": FakeTensor(np.zeros(8, dtype=np.float16)),
        "layers.0.mixer.q_proj.weight": tensor((8, 8)),
        "layers.0.mixer.k_proj.weight": tensor((4, 8)),
        "layers.0.mixer.v_proj.weight": tensor((4, 8)),
        "layers.0.mixer.o_proj.weight": tensor((8, 8)),
        "layers.0.mixer.gate_proj.weight": tensor((8, 8)),
        "layers.0.mixer.q_norm.weight": FakeTensor(np.zeros(4, dtype=np.float16)),
        "layers.0.mixer.k_norm.weight": FakeTensor(np.zeros(4, dtype=np.float16)),
        "layers.0.post_attention_layernorm.weight": FakeTensor(np.zeros(8, dtype=np.float16)),
        "layers.0.mlp.gate_proj.weight": tensor((12, 8)),
        "layers.0.mlp.up_proj.weight": tensor((12, 8)),
        "layers.0.mlp.down_proj.weight": tensor((8, 12)),
    }
    return {"model_cfg": config, "model": state, "stage": "sft", "step": 3}


class BaguetteConverterTests(unittest.TestCase):
    def test_converts_to_loadable_runtime_directory(self):
        checkpoint = fake_checkpoint()
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "baguette.pt"
            source.write_bytes(b"fake checkpoint")
            tokenizer = root / "tokenizer.json"
            vocab = {
                "<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
                "<think>": 3, "</think>": 4, "a": 5, "b": 6, "c": 7,
            }
            tokenizer.write_text(json.dumps({
                "version": "1.0",
                "added_tokens": [
                    {"id": index, "content": token, "special": True}
                    for token, index in list(vocab.items())[:5]
                ],
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
                "model": {"type": "BPE", "vocab": vocab, "merges": []},
            }), encoding="utf-8")
            output = root / "converted"
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                convert_baguette(source, tokenizer, output)

            model, loaded_tokenizer = load_runtime(output)
            self.assertEqual(model.config.rope_dimension_count, 2)
            self.assertTrue(model.config.qk_norm)
            self.assertTrue(model.config.attention_gate)
            self.assertEqual(loaded_tokenizer.eos_token_id, 2)
            np.testing.assert_array_equal(
                model.weights["model.norm.weight"], np.ones(8, dtype=np.float32)
            )
            self.assertEqual(model.forward(np.array([5, 6])).shape, (2, 8))

    def test_rejects_hybrid_checkpoint(self):
        checkpoint = fake_checkpoint()
        checkpoint["model_cfg"]["hybrid"] = True
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "baguette.pt"
            tokenizer = root / "tokenizer.json"
            source.write_bytes(b"fake")
            tokenizer.write_text("{}", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "DeltaNet"):
                    convert_baguette(source, tokenizer, root / "output")


if __name__ == "__main__":
    unittest.main()
