import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_llm.config import ModelConfig
from local_llm.generation import generate, greedy_generate_without_cache
from local_llm.model import LlamaModel
from local_llm.tokenizer import ByteTokenizer
from local_llm.toy import create_toy_model, make_toy_weights


def tiny_model() -> LlamaModel:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaModel(config, make_toy_weights(config, seed=7))


def tiny_gated_model() -> LlamaModel:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rope_dimension_count=2,
        qk_norm=True,
        attention_gate=True,
    )
    weights = make_toy_weights(config, seed=8)
    rng = np.random.default_rng(9)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.self_attn"
        weights[f"{prefix}.q_norm.weight"] = np.ones(config.head_dim, dtype=np.float32)
        weights[f"{prefix}.k_norm.weight"] = np.ones(config.head_dim, dtype=np.float32)
        weights[f"{prefix}.gate_proj.weight"] = rng.normal(
            0, 0.1, (config.hidden_size, config.hidden_size)
        ).astype(np.float32)
    return LlamaModel(config, weights)


class ModelTests(unittest.TestCase):
    def test_operation_profiler_is_opt_in_and_records_forward_sections(self):
        model = tiny_model()
        self.assertIsNone(model.profiler)
        profiler = model.start_profiling()
        model.forward(np.array([1, 2, 3]))
        self.assertIs(model.stop_profiling(), profiler)
        operations = {entry.operation for entry in profiler.entries()}
        self.assertIn("qkv_projections", operations)
        self.assertIn("attention_scores", operations)
        self.assertIn("ffn_gate_up", operations)
        self.assertIn("vocab_projection", operations)
        self.assertGreater(profiler.total_seconds, 0.0)
        json.dumps(profiler.to_dict())
        self.assertIsNone(model.profiler)

    def test_gated_qk_norm_cached_logits_match_full_forward(self):
        model = tiny_gated_model()
        tokens = np.array([1, 5, 7, 9, 4], dtype=np.int64)
        expected = model.forward(tokens)
        cache = model.new_cache(capacity=16)
        actual = np.concatenate([
            model.forward(tokens[:3], cache),
            model.forward(tokens[3:4], cache),
            model.forward(tokens[4:], cache),
        ])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_prefill_and_incremental_logits_match_full_forward(self):
        model = tiny_model()
        tokens = np.array([1, 5, 7, 9, 4], dtype=np.int64)
        expected = model.forward(tokens)

        cache = model.new_cache(capacity=16)
        parts = [model.forward(tokens[:3], cache)]
        parts.append(model.forward(tokens[3:4], cache))
        parts.append(model.forward(tokens[4:], cache))
        actual = np.concatenate(parts, axis=0)

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(cache.length, len(tokens))

    def test_cached_greedy_generation_matches_recomputation(self):
        model = tiny_model()
        prompt = [1, 3, 8]
        cached = generate(model, prompt, max_new_tokens=8).token_ids
        reference = greedy_generate_without_cache(model, prompt, max_new_tokens=8)
        self.assertEqual(cached, reference)

    def test_cache_memory_accounting(self):
        model = tiny_model()
        cache = model.new_cache(capacity=10)
        expected = 2 * 10 * 2 * model.config.num_key_value_heads * model.config.head_dim * 4
        self.assertEqual(cache.nbytes, expected)

    def test_directory_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_toy_model(Path(directory))
            model = LlamaModel.from_directory(path)
            tokenizer = ByteTokenizer.load(path / "tokenizer.json")
            logits = model.forward(np.asarray(tokenizer.encode("ok")))
            self.assertEqual(logits.shape, (3, tokenizer.vocab_size))

    def test_activation_trace_contains_each_layer(self):
        model = tiny_model()
        logits, activations = model.forward_with_activations(np.array([1, 2, 3]))
        self.assertEqual(set(activations), {"embeddings", "layer.0", "layer.1", "norm", "logits"})
        np.testing.assert_array_equal(activations["logits"], logits)


if __name__ == "__main__":
    unittest.main()
