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


class ModelTests(unittest.TestCase):
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
