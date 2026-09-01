import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from local_llm.evaluation import (
    capture_trace,
    compare_traces,
    evaluate_runtime,
    load_trace,
    save_trace,
)
from local_llm.loading import load_runtime
from local_llm.toy import create_toy_model


class EvaluationTests(unittest.TestCase):
    def test_cached_logits_match_full_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_toy_model(Path(directory) / "model")
            model, tokenizer = load_runtime(path)
            prompt = tokenizer.encode("cache")
            cached = capture_trace(model, prompt, 4, use_cache=True)
            uncached = capture_trace(model, prompt, 4, use_cache=False)
            comparison = compare_traces(cached, uncached, 1e-5, 1e-5)
            self.assertTrue(comparison.within_tolerance)
            self.assertTrue(comparison.greedy_tokens_identical)

    def test_trace_round_trip_and_detects_logit_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_toy_model(root / "model")
            model, tokenizer = load_runtime(path)
            trace = capture_trace(model, tokenizer.encode("trace"), 3, use_cache=True)
            reference = root / "reference.npz"
            save_trace(trace, reference, "fingerprint")
            loaded = load_trace(reference, "fingerprint")
            self.assertTrue(compare_traces(trace, loaded, 0.0, 0.0).within_tolerance)

            changed = trace.decision_logits.copy()
            changed[0, 0] += 0.1
            comparison = compare_traces(
                replace(trace, decision_logits=changed), loaded, 1e-5, 1e-5
            )
            self.assertFalse(comparison.within_tolerance)
            self.assertAlmostEqual(comparison.max_absolute_error, 0.1, places=5)

    def test_saved_runtime_reference_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_toy_model(root / "model")
            reference = root / "baseline.npz"
            first = evaluate_runtime(path, "baseline", 3, save_reference=reference)
            second = evaluate_runtime(path, "baseline", 3, reference=reference)
            self.assertTrue(first.passed)
            self.assertTrue(second.passed)
            self.assertIsNotNone(second.runtime_vs_reference)
            self.assertEqual(second.runtime_vs_reference.max_absolute_error, 0.0)

    def test_trace_rejects_a_different_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_toy_model(root / "model")
            model, tokenizer = load_runtime(path)
            trace = capture_trace(model, tokenizer.encode("x"), 2, use_cache=True)
            reference = root / "reference.npz"
            save_trace(trace, reference, "first")
            with self.assertRaisesRegex(ValueError, "different model"):
                load_trace(reference, "second")


if __name__ == "__main__":
    unittest.main()
