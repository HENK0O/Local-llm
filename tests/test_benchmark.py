import json
import tempfile
import unittest
from pathlib import Path

from local_llm.benchmark import compare_report, load_report, model_fingerprint, run_benchmark, save_report
from local_llm.toy import create_toy_model


class BenchmarkTests(unittest.TestCase):
    def test_model_fingerprint_changes_with_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = create_toy_model(root / "model")
            first, size = model_fingerprint(model)
            self.assertGreater(size, 0)
            config = model / "config.json"
            config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
            second, _ = model_fingerprint(model)
            self.assertNotEqual(first, second)

    def test_single_file_fingerprint_does_not_depend_on_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, second_path = root / "first.gguf", root / "renamed.gguf"
            first_path.write_bytes(b"same model bytes")
            second_path.write_bytes(b"same model bytes")
            self.assertEqual(model_fingerprint(first_path), model_fingerprint(second_path))

    def test_report_round_trip_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = create_toy_model(root / "model")
            baseline = run_benchmark(model, "benchmark", tokens=4, runs=2)
            path = root / "baseline.json"
            save_report(baseline, path)
            loaded = load_report(path)
            self.assertEqual(loaded["model_sha256"], baseline.model_sha256)
            changes = compare_report(baseline, loaded)
            self.assertEqual(changes["prefill_percent"], 0.0)
            self.assertEqual(changes["decode_percent"], 0.0)
            self.assertEqual(changes["kv_cache_percent"], 0.0)
            json.dumps(baseline.to_dict())

    def test_comparison_rejects_different_generated_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            model = create_toy_model(Path(directory) / "model")
            report = run_benchmark(model, "benchmark", tokens=4, runs=1)
            baseline = report.to_dict()
            baseline["generated_token_ids"] = [999]
            with self.assertRaisesRegex(ValueError, "generated_token_ids"):
                compare_report(report, baseline)


if __name__ == "__main__":
    unittest.main()
