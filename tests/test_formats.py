import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from local_llm.discovery import inspect_model
from local_llm.formats import directory_format, model_availability, quantization_label
from local_llm.mlx_experiment import validate_model


class FormatTests(unittest.TestCase):
    def test_quantized_container_is_not_mistaken_for_mlx(self):
        for field in ('quant_method', 'checkpoint_format', 'format', 'method'):
            config = {'quantization_config': {field: 'exl3', 'avg_bits': 3.5}}
            self.assertEqual(directory_format(config), 'EXL3')
            self.assertEqual(quantization_label(config, 'EXL3'), 'EXL3 · 3.5 bpw')
            nested = {'text_config': config}
            self.assertEqual(directory_format(nested), 'EXL3')
            self.assertEqual(quantization_label(nested, 'EXL3'), 'EXL3 · 3.5 bpw')
        self.assertEqual(directory_format({'name_or_path': 'EXL3-old-name'}), 'Safetensors')
        self.assertEqual(directory_format({'quantization': {'bits': 4, 'group_size': 64}}), 'Safetensors')

    def test_exl3_discovery_and_direct_loading_explain_missing_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'quantization_config': {'quant_method': 'exl3'}}))
            (root / 'model.safetensors').write_bytes(b'fixture')
            for name in ('tokenizer.json', 'tokenizer_config.json'):
                (root / name).write_text('{}')
            item = inspect_model(root)
            self.assertEqual(item.format, 'EXL3')
            self.assertFalse(item.compatible)
            self.assertIn('MLXL3', item.reason)
            with self.assertRaisesRegex(ValueError, 'runtime dédié'):
                validate_model(root)
            # An upstream engine's false positive must never make this loadable.
            self.assertEqual(model_availability(item, ['mlx'], {})['code'], 'unsupported_format')

    def test_availability_distinguishes_runtime_auxiliary_and_candidate(self):
        item = SimpleNamespace(format='GGUF', role='model', compatible=False, reason='Architecture non prise en charge')
        self.assertEqual(model_availability(item, [], {})['code'], 'missing_runtime')
        self.assertEqual(model_availability(item, [], {'llamacpp': {'available': True}})['code'], 'unavailable')
        self.assertEqual(model_availability(item, ['llamacpp'], {})['code'], 'candidate')
        item.role = 'auxiliary'
        self.assertEqual(model_availability(item, ['llamacpp'], {})['code'], 'auxiliary')
