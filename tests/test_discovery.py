import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_llm.discovery import discover_models
from local_llm.server import ChatService
from local_llm.tokenizer import _bytes_to_unicode
from local_llm.toy import create_toy_model


def chat_toy(path, seed=42):
    create_toy_model(path, seed)
    vocab = {char: byte + 3 for byte, char in _bytes_to_unicode()[0].items()}
    vocab.update({'<pad>': 0, '<s>': 1, '</s>': 2})
    (path / 'tokenizer.json').write_text(json.dumps({
        'model': {'type': 'BPE', 'vocab': vocab, 'merges': []},
        'added_tokens': [{'id': i, 'content': token, 'special': True}
                         for token, i in [('<pad>', 0), ('<s>', 1), ('</s>', 2)]],
    }))
    (path / 'tokenizer_config.json').write_text(json.dumps({
        'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
        'chat_template': "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }))
    config = json.loads((path / 'config.json').read_text())
    config['eos_token_id'] = None
    (path / 'config.json').write_text(json.dumps(config))
    return path


class DiscoveryTests(unittest.TestCase):
    def test_discovers_deduplicated_models_and_explains_incompatibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chat_toy(root / 'good')
            (root / 'broken.gguf').write_bytes(b'not a gguf')
            (root / 'mmproj-F16.gguf').write_bytes(b'not a text model')
            models = discover_models([root, root])
            self.assertEqual(len(models), 2)
            self.assertTrue(models[0].compatible)
            self.assertIsNotNone(models[1].reason)
            self.assertEqual(len({m.id for m in models}), 2)

    def test_empty_and_missing_libraries(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(discover_models([Path(d), Path(d) / 'missing']), [])

    def test_scan_does_not_follow_directory_symlink_cycles(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chat_toy(root / 'good')
            (root / 'loop').symlink_to(root, target_is_directory=True)
            self.assertEqual(len(discover_models([root])), 1)

    def test_model_switch_is_transactional_and_invalidates_old_comparisons(self):
        with tempfile.TemporaryDirectory() as d, patch('local_llm.server.default_model_roots', return_value=[]):
            root = Path(d)
            first = chat_toy(root / 'first')
            chat_toy(root / 'second', seed=4)
            service = ChatService(first, model_dirs=[root])
            request = service.parse({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4})
            service.complete(request, 'old-response')
            old_id = service.current_id
            with self.assertRaises(ValueError):
                service.load_model({'id': '../../etc/passwd'})
            self.assertEqual(service.current_id, old_id)
            second = next(m for m in service.catalog.values() if m.name == 'second')
            service.load_model({'id': second.id})
            with self.assertRaisesRegex(ValueError, 'changé'):
                list(service.iter_completion(request))
            with self.assertRaisesRegex(ValueError, 'expirée'):
                service.compare_completion({'completion_id': 'old-response'})

    def test_failed_load_keeps_active_model(self):
        with tempfile.TemporaryDirectory() as d, patch('local_llm.server.default_model_roots', return_value=[]):
            root = Path(d)
            first = chat_toy(root / 'first')
            second = chat_toy(root / 'second')
            service = ChatService(first, model_dirs=[root])
            old_model = service.model
            second_id = next(m.id for m in service.catalog.values() if m.name == 'second')
            (second / 'weights.npz').write_bytes(b'corrupt weights')
            with self.assertRaises(ValueError):
                service.load_model({'id': second_id})
            self.assertIs(service.model, old_model)
            self.assertEqual(service.model_name, 'first')

    def test_service_can_start_without_a_model(self):
        with patch('local_llm.server.default_model_roots', return_value=[]):
            service = ChatService()
            self.assertFalse(service.info()['loaded'])
            with self.assertRaisesRegex(ValueError, 'Charge'):
                service.parse({'messages': [{'role': 'user', 'content': 'hi'}]})

    def test_comparison_uses_recorded_response_and_handles_one_token(self):
        with tempfile.TemporaryDirectory() as d, patch('local_llm.server.default_model_roots', return_value=[]):
            service = ChatService(chat_toy(Path(d) / 'toy'))
            request = service.parse({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4})
            service.complete(request, 'recorded')
            report = service.compare_completion({'completion_id': 'recorded'})
            self.assertTrue(report['same_weights'])
            self.assertTrue(report['comparable'])
            self.assertEqual(report['decode_steps'], 3)
            self.assertEqual(report['runs'], 3)
            json.dumps(report, allow_nan=False)
            service.complete(service.parse({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1}), 'single')
            with self.assertRaisesRegex(ValueError, 'peu de tokens'):
                service.compare_completion({'completion_id': 'single'})
