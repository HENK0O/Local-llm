import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm.server import ChatService
from tests.test_discovery import chat_toy


class ContextInspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch('local_llm.server.default_model_roots', return_value=[]):
            self.service = ChatService(chat_toy(Path(self.directory.name) / 'toy'))
        self.addCleanup(self.service.telemetry.close)
        # An embedded template can add instructions absent from the UI messages.
        self.service.tokenizer.chat_template = (
            "SYSTEM: Remember the conversation.\n"
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
        self.payload = {'messages': [{'role': 'user', 'content': 'bonjour'}],
                        'max_tokens': 2}

    def test_preview_matches_inference_and_does_not_generate_or_change_cache(self):
        request = self.service.parse(self.payload)
        result = self.service.complete(request, 'first')
        retained_cache = self.service.prefix_cache.nbytes
        cache = self.service.prefix_cache.cache
        tokens = list(self.service.prefix_cache.tokens)
        with patch.object(self.service.model, 'forward', side_effect=AssertionError('inference')):
            snapshot = self.service.context_snapshot(self.payload)
        self.assertEqual(snapshot['prompt'], self.service._records['first']['prompt'])
        self.assertIn('SYSTEM: Remember the conversation.', snapshot['prompt'])
        self.assertEqual(snapshot['prompt_tokens'], result.prompt_tokens)
        self.assertEqual(snapshot['compression'], 'none')
        self.assertEqual(self.service.prefix_cache.nbytes, retained_cache)
        self.assertIs(self.service.prefix_cache.cache, cache)
        self.assertEqual(self.service.prefix_cache.tokens, tokens)
        self.assertEqual(len(self.service._records), 1)

    def test_next_context_includes_reply_without_overwriting_previous_request(self):
        first = self.service.complete(self.service.parse(self.payload), 'first')
        original = self.service.context_snapshot({'completion_id': 'first'})
        continuation = dict(self.payload, messages=self.payload['messages'] + [
            {'role': 'assistant', 'content': first.text},
            {'role': 'user', 'content': 'Et ensuite ?'},
        ])
        preview = self.service.context_snapshot(continuation)
        second = self.service.complete(self.service.parse(continuation), 'second')
        actual = self.service.context_snapshot({'completion_id': 'second'})
        self.assertEqual(preview['prompt'], actual['prompt'])
        self.assertEqual(preview['prompt_tokens'], second.prompt_tokens)
        self.assertIn('Et ensuite ?', actual['prompt'])
        self.assertEqual(self.service.context_snapshot({'completion_id': 'first'}), original)

    def test_last_request_is_the_captured_prompt_not_a_new_render(self):
        self.service.complete(self.service.parse(self.payload), 'first')
        original = self.service._records['first']['prompt']
        self.service.tokenizer.chat_template = 'Changed instructions'
        snapshot = self.service.context_snapshot({'completion_id': 'first'})
        self.assertEqual(snapshot['kind'], 'request')
        self.assertEqual(snapshot['prompt'], original)
        self.assertEqual(snapshot['prompt_tokens'], len(self.service.tokenizer.encode(original)))

    def test_missing_expired_unloaded_and_external_contexts_are_not_fabricated(self):
        self.service.complete(self.service.parse(self.payload), 'first')
        self.service._records['first']['created'] -= 901
        for payload in ({'completion_id': 'missing'}, {'completion_id': 'first'},
                        {'completion_id': []}, {'backend': 'lmstudio'},
                        dict(self.payload, model='wrong-model'), None):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.context_snapshot(payload)
        self.service.unload_model()
        with self.assertRaises(ValueError):
            self.service.context_snapshot(self.payload)

    def test_inspection_does_not_queue_behind_a_generation(self):
        held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with self.service._generation_lock:
                held.set()
                release.wait(2)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            self.assertTrue(held.wait(2))
            with self.assertRaisesRegex(ValueError, 'Génération en cours'):
                self.service.context_snapshot(self.payload)
        finally:
            release.set()
            thread.join(2)
