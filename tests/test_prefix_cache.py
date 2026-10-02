import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm.cache import PrefixCache
from local_llm.generation import generate, generate_tokens
from local_llm.server import ChatService
from tests.test_discovery import chat_toy
from tests.test_model import tiny_model, tiny_gated_model


def run(model, prompt, cache, count=4):
    stream = list(generate_tokens(model, prompt, count, prefix_cache=cache))
    return [token for token, _ in stream], stream[-1][1]


class PrefixCacheTests(unittest.TestCase):
    def test_continuation_preserves_greedy_output_and_skips_actual_prompt_work(self):
        for factory in (tiny_model, tiny_gated_model):
            model = factory()
            cache = PrefixCache()
            prompt = [1, 3, 5, 7]
            emitted, cold = run(model, prompt, cache)
            self.assertEqual(cold.reused_prompt_tokens, 0)
            continuation = prompt + emitted + [9, 11]
            expected = generate(model, continuation, 4).token_ids
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                actual, warm = run(model, continuation, cache)
            self.assertEqual(actual, expected)
            self.assertEqual(warm.reused_prompt_tokens, len(prompt) + len(emitted) - 1)
            self.assertEqual(len(forward.call_args_list[0].args[0]),
                             len(continuation) - warm.reused_prompt_tokens)
            self.assertLess(warm.reused_prompt_tokens, warm.prompt_tokens)

    def test_changed_and_identical_prompts_use_only_the_exact_prefix(self):
        model = tiny_model()
        cache = PrefixCache()
        prompt = [1, 3, 5, 7]
        original, _ = run(model, prompt, cache)
        identical, stats = run(model, prompt, cache)
        self.assertEqual(identical, original)
        self.assertEqual(stats.reused_prompt_tokens, len(prompt) - 1)
        changed = [1, 3, 8, 9]
        actual, stats = run(model, changed, cache)
        self.assertEqual(stats.reused_prompt_tokens, 2)
        self.assertEqual(actual, generate(model, changed, 4).token_ids)

    def test_interruption_does_not_leave_a_partially_overwritten_cache(self):
        model = tiny_model()
        cache = PrefixCache()
        prompt = [1, 3, 5, 7]
        run(model, prompt, cache)
        stream = generate_tokens(model, [1, 3, 9, 8], 4, prefix_cache=cache)
        next(stream)
        stream.close()
        actual, stats = run(model, prompt, cache)
        self.assertEqual(stats.reused_prompt_tokens, 0)
        self.assertEqual(actual, generate(model, prompt, 4).token_ids)

    def test_memory_budget_and_model_identity_prevent_reuse(self):
        model = tiny_model()
        bounded = PrefixCache(max_bytes=1)
        run(model, [1, 3, 5], bounded)
        self.assertEqual(bounded.nbytes, 0)
        _, stats = run(model, [1, 3, 5], bounded)
        self.assertEqual(stats.reused_prompt_tokens, 0)
        cache = PrefixCache()
        run(model, [1, 3, 5], cache)
        _, stats = run(tiny_model(), [1, 3, 5], cache)
        self.assertEqual(stats.reused_prompt_tokens, 0)

    def test_service_keeps_model_loaded_and_unload_releases_model_and_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch('local_llm.server.default_model_roots', return_value=[]):
            service = ChatService(chat_toy(Path(directory) / 'toy'))
            request = service.parse({'messages': [{'role': 'user', 'content': 'bonjour'}], 'max_tokens': 4})
            model = service.model
            first = service.complete(request)
            second = service.complete(request)
            self.assertIs(service.model, model)
            self.assertEqual(first.token_ids, second.token_ids)
            self.assertGreater(second.stats.reused_prompt_tokens, 0)
            self.assertGreater(service.info()['retained_cache_bytes'], 0)
            service.unload_model()
            self.assertIsNone(service.model)
            self.assertIsNone(service.tokenizer)
            self.assertEqual(service.info()['retained_cache_bytes'], 0)
            with self.assertRaisesRegex(ValueError, 'Charge'):
                service.parse({'messages': [{'role': 'user', 'content': 'bonjour'}]})
