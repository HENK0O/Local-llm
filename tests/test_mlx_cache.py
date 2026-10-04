import unittest

from local_llm.mlx_cache import ConversationCache


class Layer:
    def __init__(self, tokens=(), trimmable=True):
        self.tokens = list(tokens)
        self.offset = len(tokens)
        self.trimmable = trimmable

    @property
    def nbytes(self):
        return 16 * len(self.tokens)

    def size(self):
        return self.offset if self.trimmable else 0

    def trim(self, count):
        removed = min(count, self.offset)
        self.offset -= removed
        return removed


def pool(budget=1024, entries=4, recurrent=False):
    return ConversationCache(lambda: [Layer(trimmable=not recurrent)],
        lambda layers: all(layer.trimmable for layer in layers),
        lambda layers, count: [layer.trim(count) for layer in layers][0], budget, entries)


class MLXCacheTests(unittest.TestCase):
    def test_reuses_only_identical_token_prefix_and_rolls_back_attention_state(self):
        cache = pool(); layer = Layer([1, 2, 3, 4])
        cache.store('chat', [1, 2, 3, 4], [layer])
        layers, reused = cache.prepare('chat', [1, 2, 3, 9, 10])
        self.assertEqual(reused, 3); self.assertIs(layers[0], layer)
        self.assertEqual(layer.offset, 3)
        self.assertEqual(cache.summary()['entries'], 0)

    def test_identical_or_shortened_prompts_always_leave_a_token_to_produce_logits(self):
        cache = pool()
        for prompt, expected in (([1, 2, 3, 4], 3), ([1, 2], 1)):
            cache.store('chat', [1, 2, 3, 4], [Layer([1, 2, 3, 4])])
            layers, reused = cache.prepare('chat', prompt)
            self.assertEqual(reused, expected); self.assertEqual(layers[0].offset, expected)

    def test_conversations_are_isolated_and_a_lease_is_not_reused_before_commit(self):
        cache = pool(); layer = Layer([1, 2])
        cache.store('a', [1, 2], [layer])
        cold, reused = cache.prepare('b', [1, 2, 3])
        self.assertEqual(reused, 0); self.assertIsNot(cold[0], layer)
        warm, reused = cache.prepare('a', [1, 2, 3])
        self.assertEqual(reused, 2); self.assertIs(warm[0], layer)
        _, reused = cache.prepare('a', [1, 2, 3])
        self.assertEqual(reused, 0)

    def test_recurrent_state_can_only_resume_an_entire_exact_saved_prefix(self):
        cache = pool(recurrent=True)
        for prompt, expected in (([1, 2, 3, 9], 3), ([1, 8, 3, 9], 0), ([1, 2, 3], 0)):
            layer = Layer([1, 2, 3], False)
            cache.store('chat', [1, 2, 3], [layer])
            layers, reused = cache.prepare('chat', prompt)
            self.assertEqual(reused, expected)
            self.assertEqual(layers[0] is layer, bool(expected))

    def test_recurrent_checkpoint_is_detached_before_decode_and_not_published_early(self):
        cache = pool(recurrent=True); layer = Layer([1, 2], False)
        snapshot = cache.checkpoint([layer], [1, 2])
        layer.tokens.append(99)
        self.assertEqual(snapshot.layers[0].tokens, [1, 2])
        self.assertEqual(cache.summary()['entries'], 0)
        cache.store('chat', snapshot.tokens, snapshot.layers)
        layers, reused = cache.prepare('chat', [1, 2, 4])
        self.assertEqual(reused, 2); self.assertNotIn(99, layers[0].tokens)

    def test_byte_budget_and_lru_bound_the_total_retained_state(self):
        cache = pool(budget=64, entries=2)
        cache.store('a', [1, 2], [Layer([1, 2])])
        cache.store('b', [3, 4], [Layer([3, 4])])
        layers, _ = cache.prepare('a', [1, 2, 5])
        cache.store('a', [1, 2], layers)
        cache.store('c', [6, 7], [Layer([6, 7])])
        self.assertEqual(list(cache.entries), ['a', 'c'])
        self.assertEqual(cache.nbytes, 64)
        self.assertFalse(cache.store('large', list(range(10)), [Layer(range(10))]))
        self.assertEqual(list(cache.entries), ['a', 'c'])
        self.assertIsNone(cache.checkpoint([Layer(range(10))], range(10)))

    def test_missing_byte_accounting_or_inconsistent_position_never_certifies_reuse(self):
        cache = pool(); layer = Layer([1, 2, 3]); layer.offset = 4
        cache.store('chat', [1, 2, 3], [layer])
        _, reused = cache.prepare('chat', [1, 2, 3, 5])
        self.assertEqual(reused, 0)
        self.assertFalse(cache.store('unknown', [1], [object()]))
        self.assertIsNone(cache.checkpoint([object()], [1]))

    def test_zero_budget_never_retains_state(self):
        cache = pool(0)
        self.assertFalse(cache.store('chat', [1], [Layer([1])]))
        self.assertIsNone(cache.checkpoint([Layer([1])], [1]))
        self.assertEqual(cache.nbytes, 0)

    def test_resizing_keeps_recent_entries_and_never_evicts_when_the_budget_is_unchanged(self):
        cache = pool(budget=128, entries=4)
        for name in ('a', 'b', 'c', 'd'):
            cache.store(name, [1, 2], [Layer([1, 2])])
        cache.resize(128)
        self.assertEqual(len(cache.entries), 4)
        cache.resize(32)
        self.assertEqual(list(cache.entries), ['d']); self.assertEqual(cache.nbytes, 32)
        cache.resize(0)
        self.assertEqual(cache.nbytes, 0); self.assertFalse(cache.entries)


if __name__ == '__main__':
    unittest.main()
