import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_llm.kv_store import KVStore


class SlotClient:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def _request(self, route, payload, timeout):
        self.calls.append(route)
        if route.endswith('save'):
            (self.root / payload['filename']).write_bytes(b'private binary state' * 20)
            return {'n_saved': 80}
        if route.endswith('restore'):
            return {'n_restored': 80}
        return {'n_erased': 80}


class PersistentCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = KVStore(self.folder.name)
        self.store.prepare()
        self.client = SlotClient(self.store.root)
        self.messages = [{'role': 'user', 'content': 'PRIVATE USER TEXT'}]

    def save(self, conversation='a', binding='weights-runtime-config', ms=1000):
        self.store.save(self.client, 0, conversation, binding, self.messages, {'prompt_ms': ms, 'prompt_n': 40})

    def test_survives_restart_and_requires_same_model_runtime_config_and_history(self):
        self.save()
        restarted = KVStore(self.folder.name)
        self.assertTrue(restarted.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages))
        self.assertTrue(restarted.last['restored'])
        self.assertFalse(restarted.restore(self.client, 0, 'b', 'weights-runtime-config', self.messages))
        self.assertFalse(restarted.restore(self.client, 0, 'a', 'other-config', self.messages))
        changed = [{'role': 'user', 'content': 'edited'}]
        self.assertFalse(restarted.restore(self.client, 0, 'a', 'weights-runtime-config', changed))
        self.assertTrue(restarted.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages + [{'role': 'assistant', 'content': 'answer'}]))

    def test_corrupt_binary_erases_partial_slot_and_falls_back_without_reuse(self):
        self.save()
        binary = next(self.store.root.glob('*.bin'))
        binary.write_bytes(b'x' * binary.stat().st_size)
        self.assertFalse(self.store.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages))
        self.assertEqual(self.client.calls[-1], '/slots/0?action=erase')
        self.assertEqual(self.store.describe()['entries'], 0)

    def test_incomplete_runtime_restore_is_erased(self):
        self.save()
        client = Mock()
        client._request.side_effect = [{'n_restored': 3}, {'n_erased': 3}]
        self.assertFalse(self.store.restore(client, 0, 'a', 'weights-runtime-config', self.messages))
        self.assertTrue(client._request.call_args.args[0].endswith('erase'))

    def test_recalculation_cost_gate_prevents_slow_restore(self):
        self.save(ms=.001)
        calls = len(self.client.calls)
        self.assertFalse(self.store.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages))
        self.assertEqual(len(self.client.calls), calls)
        self.assertIn('moins coûteux', self.store.last['reason'])

    def test_observed_restore_cost_survives_next_save_and_controls_future_decision(self):
        self.save()
        self.assertTrue(self.store.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages))
        meta = next(self.store.root.glob('*.json'))
        value = json.loads(meta.read_text())
        value.update(restore_seconds=2.0, restore_measured=True)
        meta.write_text(json.dumps(value))
        self.save()
        self.assertEqual(json.loads(meta.read_text())['restore_seconds'], 2.0)
        self.assertFalse(self.store.restore(self.client, 0, 'a', 'weights-runtime-config', self.messages))

    def test_warm_repeated_save_keeps_cold_prefill_cost(self):
        self.save(ms=1000)
        self.store.save(self.client, 0, 'a', 'weights-runtime-config', self.messages, {'cache_n': 40, 'prompt_ms': .01})
        meta = json.loads(next(self.store.root.glob('*.json')).read_text())
        self.assertEqual(meta['prefill_ms'], 1000)

    def test_budget_evicts_oldest_snapshot_and_oversized_file_is_not_retained(self):
        self.store.budget = 600
        self.save('a'); self.save('b')
        self.assertEqual(self.store.describe()['entries'], 1)
        self.assertFalse((self.store.root / (self.store.key('a', 'weights-runtime-config') + '.bin')).exists())
        self.store.budget = 10
        self.save('c')
        self.store.prune()
        self.assertEqual(self.store.describe()['bytes'], 0)
        self.assertFalse(list(self.store.root.glob('*.tmp')))

    def test_disable_is_persisted_clears_only_owned_snapshots_and_blocks_new_saves(self):
        self.save()
        unrelated = self.store.root / 'unrelated.txt'
        unrelated.write_text('keep')
        self.store.configure(False)
        self.assertEqual(self.store.describe()['entries'], 0)
        self.assertFalse(KVStore(self.folder.name).enabled)
        calls = len(self.client.calls)
        self.save('b')
        self.assertEqual(len(self.client.calls), calls)
        self.assertTrue(unrelated.exists())

    def test_metadata_has_no_conversation_text_or_identifier_and_permissions_are_private(self):
        self.save(conversation='secret-conversation-id')
        metadata = next(self.store.root.glob('*.json'))
        self.assertNotIn('PRIVATE USER TEXT', metadata.read_text())
        self.assertNotIn('secret-conversation-id', metadata.read_text())
        self.assertEqual(os.stat(metadata).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(next(self.store.root.glob('*.bin'))).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.store.root).st_mode & 0o777, 0o700)

    def test_orphan_cleanup_and_symlink_root_rejection(self):
        orphan = self.store.root / ('f' * 64 + '.bin')
        orphan.write_bytes(b'partial')
        self.store.prepare()
        self.assertFalse(orphan.exists())
        target = Path(self.folder.name) / 'other'; target.mkdir()
        link = Path(self.folder.name) / 'linked'; link.mkdir()
        (link / 'kv-cache').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symbolique'):
            KVStore(link).prepare()

    def test_loaded_weight_identity_allows_snapshots_until_file_changes(self):
        from local_llm.accelerator import Accelerator
        path = Path(self.folder.name) / 'weights.gguf'; path.write_bytes(b'weights')
        runtime = Accelerator(executable='mock', state_dir=self.folder.name)
        runtime.path, runtime.cache_binding, runtime._cache_stat = path, 'binding', path.stat()
        self.assertTrue(runtime._cache_compatible())
        path.write_bytes(b'changed weights')
        self.assertFalse(runtime._cache_compatible())
