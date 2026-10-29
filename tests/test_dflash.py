import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_llm.dflash import (AdaptiveDraft, DFlashSession, capability, draft_contract,
                              load_local_draft, UPSTREAM_SHA256)
from local_llm.discovery import discover_models
from local_llm.mac_runtime import MacConfig, MacRuntime, mac_memory_plan, model_engines
from local_llm.mac_worker import Worker


def pair(root, name='Qwen3-4B', method='DFlash'):
    target, draft = Path(root).resolve() / name, Path(root).resolve() / (name + '-' + method)
    target.mkdir(); draft.mkdir()
    text = dict(model_type='qwen3', hidden_size=64, vocab_size=128, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=8192)
    aux = dict(text, architectures=[method + 'DraftModel'], num_target_layers=2,
               dflash_config={'target_layer_ids': [0, 1], 'mask_token_id': 127})
    for path, config in ((target, text), (draft, aux)):
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'model.safetensors').write_bytes(b'not a model; unit tests never load this')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (target / name).write_text('{}')
    return target, draft


class DFlashTests(unittest.TestCase):
    def test_discovery_keeps_drafts_without_tokenizers_but_never_as_chat_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder)
            items = discover_models([Path(folder)])
            self.assertEqual({m.path for m in items}, {str(target), str(draft)})
            self.assertEqual(model_engines(draft, {'available': True}), [])
            self.assertEqual(draft_contract(target, draft)['method'], 'dflash')

    def test_contract_refuses_finetunes_other_generations_and_bad_shapes_or_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder)
            original = json.loads((draft / 'config.json').read_text())
            for change in ({'hidden_size': 128}, {'vocab_size': 129}, {'num_target_layers': 3},
                           {'dflash_config': {'target_layer_ids': [0, 2], 'mask_token_id': 127}},
                           {'dflash_config': {'target_layer_ids': [True], 'mask_token_id': 127}},
                           {'dflash_config': {'target_layer_ids': [0, 1], 'mask_token_id': 128}},
                           {'quantization': {'bits': 4}}):
                (draft / 'config.json').write_text(json.dumps(dict(original, **change)))
                with self.assertRaises(ValueError):
                    draft_contract(target, draft)
            (draft / 'config.json').write_text(json.dumps(original))
            for name in ('Qwen3-4B-Heretic', 'Qwen3.5-4B', 'Ling-3.0-tiny'):
                renamed = target.rename(Path(folder) / name)
                with self.assertRaises(ValueError):
                    draft_contract(renamed, draft)
                renamed.rename(target)
            # auto_map is inert metadata: only built-in DFlash classes load.
            (draft / 'config.json').write_text(json.dumps(dict(original, auto_map={'AutoModel': 'untrusted.Code'})))
            self.assertTrue(draft_contract(target, draft)['candidate'])

    def test_dflash2_is_a_separate_contract_and_quantized_target_names_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder, 'Qwen3.8-27B', 'DFlash2')
            root = json.loads((target / 'config.json').read_text())
            (target / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'text_config': root}))
            target = target.rename(target.with_name(target.name + '-4bit'))
            self.assertEqual(draft_contract(target, draft)['method'], 'dflash2')

    def test_local_resolver_never_calls_hub_and_restores_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Mock(side_effect=AssertionError('network forbidden'))
            module = SimpleNamespace(snapshot_download=original)
            def load(path):
                self.assertEqual(module.snapshot_download(path), str(Path(folder).resolve()))
                with self.assertRaises(ValueError):
                    module.snapshot_download('z-lab/some-other-model')
                return 'loaded'
            module.load_draft = load
            self.assertEqual(load_local_draft(module, folder), 'loaded')
            self.assertIs(module.snapshot_download, original)
            module.load_draft = Mock(side_effect=ValueError('bad weights'))
            with self.assertRaises(ValueError):
                load_local_draft(module, folder)
            self.assertIs(module.snapshot_download, original)
            original.assert_not_called()

    def test_unknown_upstream_source_is_refused_without_importing_mlx(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'model_mlx.py'; source.write_text('not the audited API')
            with patch('local_llm.dflash.importlib.util.find_spec', return_value=SimpleNamespace(origin=str(Path(folder) / '__init__.py'))):
                info = capability()
                self.assertFalse(info['available'])
                self.assertNotEqual(info['source_sha256'], UPSTREAM_SHA256)

    def test_memory_plan_covers_auxiliary_load_peak_kv_and_hidden_states(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder)
            base = mac_memory_plan(target, 8 << 30, 24 << 30)
            spec = mac_memory_plan(target, 8 << 30, 24 << 30, draft=draft)
            self.assertGreater(spec['draft_budget_bytes'], 3 * (draft / 'model.safetensors').stat().st_size)
            self.assertGreater(spec['estimated_bytes'], base['estimated_bytes'])
            with self.assertRaisesRegex(ValueError, 'RAM disponible insuffisante'):
                mac_memory_plan(target, base['estimated_bytes'] - base['conversation_cache_budget_bytes'], 24 << 30, draft=draft)

    def test_bad_auxiliary_is_rejected_before_dropping_working_model(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder)
            runtime = MacRuntime(folder); runtime.path = target
            runtime.process = Mock(); runtime.process.poll.return_value = None
            runtime.memory_probe = lambda: 8 << 30
            config = MacConfig(draft_path=str(draft), speculative='dflash')
            with patch('local_llm.mac_runtime.macos_memory_pressure', return_value='normal'), patch(
                    'local_llm.mac_runtime.detect_hardware', return_value={'memory_bytes': 24 << 30}), patch.object(
                    runtime, 'available', return_value={'dflash': {'available': False}}), patch.object(runtime, '_stop') as stop:
                with self.assertRaisesRegex(ValueError, 'DFlash'):
                    runtime._start(config)
                stop.assert_not_called()

    def test_candidate_search_is_bounded_and_pairs_only_local_compatible_drafts(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = pair(folder)
            runtime = MacRuntime(folder); runtime.path = target
            runtime.draft_paths = runtime.compatible_drafts([target, draft, draft, '/missing'])
            self.assertEqual(runtime.draft_paths, [str(draft)])
            with patch.object(runtime, 'available', return_value={'available': True, 'dflash': {'available': True}}):
                configs = runtime._candidate_configs(4096)
            self.assertEqual({c.block_size for c in configs.values() if c.draft_path}, {2, 5})
            with patch.object(runtime, 'available', return_value={'available': True}):
                self.assertFalse(any(c.draft_path for c in runtime._candidate_configs(4096).values()))

    def test_separate_dflash_environment_does_not_hide_installed_mtplx(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ('mtplx-python', 'dflash-python')]
            for path in paths: path.touch()
            metadata = [{'available': True, 'packages': {'mlx': 'old', 'mlx-lm': 'old', 'mtplx': '2.12.2'}},
                        {'available': True, 'packages': {'mlx': '0.32.0', 'mlx-lm': '0.31.3'}, 'dflash': {'available': True}}]
            results = [SimpleNamespace(returncode=0, stdout=json.dumps(m)) for m in metadata]
            runtime = MacRuntime(folder)
            with patch('local_llm.mac_runtime.platform.system', return_value='Darwin'), patch(
                    'local_llm.mac_runtime.platform.machine', return_value='arm64'), patch(
                    'local_llm.mac_runtime.interpreter_candidates', return_value=paths), patch(
                    'local_llm.mac_runtime.subprocess.run', side_effect=results):
                info = runtime.available()
            self.assertEqual(info['interpreter'], str(paths[1]))
            self.assertEqual(info['mtplx_runtime']['interpreter'], str(paths[0]))
            target, _ = pair(folder)
            (target / 'mtp.safetensors').write_bytes(b'head')
            self.assertIn('mtplx', model_engines(target, info))

    def test_config_refuses_mtp_dflash_mixing_relative_paths_and_invalid_blocks(self):
        for values in ({'draft_path': 'local', 'speculative': 'dflash'}, {'speculative': 'dflash'},
                       {'draft_path': '/local', 'speculative': 'none'},
                       {'draft_path': '/local', 'speculative': 'dflash', 'engine': 'mtplx'},
                       {'draft_path': '/local', 'speculative': 'dflash', 'block_size': 16}):
            with self.assertRaises(ValueError):
                MacConfig(**values)

    def test_adaptation_backs_off_low_progress_then_retries_without_faking_acceptance(self):
        policy = AdaptiveDraft()
        self.assertEqual(policy.choose(16)[0], 0)
        self.assertEqual(policy.choose(128)[0], 3)
        key = policy.bucket(100, 128, 0)
        policy.observe(key, 3, 20, 20, 21, 1)
        self.assertEqual([policy.choose(128)[0] for _ in range(4)], [0, 0, 0, 2])
        policy.observe(key, 3, 10, 30, 31, 1)
        self.assertEqual(policy.choose(128)[0], 5)

    def test_adaptation_compares_only_matching_buckets_and_bounds_memory(self):
        policy = AdaptiveDraft(); key = policy.bucket(100, 128, 0)
        for _ in range(2):
            policy.observe(key, 0, 0, 0, 100, 1)
            policy.observe(key, 3, 20, 40, 100, 2)
        self.assertEqual(policy.choose(128)[0], 0)
        for i in range(100):
            policy.observe((i, 0, False), 0, 0, 0, 20, 1)
        self.assertLessEqual(len(policy.rates), 32)

    def test_stream_restores_hooks_and_hidden_activations_on_cancellation(self):
        session = DFlashSession.__new__(DFlashSession)
        session.model = SimpleNamespace(); session.draft = object(); session.tokenizer = object()
        originals = [object(), object()]; layers = list(originals)
        closed = []
        def generate(*args, **kwargs):
            layers[0] = 'hook'; session.model._hidden_states = ['large activations']
            try:
                yield 'first block'
                yield 'second block'
            finally:
                closed.append(True)
        session.module = SimpleNamespace(_get_layers=lambda model: layers, stream_generate=generate)
        stream = session.stream([1, 2], max_tokens=10)
        self.assertEqual(next(stream), 'first block')
        stream.close()
        self.assertEqual(layers, originals)
        self.assertFalse(hasattr(session.model, '_hidden_states'))
        self.assertEqual(closed, [True])

    def test_worker_counts_verified_blocks_eos_and_final_tail_exactly_once(self):
        worker = Worker.__new__(Worker)
        worker.engine = 'mlx'; worker.context_length = 4096
        worker.mx = Mock(); worker.mx.get_peak_memory.return_value = 1234
        worker.tokenizer = SimpleNamespace(eos_token_ids={0})
        class Detokenizer:
            def __init__(self): self.pending = ''
            def add_token(self, token): self.pending += {4: 'hel', 5: 'l', 6: 'o', 7: '!'}[token]
            def finalize(self): pass
            @property
            def last_segment(self):
                value, self.pending = self.pending, ''
                return value
        worker.tokenizer.detokenizer = Detokenizer()
        worker.context = lambda *args: ({'prompt': 'question', 'prompt_sha256': 'input'}, [1, 2])
        def row(text, tokens, count, accepted=None, finish=None):
            return SimpleNamespace(text=text, tokens=tokens, generation_tokens=count, accepted=accepted,
                                   prompt_tokens=2, prompt_tps=20, generation_tps=10, finish_reason=finish)
        rows = [row('hel', [4], 1), row('lo', [5, 6], 3, 2), row('!<eos>', [7, 0], 5, 2, 'stop')]
        def stream(*args, **kwargs):
            yield from rows
        worker.dflash = SimpleNamespace(contract={'method': 'dflash2'}, adaptive=AdaptiveDraft(),
                                       stream=stream, last=None)
        frames = []
        worker.generate({'messages': [], 'max_tokens': 64}, frames.append)
        final = frames[-1]
        self.assertEqual(final['usage'], {'prompt_tokens': 2, 'completion_tokens': 4})
        self.assertEqual(final['sample']['output_sha256'], hashlib.sha256(json.dumps([4, 5, 6, 7]).encode()).hexdigest())
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in frames), 'hello!')
        self.assertEqual(final['speculation']['last_request']['tokens_per_cycle'], 1.5)
        self.assertEqual(final['timings']['cache_n'], 0)
        self.assertNotIn('acceptance_rate', final['speculation']['last_request'])
        rows[-1] = row('!', [7, 0], 6, 2, 'stop')
        with self.assertRaisesRegex(ValueError, 'Comptage'):
            worker.generate({'messages': [], 'max_tokens': 64}, lambda frame: None)

    def test_profile_gain_is_not_attached_to_unvalidated_online_blocks_or_ar_fallback(self):
        runtime = MacRuntime('/tmp/unused')
        runtime.config = MacConfig(draft_path='/local/draft', speculative='dflash', block_size=5)
        profile = {'winner': 'dflash-0-b5', 'gain_percent': 12}
        runtime.profile = {'profiles': {'balanced': profile}}
        for method, block in (('dflash', 3), ('autoregressive', 0)):
            runtime.speculation_state = {'last_request': {'method': method, 'block_size': block}}
            self.assertIsNone(runtime.active_measurement())
        runtime.speculation_state = {'last_request': {'method': 'dflash', 'block_size': 5}}
        self.assertEqual(runtime.active_measurement(), profile)


if __name__ == '__main__':
    unittest.main()
