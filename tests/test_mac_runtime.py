import copy
import hashlib
import json
import queue
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_llm.calibration import TRAIN_PASSES, VALIDATION_PASSES, CATEGORIES, OUTPUT_LIMITS, verified_profiles, summarize
from local_llm.discovery import inspect_model
from local_llm.engines import EngineManager
from local_llm.mac_runtime import MacConfig, MacRuntime, mac_memory_plan, model_engines, worker_environment
from local_llm.mac_worker import ReasoningSplit, Worker
from local_llm.mlx_experiment import validate_model


def fixture(root, architecture='qwen3_5', mtp=True):
    config = {'model_type': architecture, 'num_hidden_layers': 2, 'num_key_value_heads': 1,
              'num_attention_heads': 2, 'hidden_size': 64, 'max_position_embeddings': 8192,
              'quantization': {'bits': 6, 'group_size': 64}}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (root / name).write_text(json.dumps(config if name == 'config.json' else {}))
    (root / 'model.safetensors').write_bytes(b'fixture')
    if mtp:
        (root / 'mtp.safetensors').write_bytes(b'head')
    return root


class MacRuntimeTests(unittest.TestCase):
    def test_discovery_retains_mlx_size_quantization_and_exact_library_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fixture(Path(folder))
            item = inspect_model(root, 'MTPLX')
            self.assertEqual(item.path, str(root))
            self.assertEqual(item.size_bytes, 11)
            self.assertEqual(item.quantization, '6-bit MLX')
            self.assertFalse(item.compatible)  # Native CPU capability is separate.
            self.assertEqual(model_engines(root, {'available': True, 'packages': {'mtplx': '2.12.2'}}), ['mlx', 'mtplx'])
            self.assertEqual(model_engines(root, {'available': True}), ['mlx'])
            self.assertEqual(model_engines(root, {'available': False}), [])
            config = json.loads((root / 'config.json').read_text())
            config['model_type'] = 'prism_hadamard_qwen35'
            (root / 'config.json').write_text(json.dumps(config))
            self.assertEqual(model_engines(root, {'available': True, 'packages': {'mtplx': '2.12.2'}}), ['mtplx'])

    def test_checkpoint_supplied_python_is_never_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fixture(Path(folder))
            for config in ({'model_file': 'custom.py'}, {'auto_map': {'AutoModel': 'custom.Model'}}, []):
                (root / 'config.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    validate_model(root)
                self.assertEqual(model_engines(root, {'available': True}), [])

    def test_memory_estimate_requires_known_free_ram_and_reserves_the_os(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fixture(Path(folder))
            for available, total in ((None, 24 << 30), (10 << 30, None), (128 << 20, 24 << 30), (1 << 30, 2 << 30)):
                with self.assertRaises(ValueError):
                    mac_memory_plan(root, available, total)
            result = mac_memory_plan(root, 10 << 30, 24 << 30)
            self.assertEqual(result['context'], 4096)
            self.assertEqual(result['memory_limit_bytes'], 10 << 30)
            self.assertEqual(result['kind'], 'estimate')

    def test_configs_reject_unmeasured_depths_and_inconsistent_methods(self):
        for config in ({'engine': 'shell'}, {'depth': 1}, {'depth': True}, {'context': 0}, {'prefill_step_size': 1},
                       {'engine': 'mtplx', 'depth': 4, 'speculative': 'draft-mtp'}, {'speculative': 'draft-mtp'}):
            with self.assertRaises(ValueError):
                MacConfig(**config)
        self.assertEqual(MacConfig(engine='mtplx', depth=2, speculative='draft-mtp').depth, 2)

    def test_mtplx_loading_starts_without_unvalidated_speculation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fixture(Path(folder)); runtime = MacRuntime(folder)
            with patch.object(runtime, 'available', return_value={'available': True, 'packages': {'mtplx': '2.12.2'}}), patch(
                    'local_llm.mac_runtime.detect_hardware', return_value={'memory_bytes': 24 << 30}), patch.object(runtime, '_start') as start:
                runtime.load(inspect_model(root), 10 << 30, 'mtplx')
            self.assertEqual(start.call_args.args[0].engine, 'mtplx')
            self.assertEqual(start.call_args.args[0].depth, 0)
            self.assertEqual(start.call_args.args[0].speculative, 'none')

    def test_memory_budget_is_refreshed_before_starting_each_owned_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = MacRuntime(folder); runtime.path = fixture(Path(folder))
            runtime.memory_probe = lambda: 128 << 20
            with patch('local_llm.mac_runtime.macos_memory_pressure', return_value='normal'), patch(
                    'local_llm.mac_runtime.detect_hardware', return_value={'memory_bytes': 24 << 30}), patch(
                    'local_llm.mac_runtime.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'RAM disponible insuffisante'):
                    runtime._start(MacConfig())
                popen.assert_not_called()

    def test_child_environment_is_offline_and_does_not_inherit_runtime_overrides_or_lm_credentials(self):
        with patch.dict('os.environ', {'LM_STUDIO_API_TOKEN': 'secret', 'MTPLX_USE_UNVERIFIED': '1', 'HF_TOKEN': 'secret', 'MLX_OPT': 'x'}):
            env = worker_environment()
        for key in ('LM_STUDIO_API_TOKEN', 'MTPLX_USE_UNVERIFIED', 'HF_TOKEN', 'MLX_OPT'):
            self.assertNotIn(key, env)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertIn('local-llm-mac-bytecode', env['PYTHONPYCACHEPREFIX'])

    def test_worker_capability_is_probed_once_and_explicit_python_does_not_install_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / 'python'; python.write_text('fixture')
            runtime = MacRuntime(folder)
            with patch('local_llm.mac_runtime.platform.system', return_value='Darwin'), patch('local_llm.mac_runtime.platform.machine', return_value='arm64'), patch('local_llm.mac_runtime.interpreter_candidates', return_value=[python]), patch('local_llm.mac_runtime.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='{"available":true,"packages":{"mlx":"0.32"}}')) as run:
                self.assertTrue(runtime.available()['available'])
                runtime.available()
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], [str(python), '-m', 'local_llm.mac_worker', '--probe'])
                self.assertNotIn('shell', run.call_args.kwargs)

    def test_closing_partial_stream_stops_only_owned_worker_and_discards_late_frames(self):
        runtime = MacRuntime('/tmp/unused')
        process = Mock(); process.poll.return_value = None
        runtime.process = process
        runtime.events.put({'event': 'chunk', 'choices': [{'delta': {'content': 'partial'}}]})
        with patch.object(runtime, '_stop') as stop:
            stream = runtime._request({'op': 'generate'})
            next(stream)
            stream.close()
            stop.assert_called_once()
        runtime.process = None
        with self.assertRaisesRegex(ValueError, 'plus chargé'):
            list(runtime._request({'op': 'context'}))

    def test_completed_context_request_keeps_weights_resident(self):
        runtime = MacRuntime('/tmp/unused'); runtime.model_name = 'Selected'
        runtime.process = Mock(); runtime.process.poll.return_value = None
        runtime.events.put({'event': 'done', 'context': {'prompt': 'exact', 'prompt_tokens': 2}})
        with patch.object(runtime, '_stop') as stop:
            self.assertEqual(runtime.context([])['prompt'], 'exact')
            stop.assert_not_called()

    def test_normalized_stream_frames_do_not_corrupt_the_ipc_completion_marker(self):
        runtime = MacRuntime('/tmp/unused'); runtime.model_id = 'selected'
        runtime.process = Mock(); runtime.process.poll.return_value = None
        runtime.events.put({'event': 'chunk', 'choices': [{'delta': {'content': 'Bonjour'}}]})
        runtime.events.put({'event': 'done', 'usage': {'completion_tokens': 1}, 'sample': {'seconds': 1}})
        with patch.object(runtime, '_stop') as stop:
            frames = list(runtime.iter_chat({'model': 'selected', 'messages': []}, 'chat'))
            self.assertEqual(frames[-1]['usage']['completion_tokens'], 1)
            self.assertNotIn('event', frames[-1]); self.assertNotIn('sample', frames[-1])
            stop.assert_not_called()

    def test_pressure_abort_timeout_and_pending_comparison_block_new_requests(self):
        runtime = MacRuntime('/tmp/unused')
        runtime._memory_abort = 'pression critique'
        with self.assertRaisesRegex(ValueError, 'pression critique'):
            runtime._next()
        runtime._memory_abort = None
        with patch.object(runtime, '_stop') as stop, self.assertRaisesRegex(ValueError, 'délai'):
            runtime._next(timeout=0)
        stop.assert_called_once()
        runtime.job = {'state': 'running'}
        with self.assertRaisesRegex(ValueError, 'occupé'):
            runtime.load(SimpleNamespace(path='/tmp'), 1)
        with self.assertRaisesRegex(ValueError, 'en cours'):
            runtime.unload()
        runtime.cancelled.set()
        with self.assertRaisesRegex(ValueError, 'interrompue'):
            runtime._next()

    def test_switching_engines_never_substitutes_weights_and_closes_both_owned_runtimes(self):
        llama, mac = Mock(), Mock()
        llama.state_dir = '/tmp/unused'; llama.available.return_value = {'available': True}
        llama.load.return_value = {}; llama.describe.return_value = {'available': True, 'loaded': True}
        mac.available.return_value = {'available': True, 'packages': {'mtplx': '2.12.2'}}
        mac.describe.return_value = {'available': True, 'loaded': True, 'engine': 'mtplx'}
        with tempfile.TemporaryDirectory() as folder, patch('local_llm.engines.Accelerator', return_value=llama), patch('local_llm.engines.MacRuntime', return_value=mac):
            root = fixture(Path(folder)); item = inspect_model(root)
            manager = EngineManager()
            with self.assertRaises(ValueError):
                manager.load(item, 10 << 30, 'llamacpp')
            llama.unload.assert_not_called(); mac.load.assert_not_called()
            manager.load(item, 10 << 30, 'mtplx')
            llama.unload.assert_called_once(); mac.load.assert_called_once_with(item, 10 << 30, 'mtplx')
            self.assertIs(manager.active, mac)
            manager.close(); llama.close.assert_called_once(); mac.close.assert_called_once()

    def test_saved_decisions_are_recomputed_and_bound_to_public_workloads(self):
        runtime = MacRuntime('/tmp/unused')
        configs = {'standard': MacConfig(), 'faster': MacConfig(prefill_step_size=512)}
        def manifest(prompts):
            return [{'workload': i, 'category': CATEGORIES[i], 'output_limit': OUTPUT_LIMITS[i], 'input_tokens': 10 + i, 'prompt_sha256': str(i)} for i in range(3)]
        def trial(config, passes, faster=False):
            return {'config': asdict(config), 'samples': [dict(manifest(None)[i % 3], passes=passes, seconds=.8 if faster else 1,
                decode_tps=120 if faster else 100, generated_tokens=64, output_sha256=str(i % 3), prefill_seconds=.01) for i in range(passes * 3)]}
        report = {'training': {k: trial(v, TRAIN_PASSES, k != 'standard') for k, v in configs.items()},
                  'validation': {k: trial(v, VALIDATION_PASSES, k != 'standard') for k, v in configs.items()},
                  'training_manifest': manifest(None), 'validation_manifest': manifest(None)}
        report['profiles'] = verified_profiles(report['training'], report['validation'])
        report['summaries'] = {k: summarize(v['samples']) for k, v in report['validation'].items()}
        with patch.object(runtime, '_candidate_configs', return_value=configs), patch.object(runtime, '_manifest', side_effect=manifest):
            self.assertEqual(runtime._validate_report(report)['balanced']['winner'], 'faster')
            for mutate in (lambda r: r['profiles']['balanced'].update(gain_percent=999),
                           lambda r: r['validation']['faster']['samples'][0].update(output_sha256='changed'),
                           lambda r: r['training']['standard']['samples'][0].update(input_tokens=999),
                           lambda r: r['validation']['standard']['samples'].pop(),
                           lambda r: r['training']['faster']['config'].update(depth=3)):
                changed = copy.deepcopy(report); mutate(changed)
                with self.assertRaises(ValueError):
                    runtime._validate_report(changed)

    def test_no_gain_or_changed_validation_output_keeps_reference(self):
        def trial(passes, seconds=1, tps=100):
            return {'config': asdict(MacConfig()), 'samples': [dict(seconds=seconds, decode_tps=tps, prefill_seconds=.1,
                generated_tokens=64, output_sha256=str(i % 3), passes=passes) for i in range(passes * 3)]}
        training = {'standard': trial(2), 'candidate': trial(2, .5, 200)}
        validation = {'standard': trial(3), 'candidate': trial(3, .5, 200)}
        validation['candidate']['samples'][0]['output_sha256'] = 'different'
        self.assertEqual(verified_profiles(training, validation)['balanced']['winner'], 'standard')


class WorkerTests(unittest.TestCase):
    def test_reasoning_markers_across_chunks_and_initial_thinking_are_not_shown_as_answer(self):
        split = ReasoningSplit()
        frames = [split.feed(t) for t in ('<thi', 'nk>private', '</th', 'ink>Bonjour')]
        self.assertEqual(''.join(f.get('content', '') for f in frames), 'Bonjour')
        self.assertEqual(''.join(f.get('reasoning_content', '') for f in frames), 'private')
        split = ReasoningSplit(True)
        self.assertEqual(split.feed('private</think>Bonjour')['content'], 'Bonjour')

    def test_mtplx_incremental_callback_preserves_every_committed_token_and_uses_real_metrics(self):
        worker = Worker.__new__(Worker)
        worker.engine = 'mtplx'; worker.runtime = object(); worker.context_length = 4096
        worker.mx = Mock(); worker.mx.get_peak_memory.return_value = 100
        segments = iter(['Bon', 'jour'])
        detokenizer = Mock(); type(detokenizer).last_segment = property(lambda s: next(segments))
        worker.tokenizer = SimpleNamespace(eos_token_ids={0}, detokenizer=detokenizer)
        worker.context = Mock(return_value=({'prompt': 'exact', 'prompt_sha256': 'hash'}, [1, 2]))
        def generate(runtime, prompt, **options):
            options['token_callback']([10]); options['token_callback']([11])
            return SimpleNamespace(tokens=[10, 11, 0], finish_reason='stop', stats=SimpleNamespace(to_dict=lambda: {'prompt_eval_time_s': .1, 'decode_elapsed_s': .2}))
        # The final detokenizer segment is empty.
        segments = iter(['Bon', 'jour', ''])
        fake_generate = SimpleNamespace(generate_ar=generate, generate_mtpk=generate)
        frames = []
        with patch.dict('sys.modules', {'mtplx.generation': fake_generate, 'mtplx.sampling': SimpleNamespace(SamplerConfig=lambda **kwargs: kwargs)}):
            worker.generate({'messages': [], 'max_tokens': 8, 'depth': 1}, frames.append)
        self.assertEqual(''.join(f['choices'][0]['delta'].get('content', '') for f in frames), 'Bonjour')
        self.assertEqual(frames[-1]['usage'], {'prompt_tokens': 2, 'completion_tokens': 2})
        self.assertEqual(frames[-1]['timings']['predicted_per_second'], 10)
        self.assertEqual(frames[-1]['sample']['output_sha256'], hashlib.sha256(json.dumps([10, 11]).encode()).hexdigest())
        self.assertEqual(frames[-1]['timings']['cache_n'], 0)


if __name__ == '__main__':
    unittest.main()
