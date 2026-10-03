import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_llm.accelerator import Accelerator, ExecutionConfig, PROTOCOL, SlotPool, draft_compatible, select_winner, summarize
from local_llm.calibration import verified_profiles


def samples(seconds=1, tps=100):
    return [{'seconds': seconds, 'decode_tps': tps, 'prefill_seconds': .01,
             'generated_tokens': 64, 'output_sha256': 'output-' + str(i % 3),
             'process_rss_bytes': 1000} for i in range(6)]


def trials(candidate=None):
    return {'standard': {'config': asdict(ExecutionConfig()), 'samples': samples()},
            'candidate': {'config': asdict(ExecutionConfig(batch=256)), 'samples': candidate or samples(.8, 120)}}


class CalibrationTests(unittest.TestCase):
    def test_winner_requires_identical_outputs_and_two_stable_faster_passes(self):
        self.assertEqual(select_winner(trials())[0], 'candidate')
        changed = samples(.5, 200)
        changed[3]['output_sha256'] = 'different'
        winner, summaries = select_winner(trials(changed))
        self.assertEqual(winner, 'standard')
        self.assertNotIn('candidate', summaries)
        noisy = samples(.5, 200)
        for row in noisy[3:]: row['seconds'] = 1.1
        self.assertEqual(select_winner(trials(noisy))[0], 'standard')
        self.assertEqual(select_winner(trials(samples(.8, 99)))[0], 'standard')
        self.assertEqual(select_winner(trials(samples(.96, 110)))[0], 'standard')

    def test_summary_measures_entire_workload_and_weights_decode_by_work(self):
        rows = samples()
        rows[2]['seconds'] = rows[5]['seconds'] = 4
        rows[2]['decode_tps'] = rows[5]['decode_tps'] = 50
        summary = summarize(rows)
        self.assertEqual(summary['seconds'], 6)
        self.assertEqual(summary['decode_tps'], 75)
        for bad in ([], samples()[:3], samples(tps=float('nan')), samples(seconds=0)):
            with self.assertRaises(ValueError): summarize(bad)
        self.assertEqual(select_winner(trials(samples(tps=float('nan'))))[0], 'standard')

    def test_configuration_cannot_inject_flags_or_change_precision(self):
        for kwargs in ({'speculative': '--external'}, {'batch': 1}, {'threads': -1},
                       {'flash': 'bad'}, {'context': 1}, {'slots': 0}, {'slots': 5}, {'kv_type': 'q4_0'}, {'draft_tokens': 128},
                       {'draft_path': '/tmp/model'}, {'speculative': 'draft-simple'}):
            with self.assertRaises(ValueError): ExecutionConfig(**kwargs)
        self.assertEqual(ExecutionConfig(speculative='draft-simple', draft_path='/tmp/draft.gguf').speculative, 'draft-simple')

    def test_exact_tokenizer_required_for_draft_and_auxiliary_models_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = Path(folder) / 'target.gguf', Path(folder) / 'draft.gguf'
            target.write_bytes(b'large-target'); draft.write_bytes(b'small')
            metadata = {'general.architecture': 'llama', 'tokenizer.ggml.tokens': ['a', 'b'], 'tokenizer.ggml.bos_token_id': 0}
            with patch('local_llm.accelerator.GGUFReader', side_effect=[SimpleNamespace(metadata=metadata), SimpleNamespace(metadata=dict(metadata))]):
                self.assertTrue(draft_compatible(target, draft))
            for changed in ({'tokenizer.ggml.tokens': ['b', 'a']}, {'tokenizer.ggml.bos_token_id': 1}, {'general.architecture': 'dflash'}):
                with patch('local_llm.accelerator.GGUFReader', side_effect=[SimpleNamespace(metadata=metadata), SimpleNamespace(metadata=dict(metadata, **changed))]):
                    self.assertFalse(draft_compatible(target, draft))
            self.assertFalse(draft_compatible(target, target))


class RuntimeTests(unittest.TestCase):
    def test_memory_abort_is_reported_and_partial_context_is_not_reused(self):
        runtime=self.runtime()
        runtime._memory_abort='Tentative arrêtée : pression mémoire macOS critique.'
        runtime.client.iter_chat.return_value=(chunk for chunk in [{'choices':[{'delta':{'content':'partial'}}]}])
        with self.assertRaisesRegex(ValueError,'pression mémoire macOS critique'):
            list(runtime.iter_chat({'model':'target','messages':[],'max_tokens':64},'pressure-failure'))
        self.assertNotIn('pressure-failure',runtime.slots.entries)
        self.assertIn('pressure-failure',runtime._dirty_conversations)

    def test_controlled_worker_stops_on_critical_or_unknown_pressure(self):
        for pressure in ('critical',None):
            with tempfile.TemporaryDirectory() as folder:
                runtime=Accelerator(executable='mock',state_dir=folder)
                child=Mock();child.poll.return_value=None
                def terminate(): child.poll.return_value=0
                child.terminate.side_effect=terminate
                runtime.process=child
                with patch('local_llm.accelerator.macos_memory_pressure',return_value=pressure):
                    runtime._watch_memory(child)
                    runtime._pressure_thread.join(timeout=3)
                self.assertIsNotNone(runtime._memory_abort)
                child.terminate.assert_called_once()
                runtime.close()

    def test_controlled_load_uses_small_batches_and_standard_has_same_capacity(self):
        metadata={'general.architecture':'llama','llama.block_count':1,'llama.context_length':4096,
                  'llama.attention.head_count':8,'llama.attention.head_count_kv':1,'llama.embedding_length':1024}
        from types import SimpleNamespace
        gib=1024**3
        with tempfile.TemporaryDirectory() as folder:
            runtime=Accelerator(executable='mock',state_dir=folder)
            target=Mock(spec=Path);target.suffix='.gguf';target.stat.return_value.st_size=int(7.83*gib)
            item=SimpleNamespace(path='/tmp/ling.gguf',architecture='llama',id='ling',name='Ling')
            def start(config):runtime.config=config
            with patch('local_llm.accelerator.Path',return_value=target), patch('local_llm.accelerator.GGUFReader',return_value=SimpleNamespace(metadata=metadata)), patch.object(runtime,'available',return_value={'available':True}), patch('local_llm.accelerator.detect_hardware',return_value={'memory_bytes':24*gib}), patch('local_llm.accelerator.macos_memory_pressure',return_value='normal'), patch.object(runtime,'_start',side_effect=start), patch.object(runtime,'_restore_profile'):
                runtime.load(item,int(7.9*gib))
            self.assertTrue(runtime.memory['controlled_attempt'])
            self.assertEqual((runtime.config.context,runtime.config.batch,runtime.config.ubatch),(2048,256,128))
            runtime.path=None
            base=runtime._candidate_configs(runtime.config,[])['standard']
            self.assertEqual((base.context,base.slots,base.batch,base.ubatch),(2048,1,256,128))
            self.assertEqual(base.speculative,'none')

    def test_missing_or_old_dependency_is_explicit(self):
        runtime = Accelerator(executable='/nonexistent/llama-server')
        self.assertFalse(runtime.available()['available'])
        runtime = Accelerator(executable='llama-server')
        with patch('local_llm.accelerator.subprocess.check_output', side_effect=['version: old', '--model']):
            self.assertFalse(runtime.available()['available'])

    def test_device_identity_does_not_include_fluctuating_memory(self):
        def capabilities(free):
            runtime = Accelerator(executable='llama-server')
            with patch('local_llm.accelerator.subprocess.check_output', side_effect=[
                'log timestamp\nversion: stable', '--cache-ram --spec-type --no-context-shift --no-webui --cache-type-k --spec-ngram-simple-size-m',
                f'Available devices:\nMTL0: Apple M5 (18000 MiB, {free} MiB free)']):
                return runtime.available()
        self.assertEqual(capabilities(1000), capabilities(5000))
        self.assertTrue(capabilities(1000)['gpu'])

    def test_worker_is_private_authenticated_gpu_and_only_owned_process_stops(self):
        runtime = Accelerator(executable='llama-server')
        runtime.path, runtime.model_id = Path('/tmp/a file.gguf'), 'target'
        runtime.capabilities = {'optional_flags': []}
        child = Mock(); child.poll.return_value = None
        client = Mock(); client._request.side_effect = [{'status': 'ok'}, {'total_slots': 2}]
        with patch('local_llm.accelerator.socket.socket') as socket, patch('local_llm.accelerator.subprocess.Popen', return_value=child) as popen, patch('local_llm.accelerator.LMStudioClient', return_value=client):
            socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 54321)
            runtime._start(ExecutionConfig(speculative='ngram-simple'))
            args = popen.call_args.args[0]
            self.assertEqual(args[args.index('--model') + 1], '/tmp/a file.gguf')
            self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')
            self.assertEqual(args[args.index('--n-gpu-layers') + 1], 'all')
            self.assertEqual(args[args.index('--cache-ram') + 1], '0')
            self.assertEqual(args[args.index('--parallel') + 1], '2')
            self.assertIn('--no-context-shift', args)
            self.assertGreaterEqual(len(args[args.index('--api-key') + 1]), 64)
            self.assertNotIn('shell', popen.call_args.kwargs)
            runtime.close()
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
        self.assertIsNone(runtime.process)

    def runtime(self):
        runtime = Accelerator(executable='unused')
        runtime.model_id = 'target'
        runtime.process = Mock(); runtime.process.poll.return_value = None
        runtime.client = Mock()
        runtime.context = Mock(return_value={'prompt_tokens': 10})
        return runtime

    def test_failed_or_interrupted_start_closes_log_and_its_child(self):
        for failure in (OSError('not found'), KeyboardInterrupt()):
            runtime = Accelerator(executable='llama-server')
            runtime.path, runtime.model_id = Path('/tmp/model.gguf'), 'target'
            with patch('local_llm.accelerator.socket.socket') as socket, patch('local_llm.accelerator.subprocess.Popen', side_effect=failure):
                socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 54321)
                with self.assertRaises(type(failure)): runtime._start(ExecutionConfig())
            self.assertIsNone(runtime.log)
            self.assertIsNone(runtime.process)

    def test_slots_reuse_exact_conversation_and_invalidate_partial_stream(self):
        runtime = self.runtime()
        payload = {'model': 'target', 'messages': [], 'max_tokens': 64}
        runtime.client.iter_chat.side_effect = lambda body: (chunk for chunk in [{'model': 'target'}])
        for conversation in ['a', 'b', 'a']:
            list(runtime.iter_chat(payload, conversation))
        bodies = [call.args[0] for call in runtime.client.iter_chat.call_args_list]
        self.assertEqual([body['id_slot'] for body in bodies], [0, 1, 0])
        self.assertEqual([body['cache_prompt'] for body in bodies], [False, False, True])
        stream = runtime.iter_chat(payload, 'a'); next(stream); stream.close()
        self.assertNotIn('a', runtime.slots.entries)
        list(runtime.iter_chat(payload, 'a'))
        self.assertFalse(runtime.client.iter_chat.call_args.args[0]['cache_prompt'])
        with self.assertRaisesRegex(ValueError, 'changé'):
            list(runtime.iter_chat(dict(payload, model='wrong'), 'a'))
        with patch('local_llm.accelerator.GGUFReader', return_value=SimpleNamespace(metadata={'llama.context_length':4096, 'general.architecture':'llama'})):
            runtime.path = Mock(); runtime.path.stat.return_value.st_size = 100
            with self.assertRaisesRegex(ValueError, 'contexte'):
                list(runtime.iter_chat(dict(payload, max_tokens=5000), 'a'))

    def test_slot_eviction_never_moves_another_conversation_to_a_dirty_slot(self):
        slots = SlotPool()
        self.assertEqual(slots.acquire('a'), (0, True))
        self.assertEqual(slots.acquire('b'), (1, True))
        self.assertEqual(slots.acquire('a'), (0, False))
        self.assertEqual(slots.acquire('c'), (1, True))
        self.assertEqual(slots.acquire('b'), (0, True))
        self.assertEqual(len(slots.entries), 2)

    def test_calibration_refuses_to_interrupt_a_running_generation(self):
        runtime = self.runtime()
        held, release = threading.Event(), threading.Event()
        def hold():
            with runtime.lock: held.set(); release.wait(5)
        thread = threading.Thread(target=hold); thread.start(); held.wait(5)
        try:
            with self.assertRaisesRegex(ValueError, 'en cours'): runtime.optimize()
        finally:
            release.set(); thread.join(5)
        self.assertIsNone(runtime.job)

    def test_cancellation_restores_original_configuration_before_marking_complete(self):
        runtime = self.runtime()
        original = ExecutionConfig(batch=256)
        runtime.config = original
        runtime.profile = {'original': True}
        runtime.job = {'state': 'running'}
        runtime.cancelled.set()
        with patch.object(runtime, '_start') as start, patch('local_llm.accelerator.model_fingerprint', return_value=('weights', 100)):
            runtime._calibrate(None)
            start.assert_called_once_with(original)
        self.assertEqual(runtime.job['state'], 'cancelled')
        self.assertEqual(runtime.profile, {'original': True})
        self.assertFalse(runtime.cancelled.is_set())

    def test_saved_profile_is_bound_to_runtime_weights_hardware_and_valid_samples(self):
        import os, platform
        rows = trials()
        for trial in rows.values():
            trial['samples'] = [dict(row, passes=2, workloads_count=6, workload=i%6, category=('discussion','code','contexte long')[i%3], input_tokens=40+i%6, output_limit=128) for i, row in enumerate(trial['samples'] * 2)]
        winner, summaries = select_winner(rows)
        validation = {name: dict(trial, samples=[dict(row, passes=3) for row in trial['samples'][:6]] * 3) for name, trial in rows.items()}
        verified = {name: summarize(trial['samples']) for name, trial in validation.items()}
        workloads = [{'id':i, 'category':('discussion','code','contexte long')[i%3], 'prompt':'public-' + str(i), 'input_tokens':40+i, 'output_limit':128} for i in range(6)]
        report = {'training_manifest': Accelerator._manifest(workloads), 'workload_manifest': Accelerator._manifest(workloads), 'profiles': verified_profiles(rows, validation), 'draft_fingerprints': {}, 'validation': {'trials': validation}, 'training_summaries': summaries, 'protocol': PROTOCOL, 'model_sha256': 'fingerprint',
                  'hardware': {'system': platform.system(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()},
                  'runtime': {'version': 'test', 'devices': 'GPU'}, 'context_length': 4096, 'slots': 2,
                  'config': rows[winner]['config'], 'trials': rows, 'summaries': verified, 'winner': winner}
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.runtime(); runtime.state_dir = Path(folder); runtime.path = Path('/tmp/model.gguf')
            source = Path(folder) / 'fingerprint.json'
            with patch('local_llm.accelerator.model_fingerprint', return_value=('fingerprint', 100)), patch.object(runtime, 'available', return_value={'version': 'test', 'devices': 'GPU'}), patch.object(runtime, '_start') as start, patch.object(runtime, '_benchmark_workloads', return_value=workloads):
                source.write_text(json.dumps(report)); runtime._restore_profile()
                start.assert_called_once_with(ExecutionConfig(batch=256))
                self.assertEqual(runtime.profile, report)
                start.reset_mock(); runtime.profile = None
                invalid_validation = json.loads(json.dumps(report['validation']))
                invalid_validation['trials']['candidate']['samples'][0]['output_sha256'] = 'changed'
                for broken in (dict(report, validation=invalid_validation), dict(report, model_sha256='changed'), dict(report, runtime={'version': 'old', 'devices': 'GPU'}), dict(report, context_length=2048), dict(report, winner='standard')):
                    source.write_text(json.dumps(broken)); runtime._restore_profile()
                    self.assertIsNone(runtime.profile)
                start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
