import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_llm.accelerator import Accelerator, ExecutionConfig, draft_method
from local_llm.calibration import (USAGE_PROFILES, assess_candidate, shortlist,
                                   speculation_summary, verified_profiles)
from tests.test_accelerator import samples


class UsageProfileTests(unittest.TestCase):
    def trials(self):
        rows = samples(1.4, 100)
        for index in (1, 4):
            rows[index].update(seconds=.6, decode_tps=170)
        return {'standard': {'config': asdict(ExecutionConfig()), 'samples': samples()},
                'code-fast': {'config': asdict(ExecutionConfig(ubatch=256)), 'samples': rows}}

    def holdout(self, trials):
        return {name: dict(trial, samples=[dict(row, passes=3) for row in trial['samples'][:3]] * 3)
                for name, trial in trials.items()}

    def test_category_gain_is_scoped_and_cannot_be_advertised_as_universal(self):
        trials = self.trials()
        self.assertFalse(assess_candidate(trials['standard'], trials['code-fast'])['accepted'])
        report = verified_profiles(trials, self.holdout(trials))
        self.assertEqual(report['balanced']['winner'], 'standard')
        self.assertEqual(report['discussion']['winner'], 'standard')
        self.assertEqual(report['code']['winner'], 'code-fast')
        self.assertEqual(report['code']['category'], 'code')
        self.assertAlmostEqual(report['code']['decode_gain_percent'], 70)
        self.assertEqual(set(report), set(USAGE_PROFILES))

    def test_category_profile_still_requires_identical_outputs_in_every_category(self):
        trials = self.trials(); holdout = self.holdout(trials)
        holdout['code-fast']['samples'][0]['output_sha256'] = 'wrong-discussion'
        report = verified_profiles(trials, holdout)
        self.assertEqual(report['code']['candidate'], 'code-fast')
        self.assertEqual(report['code']['winner'], 'standard')
        self.assertEqual(report['code']['gain_percent'], 0)

    def test_screening_is_bounded_and_preserves_best_per_category_but_never_quality_failures(self):
        trials = self.trials()
        screening = {name: dict(trial, samples=trial['samples'][:3]) for name, trial in trials.items()}
        screening['changed'] = {'samples': [dict(s, seconds=.1, output_sha256='changed') for s in samples()[:3]]}
        screening['failed'] = {'samples': [], 'error':'runtime failed'}
        selected = shortlist(screening, 2)
        self.assertEqual(selected, ['standard','code-fast'])

    def test_actual_speculation_counters_and_missing_values_are_distinct(self):
        self.assertIsNone(speculation_summary(samples())['acceptance_percent'])
        rows = [dict(row, timings={'draft_n':10,'draft_n_accepted':7}) for row in samples()]
        self.assertEqual(speculation_summary(rows), {'proposed_tokens':60,'accepted_tokens':42,'acceptance_percent':70})
        rows[0]['timings']['draft_n_accepted'] = 100
        self.assertIsNone(speculation_summary(rows)['acceptance_percent'])

    def runtime(self):
        runtime = Accelerator(executable='unused')
        trials = self.trials()
        runtime.profile = {'profiles': verified_profiles(trials, self.holdout(trials))}
        return runtime

    def test_profile_switch_only_reloads_for_changed_settings_and_rolls_back_on_failure(self):
        runtime = self.runtime()
        def start(config): runtime.config = config
        with patch.object(runtime, '_start', side_effect=start) as launch:
            runtime.set_usage_profile('discussion')
            launch.assert_not_called()
            runtime.set_usage_profile('code')
            self.assertEqual(runtime.config.ubatch, 256)
            self.assertEqual(runtime.active_measurement()['category'], 'code')
            self.assertGreaterEqual(runtime.last_profile_switch_seconds, 0)
            launch.reset_mock()
            original = runtime.config
            launch.side_effect = [ValueError('failed'), None]
            with self.assertRaisesRegex(ValueError, 'failed'): runtime.set_usage_profile('balanced')
            self.assertEqual(runtime.usage_profile, 'code')
            self.assertEqual(launch.call_args.args[0], original)
        with self.assertRaises(ValueError): runtime.set_usage_profile('invalid')
        runtime.job = {'state':'running'}
        with self.assertRaises(ValueError): runtime.set_usage_profile('balanced')

    def test_independent_prefill_controls_and_extended_depths_are_validated(self):
        config = ExecutionConfig(threads=2, threads_batch=8, ubatch=1024, backend_sampling=True,
                                 speculative='ngram-mod', draft_tokens=64)
        self.assertEqual(config.threads_batch, 8)
        for kwargs in ({'ubatch':1024,'batch':256}, {'threads_batch':-1}, {'backend_sampling':1}, {'cache_ram_mib':999}):
            with self.assertRaises(ValueError): ExecutionConfig(**kwargs)
        runtime = Accelerator(executable='mock')
        with patch.object(runtime, 'available', return_value={'optional_flags':[]}):
            candidates = runtime._candidate_configs(ExecutionConfig(), [])
        self.assertNotIn('sampling-gpu', candidates)
        self.assertNotIn('motifs-adaptatifs-64', candidates)
        self.assertEqual(candidates['threads-décodage'].threads_batch, 0)
        self.assertEqual(candidates['threads-préparation'].threads, 0)

    def test_specialized_draft_requires_declared_exact_target_and_draft_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            target, draft = Path(folder)/'target.gguf', Path(folder)/'draft.gguf'
            target.write_bytes(b'target-long'); draft.write_bytes(b'draft')
            metadata = {'general.architecture':'dflash','tokenizer.ggml.tokens':['a']}
            with patch('local_llm.accelerator.GGUFReader', return_value=SimpleNamespace(metadata=metadata)):
                self.assertIsNone(draft_method(target, draft))
            binding = {'method':'draft-dflash','target_sha256':'target-sha','draft_sha256':'draft-sha','source':'https://example.org/upstream-model-card'}
            sidecar = draft.with_suffix('.local-llm-draft.json')
            with patch('local_llm.accelerator.model_fingerprint', side_effect=lambda p: ('target-sha' if p == target else 'draft-sha',0)):
                sidecar.write_text(json.dumps(binding))
                self.assertEqual(draft_method(target,draft),'draft-dflash')
                sidecar.write_text(json.dumps(dict(binding,target_sha256='another-model')))
                self.assertIsNone(draft_method(target,draft))
                sidecar.write_text(json.dumps(dict(binding,method='--external')))
                self.assertIsNone(draft_method(target,draft))

    def test_specialized_auxiliary_weights_are_never_loaded_as_standalone_chat_models(self):
        runtime = Accelerator(executable='mock')
        with patch.object(runtime, 'available', return_value={'available':True}), patch.object(runtime, '_start') as start:
            for architecture in ('dflash','dspark'):
                item = SimpleNamespace(path='/tmp/auxiliary.gguf', architecture=architecture)
                with self.assertRaisesRegex(ValueError, 'auxiliaires'): runtime.load(item)
            start.assert_not_called()

    def test_bounded_host_cache_still_invalidates_interrupted_requests(self):
        runtime = Accelerator(executable='unused')
        runtime.config = replace(runtime.config, cache_ram_mib=64, slots=1)
        runtime.model_id = 'target'; runtime.process = Mock(); runtime.process.poll.return_value=None
        runtime.client = Mock(); runtime.context = Mock(return_value={'prompt_tokens':10})
        runtime.client.iter_chat.side_effect=lambda body: (chunk for chunk in [{'model':'target'}])
        payload = {'model':'target','messages':[],'max_tokens':32}
        stream = runtime.iter_chat(payload,'a'); next(stream); stream.close()
        list(runtime.iter_chat(payload,'a'))
        self.assertFalse(runtime.client.iter_chat.call_args.args[0]['cache_prompt'])
        list(runtime.iter_chat(payload,'a'))
        self.assertTrue(runtime.client.iter_chat.call_args.args[0]['cache_prompt'])

    def test_disk_snapshots_are_never_saved_for_interrupted_generation(self):
        runtime = Accelerator(executable='unused')
        runtime.model_id = 'target'; runtime.process = Mock(); runtime.process.poll.return_value = None
        runtime.client = Mock(); runtime.context = Mock(return_value={'prompt_tokens':10})
        runtime.client.iter_chat.side_effect = lambda payload: (chunk for chunk in [{'model':'target'}])
        runtime._cache_compatible = Mock(return_value=True)
        runtime.cache_binding = 'bound-runtime'
        runtime.kv_store = Mock(); runtime.kv_store.budget = 512 * 1024**2
        runtime.kv_store.restore.return_value = False
        payload = {'model':'target', 'messages':[], 'max_tokens':32}
        stream = runtime.iter_chat(payload, 'a'); next(stream); stream.close()
        runtime.kv_store.save.assert_not_called()
        runtime.kv_store.remove.assert_called_once_with('a', 'bound-runtime')
        runtime.kv_store.restore.reset_mock()
        list(runtime.iter_chat(payload, 'a'))
        runtime.kv_store.restore.assert_not_called()
        self.assertFalse(runtime.client.iter_chat.call_args.args[0]['cache_prompt'])
        runtime.kv_store.save.assert_called_once()


if __name__ == '__main__': unittest.main()
