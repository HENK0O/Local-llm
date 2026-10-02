import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_llm.accelerator import Accelerator, ExecutionConfig
from local_llm.calibration import (TRAIN_PROMPTS, VALIDATION_PROMPTS, CATEGORIES,
                                   assess_candidate, memory_plan)
from tests.test_accelerator import samples


class MemoryAndValidationTests(unittest.TestCase):
    def metadata(self):
        return {'general.architecture': 'llama', 'llama.context_length': 32768,
                'llama.block_count': 32, 'llama.embedding_length': 4096,
                'llama.attention.head_count': 32, 'llama.attention.head_count_kv': 8}

    def test_memory_plan_trades_slots_and_context_for_real_budget(self):
        metadata = self.metadata(); gib = 1024 ** 3
        small = memory_plan(metadata, gib, 3 * gib)
        large = memory_plan(metadata, gib, 12 * gib)
        self.assertEqual(large['context'], 4096)
        self.assertEqual(large['slots'], 1)
        self.assertGreater(memory_plan(metadata, gib, 12*gib, required=9000)['context'], large['context'])
        for plan in (small, large):
            self.assertLessEqual(plan['estimated_bytes'] + plan['reserve_bytes'], plan['available_bytes'])
            self.assertGreaterEqual(plan['context'], 512)
        with self.assertRaises(ValueError): memory_plan(metadata, gib, gib)
        with self.assertRaises(ValueError): memory_plan(metadata, gib, 3 * gib, required=32768)
        with self.assertRaises(ValueError): memory_plan(metadata, gib, None, required=8192)
        hybrid = memory_plan(dict(metadata, **{'general.architecture':'bailingmoe3', 'bailingmoe3.attention.head_count_kv':[0,1]}), gib, 8 * gib)
        self.assertEqual(hybrid['slots'], 1)
        self.assertTrue(hybrid['conservative'])

    def test_regression_in_one_category_cannot_hide_behind_faster_other_workloads(self):
        base = {'samples': samples()}
        rows = samples(.4, 200)
        rows[1]['seconds'] = rows[4]['seconds'] = 1.2
        decision = assess_candidate(base, {'samples': rows})
        self.assertFalse(decision['accepted'])
        self.assertIn('catégorie', decision['reason'])
        self.assertTrue(set(TRAIN_PROMPTS).isdisjoint(VALIDATION_PROMPTS))

    def test_validation_failure_keeps_standard_without_selecting_on_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Accelerator(executable='mock', state_dir=folder)
            runtime.path = Path('/tmp/mock.gguf'); runtime.model_id = 'target'
            runtime.job = {'state':'running'}
            starts = []
            def start(config): runtime.config = config; starts.append(config)
            def run(configs, prompts, passes, phase, trials):
                for name in configs:
                    # Screening prefers a fast speculative candidate. The winner's
                    # independent validation differs: never try the runner-up there.
                    scale = 1 if name == 'standard' else .6 if name == 'motifs-4' else .8
                    rows = [dict(row, passes=passes) for row in samples(scale, 100 / scale)[:3]] * passes
                    if prompts == VALIDATION_PROMPTS and name != 'standard':
                        rows = [dict(row, output_sha256='changed') for row in rows]
                    trials[name]['samples'] = rows
            with patch.object(runtime, '_start', side_effect=start), patch.object(runtime, '_run_trials', side_effect=run) as run_trials, patch.object(runtime, '_cache_benchmark', return_value={}), patch.object(runtime, 'available', return_value={}), patch('local_llm.accelerator.model_fingerprint', return_value=('weights', 100)):
                runtime._calibrate([])
            self.assertEqual(runtime.job['state'], 'complete')
            self.assertEqual(runtime.profile['candidate'], 'motifs-4')
            self.assertEqual(runtime.profile['winner'], 'standard')
            self.assertEqual(set(run_trials.call_args_list[2].args[0]), {'standard','motifs-4'})
            self.assertIn('Sorties différentes', runtime.profile['validation']['decision']['reason'])
            self.assertEqual(starts[-1].speculative, 'none')
            self.assertEqual(runtime.profile['gain_percent'], 0)

    def test_cache_saving_is_unavailable_without_verified_reuse(self):
        runtime = Accelerator(executable='mock')
        runtime._latency_sample = Mock(return_value={'first_token_seconds':.1,'first_text_seconds':None,
            'seconds':.3,'prefill_seconds':.05,'cached_tokens':0})
        result = runtime._cache_benchmark()
        self.assertFalse(result['cache_verified'])
        self.assertIsNone(result['prefill_seconds_saved'])

    def test_context_growth_preserves_messages_but_invalidates_measured_profile(self):
        runtime = Accelerator(executable='mock')
        runtime.path = Mock(); runtime.path.stat.return_value.st_size = 100
        runtime.model_id = 'target'; runtime.profile = {'old':True}
        runtime.process = Mock(); runtime.process.poll.return_value = None
        runtime.client = Mock()
        # iter_chat requires a generator with close().
        runtime.client.iter_chat.return_value = (chunk for chunk in [{'answer':True}])
        runtime.context = Mock(return_value={'prompt_tokens':5000})
        runtime.memory_probe = lambda: 8 * 1024 ** 3
        payload = {'model':'target', 'messages':[{'role':'user','content':'Preserve me'}], 'max_tokens':128}
        def start(config): runtime.config=config; runtime.slots.entries.clear()
        with patch('local_llm.accelerator.GGUFReader', return_value=Mock(metadata=self.metadata())), patch.object(runtime, '_start', side_effect=start):
            self.assertEqual(len(list(runtime.iter_chat(payload, 'a'))), 1)
        self.assertGreaterEqual(runtime.config.context, 5128)
        self.assertIsNone(runtime.profile)
        self.assertEqual(runtime.client.iter_chat.call_args.args[0]['messages'], payload['messages'])
        # The new bounded host cache permits exact-prefix reuse; the newly
        # created worker has no state to reuse yet.
        self.assertTrue(runtime.client.iter_chat.call_args.args[0]['cache_prompt'])


if __name__ == '__main__': unittest.main()
