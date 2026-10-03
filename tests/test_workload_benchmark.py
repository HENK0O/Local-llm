import unittest
from local_llm.calibration import summarize, assess_candidate


def rows(scale=1, tps=100):
    return [dict(seconds=scale, decode_tps=tps, prefill_seconds=.02, generated_tokens=128,
                 first_token_seconds=.05, input_tokens=(32, 40, 900, 48, 56, 3000)[i],
                 output_limit=512, output_sha256='output-'+str(i), process_rss_bytes=100,
                 process_rss_peak_bytes=200, passes=3, workload=i, workloads_count=6,
                 category=('discussion','code','contexte long')[i%3])
            for _ in range(3) for i in range(6)]


class RepresentativeBenchmarkTests(unittest.TestCase):
    def test_summary_covers_two_shapes_per_category_and_real_stream_latency(self):
        data = rows()
        summary = summarize(data)
        self.assertEqual(summary['seconds'], 6)
        self.assertEqual(summary['categories']['code']['seconds'], 2)
        self.assertEqual(len(summary['workloads']), 6)
        self.assertEqual(summary['workloads'][-1]['input_tokens'], 3000)
        self.assertEqual(summary['first_token_seconds'], .05)
        self.assertEqual(summary['process_rss_peak_bytes'], 200)

    def test_one_long_shape_regression_cannot_hide_behind_faster_short_shape(self):
        candidate = rows(.5, 200)
        for index in (4, 10, 16): candidate[index]['seconds'] = 1.2
        decision = assess_candidate({'samples': rows()}, {'samples': candidate}, category='code')
        self.assertFalse(decision['accepted'])
        self.assertIn('longueur', decision['reason'])

    def test_latency_regression_is_rejected_even_when_decode_and_total_are_faster(self):
        candidate = rows(.7, 150)
        for index in (5, 11, 17): candidate[index]['first_token_seconds'] = .1
        decision = assess_candidate({'samples': rows()}, {'samples': candidate})
        self.assertFalse(decision['accepted'])
        self.assertIn('premier token', decision['reason'])

    def test_incomplete_or_misordered_workloads_cannot_validate_a_profile(self):
        data = rows()
        with self.assertRaises(ValueError): summarize(data[:-1])
        data[8]['workload'] = 99
        with self.assertRaises(ValueError): summarize(data)
        candidate = rows(.7, 150)
        for row in candidate: row['input_tokens'] += 100
        self.assertFalse(assess_candidate({'samples':rows()}, {'samples':candidate})['accepted'])

class SpeculationSearchTests(unittest.TestCase):
    def test_capability_gated_mtp_depths_and_confidence_are_search_candidates(self):
        from pathlib import Path
        from unittest.mock import patch
        from local_llm.accelerator import Accelerator, ExecutionConfig
        runtime = Accelerator(executable='mock')
        runtime.path = Path('/existing/model.gguf')
        caps = {'specialized_methods':['draft-mtp'], 'optional_flags':['--spec-draft-p-min']}
        with patch.object(runtime, 'available', return_value=caps), patch('local_llm.accelerator.GGUFReader'), patch('local_llm.accelerator.mtp_head_count', return_value=1):
            configs = runtime._candidate_configs(ExecutionConfig(), [])
        mtp = [c for c in configs.values() if c.speculative == 'draft-mtp']
        self.assertEqual(set(c.draft_tokens for c in mtp), {2,3,4,6,8,12,16})
        self.assertEqual(set(c.draft_p_min for c in mtp), {0,.5,.8})
        self.assertTrue(all(c.draft_path is None for c in mtp))
        for value in (float('nan'), float('inf'), -.1, 1, True, '0.5'):
            with self.assertRaises(ValueError): ExecutionConfig(draft_p_min=value)
