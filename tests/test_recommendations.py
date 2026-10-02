import json
import unittest
from unittest.mock import patch
from local_llm.recommendations import GIB, detect_hardware, recommend_models


class RecommendationTests(unittest.TestCase):
    def test_memory_budget_filters_large_models_and_reserves_system_memory(self):
        low = recommend_models({'memory_bytes': 4 * GIB, 'logical_cores': 2})
        self.assertEqual(low['budget_bytes'], GIB)
        self.assertEqual(low['models'], [])
        small = recommend_models({'memory_bytes': 8 * GIB, 'logical_cores': 2})
        self.assertNotIn('LM Studio', [m['runtime'] for m in small['models']])
        self.assertEqual(next(m['family'] for m in small['models'] if m['recommended']), 'SmolLM2-360M')
        large = recommend_models({'memory_bytes': 32 * GIB, 'logical_cores': 10})
        self.assertEqual(len(large['models']), 3)
        self.assertEqual(next(m['family'] for m in large['models'] if m['recommended']), 'SmolLM2-1.7B')

    def test_unknown_memory_is_explicit_and_no_speed_is_invented(self):
        report = recommend_models({'memory_bytes': None})
        self.assertTrue(all(m['fits'] is None for m in report['models']))
        self.assertNotIn('tokens_per_second', json.dumps(report))
        json.dumps(report, allow_nan=False)

    def test_failed_hardware_probe_preserves_unknown_memory(self):
        with patch('local_llm.recommendations.platform.system', return_value='Darwin'), patch('local_llm.recommendations.subprocess.check_output', side_effect=OSError):
            self.assertIsNone(detect_hardware()['memory_bytes'])
