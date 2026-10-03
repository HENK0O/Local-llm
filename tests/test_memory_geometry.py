import unittest

from local_llm.calibration import memory_plan
from local_llm.memory import allocations, geometry


class ArchitectureMemoryTests(unittest.TestCase):
    def test_ling_mla_and_kda_are_separate_from_per_token_dense_kv(self):
        m = {'general.architecture': 'bailingmoe3', 'bailingmoe3.block_count': 24,
             'bailingmoe3.attention.head_count': 16, 'bailingmoe3.attention.head_count_kv': [0, 0, 0, 1] * 6,
             'bailingmoe3.attention.key_length': 576, 'bailingmoe3.attention.value_length': 128,
             'bailingmoe3.kda.head_dim': 128, 'bailingmoe3.ssm.conv_kernel': 4}
        shape = geometry(m)
        self.assertEqual(shape['kv_bytes_per_token'], 6 * 576 * 2)
        self.assertEqual(shape['recurrent_state_bytes'], 18 * 4 * (3 * 3 * 16 * 128 + 16 * 128 * 128))
        plan = memory_plan(m, 8 * 1024**3, 16 * 1024**3, required=4096)
        self.assertEqual(plan['context_bytes'], 4096 * 6 * 576 * 2)
        self.assertLessEqual(plan['estimated_bytes'] + plan['reserve_bytes'], plan['available_bytes'])

    def test_qwen_hybrid_trunk_excludes_mtp_layer_and_counts_fixed_delta_state(self):
        m = {'general.architecture': 'qwen35', 'qwen35.block_count': 65, 'qwen35.nextn_predict_layers': 1,
             'qwen35.attention.head_count': 24, 'qwen35.attention.head_count_kv': 4,
             'qwen35.attention.key_length': 256, 'qwen35.attention.value_length': 256,
             'qwen35.ssm.conv_kernel': 4, 'qwen35.ssm.inner_size': 6144,
             'qwen35.ssm.state_size': 128, 'qwen35.ssm.group_count': 16,
             'qwen35.ssm.time_step_rank': 48, 'qwen35.full_attention_interval': 4}
        shape = geometry(m)
        self.assertEqual(shape['kv_bytes_per_token'], 16 * 4 * 512 * 2)
        self.assertEqual(shape['recurrent_state_bytes'], 48 * 4 * (3 * (6144 + 2 * 16 * 128) + 128 * 6144))

    def test_unknown_or_incomplete_hybrid_keeps_conservative_fallback(self):
        for arch in ('unknown', 'qwen35', 'bailingmoe3'):
            shape = geometry({'general.architecture': arch})
            self.assertTrue(shape['conservative'])
            self.assertGreaterEqual(shape['kv_bytes_per_token'], 256 * 1024)

    def test_allocations_sum_only_buffer_declarations_not_aggregate_summaries(self):
        logs = 'load_tensors: Metal_Mapped model buffer size = 10.00 MiB\nllama_kv_cache: Metal KV buffer size = 2.00 MiB\nllama_kv_cache: size = 2.00 MiB\nllama_memory_recurrent: Metal RS buffer size = 1.00 MiB'
        data = allocations(logs)
        self.assertEqual(data['declared_buffer_bytes'], 13 * 1024**2)
        self.assertEqual(len(data['buffers']), 3)
        self.assertFalse(data['physical_ram'])
        self.assertIsNone(allocations('unavailable')['declared_buffer_bytes'])
