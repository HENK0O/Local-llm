import unittest

import numpy as np

from local_llm.ops import apply_rope, rms_norm, softmax


class OpsTests(unittest.TestCase):
    def test_rms_norm_has_expected_value(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        result = rms_norm(x, np.ones(2, dtype=np.float32), 0.0)
        expected = x / np.sqrt((9.0 + 16.0) / 2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_rope_position_zero_is_identity(self):
        x = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
        np.testing.assert_array_equal(apply_rope(x, np.array([0]), 10000.0), x)

    def test_rope_uses_llama_split_half_layout(self):
        x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        result = apply_rope(x, np.array([1]), 10000.0)
        angles = np.array([1.0, 0.01], dtype=np.float32)
        expected = np.array(
            [[[
                1.0 * np.cos(angles[0]) - 3.0 * np.sin(angles[0]),
                2.0 * np.cos(angles[1]) - 4.0 * np.sin(angles[1]),
                3.0 * np.cos(angles[0]) + 1.0 * np.sin(angles[0]),
                4.0 * np.cos(angles[1]) + 2.0 * np.sin(angles[1]),
            ]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_rope_supports_gguf_interleaved_layout(self):
        x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        result = apply_rope(x, np.array([1]), 10000.0, interleaved=True)
        angles = np.array([1.0, 0.01], dtype=np.float32)
        expected = np.array([[[
            np.cos(angles[0]) - 2 * np.sin(angles[0]),
            np.sin(angles[0]) + 2 * np.cos(angles[0]),
            3 * np.cos(angles[1]) - 4 * np.sin(angles[1]),
            3 * np.sin(angles[1]) + 4 * np.cos(angles[1]),
        ]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_softmax_is_stable(self):
        result = softmax(np.array([10000.0, 10000.0], dtype=np.float32))
        np.testing.assert_allclose(result, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
