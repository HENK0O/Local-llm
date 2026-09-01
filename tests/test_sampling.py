import unittest

import numpy as np

from local_llm.generation import sample_token


class SamplingTests(unittest.TestCase):
    def test_zero_temperature_is_greedy(self):
        self.assertEqual(sample_token(np.array([0.0, 3.0, 1.0]), temperature=0), 1)

    def test_top_k_one_is_greedy(self):
        rng = np.random.default_rng(4)
        samples = [sample_token(np.array([0.0, 3.0, 1.0]), 1.0, top_k=1, rng=rng) for _ in range(20)]
        self.assertEqual(set(samples), {1})

    def test_invalid_top_p_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_token(np.array([0.0, 1.0]), temperature=1.0, top_p=0.0)


if __name__ == "__main__":
    unittest.main()

