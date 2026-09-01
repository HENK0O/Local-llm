import os
import unittest
from pathlib import Path

from local_llm.generation import generate
from local_llm.loading import load_runtime


GGUF_PATH = os.environ.get("LOCAL_LLM_TEST_GGUF")


@unittest.skipUnless(GGUF_PATH, "set LOCAL_LLM_TEST_GGUF to run the GGUF integration test")
class RealGGUFTests(unittest.TestCase):
    def test_smollm2_f16_gguf_matches_reference_tokens(self):
        model, tokenizer = load_runtime(Path(GGUF_PATH))
        prompt = "Bonjour, comment ça va ?"
        input_ids = [30904, 24583, 28, 5189, 5549, 117, 81, 46316, 9148]
        expected = [1715, 198, 198, 19, 216, 34, 30, 428, 2756, 28627, 2756, 16863, 8897, 886, 85, 2786]
        self.assertEqual(tokenizer.encode(prompt), input_ids)
        self.assertEqual(generate(model, input_ids, len(expected)).token_ids, expected)


if __name__ == "__main__":
    unittest.main()
