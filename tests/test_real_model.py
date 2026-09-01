import os
import unittest
from pathlib import Path

from local_llm.generation import generate
from local_llm.model import LlamaModel
from local_llm.tokenizer import load_tokenizer


MODEL_PATH = os.environ.get("LOCAL_LLM_TEST_MODEL")


@unittest.skipUnless(MODEL_PATH, "set LOCAL_LLM_TEST_MODEL to run the checkpoint integration test")
class RealModelTests(unittest.TestCase):
    def test_smollm2_tokenizer_and_greedy_tokens_match_reference(self):
        root = Path(MODEL_PATH)
        tokenizer = load_tokenizer(root / "tokenizer.json")
        prompt = "Bonjour, comment ça va ?"
        input_ids = [30904, 24583, 28, 5189, 5549, 117, 81, 46316, 9148]
        expected = [1715, 198, 198, 19, 216, 34, 30, 428, 2756, 28627, 2756, 16863, 8897, 886, 85, 2786]
        self.assertEqual(tokenizer.encode(prompt), input_ids)
        model = LlamaModel.from_directory(root)
        self.assertEqual(generate(model, input_ids, len(expected)).token_ids, expected)


if __name__ == "__main__":
    unittest.main()
