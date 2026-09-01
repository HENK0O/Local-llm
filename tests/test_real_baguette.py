import os
import unittest
from pathlib import Path

from local_llm.chat import ChatMessage, format_chat
from local_llm.generation import generate
from local_llm.loading import load_runtime


MODEL_PATH = os.environ.get("LOCAL_LLM_TEST_BAGUETTE")


@unittest.skipUnless(
    MODEL_PATH, "set LOCAL_LLM_TEST_BAGUETTE to run the converted Baguette integration test"
)
class RealBaguetteTests(unittest.TestCase):
    def test_greedy_chat_tokens_match_pytorch_reference(self):
        model, tokenizer = load_runtime(Path(MODEL_PATH))
        prompt = format_chat([ChatMessage("user", "Combien font 2 + 2 ?")], tokenizer)
        generated = generate(model, tokenizer.encode(prompt), 24, temperature=0).token_ids
        expected = [3, 203, 22, 2649, 412, 5805, 754, 18, 203, 4, 203, 24, 2]
        self.assertEqual(generated, expected)
        self.assertEqual(tokenizer.decode(generated).strip(), "2 + 2 = 4.\n\n4")


if __name__ == "__main__":
    unittest.main()
