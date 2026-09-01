import os
import unittest
from pathlib import Path

from local_llm.chat import ChatMessage, format_chat
from local_llm.generation import generate
from local_llm.loading import load_runtime


MODEL_PATH = os.environ.get("LOCAL_LLM_TEST_INSTRUCT")


@unittest.skipUnless(
    MODEL_PATH, "set LOCAL_LLM_TEST_INSTRUCT to run the Instruct integration test"
)
class RealInstructTests(unittest.TestCase):
    def test_chat_template_and_greedy_answer_match_transformers(self):
        model, tokenizer = load_runtime(Path(MODEL_PATH))
        prompt = format_chat(
            [ChatMessage("user", "Quelle est la capitale de la France ? Réponds en une phrase.")],
            tokenizer,
        )
        input_ids = [
            1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308,
            34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 65,
            2726, 290, 1264, 2618, 47765, 1121, 367, 2618, 4649, 9148, 428,
            2756, 96, 19870, 430, 16520, 8715, 30, 2, 198, 1, 520, 9531, 198,
        ]
        expected = [15319, 47765, 1121, 367, 2618, 4649, 1264, 7042, 30, 2]
        self.assertEqual(tokenizer.encode(prompt), input_ids)
        self.assertEqual(generate(model, input_ids, len(expected)).token_ids, expected)
        self.assertEqual(tokenizer.decode(expected), "La capitale de la France est Paris.")


if __name__ == "__main__":
    unittest.main()
