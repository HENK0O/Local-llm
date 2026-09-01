import unittest

from local_llm.chat import (
    DEFAULT_SYSTEM_PROMPT,
    ChatMessage,
    format_chatml,
    require_chatml,
)
from local_llm.tokenizer import BPETokenizer, ByteTokenizer


class ChatTests(unittest.TestCase):
    def test_formats_official_smollm2_template(self):
        rendered = format_chatml([ChatMessage("user", "Bonjour")])
        self.assertEqual(
            rendered,
            "<|im_start|>system\n"
            f"{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\nBonjour<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_keeps_history_and_overrides_system_prompt(self):
        rendered = format_chatml(
            [
                ChatMessage("system", "old"),
                ChatMessage("user", "2+2 ?"),
                ChatMessage("assistant", "4"),
                ChatMessage("user", "Et +1 ?"),
            ],
            system_prompt="Réponds brièvement.",
        )
        self.assertTrue(rendered.startswith("<|im_start|>system\nRéponds brièvement.<|im_end|>\n"))
        self.assertIn("<|im_start|>assistant\n4<|im_end|>\n", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            ChatMessage("tool", "result")

    def test_requires_chatml_special_tokens(self):
        with self.assertRaises(ValueError):
            require_chatml(ByteTokenizer())
        tokenizer = BPETokenizer(
            {"a": 0, "<|im_start|>": 1, "<|im_end|>": 2},
            [],
            [
                {"id": 1, "content": "<|im_start|>", "special": True},
                {"id": 2, "content": "<|im_end|>", "special": True},
            ],
        )
        require_chatml(tokenizer)


if __name__ == "__main__":
    unittest.main()
