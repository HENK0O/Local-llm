import unittest

from local_llm.tokenizer import BPETokenizer, ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_normalizes_named_chat_templates(self):
        templates = BPETokenizer._normalize_chat_template([
            {"name": "default", "template": "default template"},
            {"name": "tool_use", "template": "tool template"},
            {"name": 3, "template": "ignored"},
        ])
        self.assertEqual(templates, {
            "default": "default template",
            "tool_use": "tool template",
        })

    def test_utf8_round_trip(self):
        tokenizer = ByteTokenizer()
        text = "Bonjour 👋 — ça va ?"
        tokens = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(tokens), text)

    def test_special_tokens(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.encode("", add_bos=True, add_eos=True), [1, 2])
        self.assertEqual(tokenizer.decode([1, 2]), "")

    def test_byte_level_bpe_round_trip(self):
        vocab = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "hel": 5, "hell": 6, "hello": 7,
                 "Ġ": 8, "w": 9, "wo": 10, "wor": 11, "worl": 12, "world": 13, "<eos>": 14}
        merges = ["h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d"]
        tokenizer = BPETokenizer(vocab, merges, [{"id": 14, "content": "<eos>", "special": True}],
                                 eos_token_id=14, pattern=r" ?\p{L}+")
        tokens = tokenizer.encode("hello world")
        self.assertEqual(tokens, [7, 8, 13])
        self.assertEqual(tokenizer.decode(tokens), "hello world")
        self.assertEqual(tokenizer.encode("hello<eos>"), [7, 14])


if __name__ == "__main__":
    unittest.main()
