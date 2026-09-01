import unittest

from local_llm.tokenizer import ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_utf8_round_trip(self):
        tokenizer = ByteTokenizer()
        text = "Bonjour 👋 — ça va ?"
        tokens = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(tokens), text)

    def test_special_tokens(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.encode("", add_bos=True, add_eos=True), [1, 2])
        self.assertEqual(tokenizer.decode([1, 2]), "")


if __name__ == "__main__":
    unittest.main()

