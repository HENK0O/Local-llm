import unittest

from local_llm.chat import (
    DEFAULT_SYSTEM_PROMPT,
    ChatMessage,
    format_chat,
    format_chatml,
    require_chat_template,
    require_chatml,
)
from local_llm.tokenizer import BPETokenizer, ByteTokenizer


class ChatTests(unittest.TestCase):
    @staticmethod
    def tokenizer(chat_template=None):
        return BPETokenizer(
            {"a": 0, "<s>": 1, "</s>": 2},
            [],
            [
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
            bos_token_id=1,
            eos_token_id=2,
            chat_template=chat_template,
        )

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

    def test_renders_embedded_zephyr_template(self):
        template = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'system' %}\n"
            "{{ '<|system|>\\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|assistant|>' }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )
        rendered = format_chat(
            [ChatMessage("system", "Bref."), ChatMessage("user", "Bonjour")],
            self.tokenizer(template),
        )
        self.assertEqual(
            rendered,
            "<|system|>\nBref.</s>\n<|user|>\nBonjour</s>\n<|assistant|>\n",
        )

    def test_selects_named_template(self):
        tokenizer = self.tokenizer({
            "default": "{{ messages[0]['content'] }} default",
            "short": "{{ messages[0]['content'] }} short",
        })
        messages = [ChatMessage("user", "test")]
        self.assertEqual(format_chat(messages, tokenizer), "test default")
        self.assertEqual(format_chat(messages, tokenizer, template_name="short"), "test short")
        with self.assertRaisesRegex(ValueError, "available"):
            format_chat(messages, tokenizer, template_name="missing")

    def test_requires_embedded_template(self):
        with self.assertRaisesRegex(ValueError, "embedded chat template"):
            require_chat_template(self.tokenizer())

    def test_supports_generation_blocks_and_template_errors(self):
        tokenizer = self.tokenizer("{% generation %}assistant{% endgeneration %}")
        self.assertEqual(format_chat([ChatMessage("user", "test")], tokenizer), "assistant")
        tokenizer = self.tokenizer("{{ raise_exception('bad conversation') }}")
        with self.assertRaisesRegex(ValueError, "bad conversation"):
            format_chat([ChatMessage("user", "test")], tokenizer)


if __name__ == "__main__":
    unittest.main()
