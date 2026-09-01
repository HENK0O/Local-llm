from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .tokenizer import BPETokenizer, Tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("chat message content must be a string")


def format_chatml(
    messages: Iterable[ChatMessage],
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render the ChatML template used by SmolLM2-Instruct."""
    conversation: List[ChatMessage] = list(messages)
    if not conversation or conversation[0].role != "system":
        conversation.insert(0, ChatMessage("system", system_prompt or DEFAULT_SYSTEM_PROMPT))
    elif system_prompt is not None:
        conversation[0] = ChatMessage("system", system_prompt)

    rendered = "".join(
        f"{CHATML_START}{message.role}\n{message.content}{CHATML_END}\n"
        for message in conversation
    )
    if add_generation_prompt:
        rendered += f"{CHATML_START}assistant\n"
    return rendered


def require_chatml(tokenizer: Tokenizer) -> None:
    if not isinstance(tokenizer, BPETokenizer):
        raise ValueError("chat mode requires a BPE tokenizer with ChatML special tokens")
    missing = {CHATML_START, CHATML_END} - set(tokenizer.special_by_text)
    if missing:
        raise ValueError("chat mode requires tokenizer tokens: " + ", ".join(sorted(missing)))
