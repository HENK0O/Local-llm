from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

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


class _GenerationExtension(Extension):
    """Render Transformers' optional generation blocks without tracking masks."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(
            self.call_method("_render"), [], [], body
        ).set_lineno(lineno)

    @staticmethod
    def _render(caller):
        return caller()


@lru_cache(maxsize=64)
def _compile_template(source: str):
    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def strftime_now(pattern: str) -> str:
        return datetime.now().strftime(pattern)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationExtension, jinja2.ext.loopcontrols],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    return environment.from_string(source)


def select_chat_template(tokenizer: Tokenizer, name: Optional[str] = None) -> str:
    if not isinstance(tokenizer, BPETokenizer) or tokenizer.chat_template is None:
        raise ValueError("chat mode requires a tokenizer with an embedded chat template")
    configured = tokenizer.chat_template
    if isinstance(configured, str):
        if name not in (None, "default"):
            raise ValueError("this model only provides the default chat template")
        return configured
    if name is not None:
        try:
            return configured[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown chat template {name!r}; available: {', '.join(sorted(configured))}"
            ) from exc
    if "default" in configured:
        return configured["default"]
    raise ValueError(
        "model provides multiple chat templates without a default; choose one with "
        f"--chat-template ({', '.join(sorted(configured))})"
    )


def format_chat(
    messages: Iterable[ChatMessage],
    tokenizer: Tokenizer,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
    template_name: Optional[str] = None,
) -> str:
    """Render the Jinja chat template embedded in a tokenizer or GGUF."""
    conversation = list(messages)
    if system_prompt is not None:
        system = ChatMessage("system", system_prompt)
        if conversation and conversation[0].role == "system":
            conversation[0] = system
        else:
            conversation.insert(0, system)
    source = select_chat_template(tokenizer, template_name)
    values = {
        "messages": [{"role": message.role, "content": message.content}
                     for message in conversation],
        "add_generation_prompt": add_generation_prompt,
    }
    if isinstance(tokenizer, BPETokenizer):
        values.update(tokenizer.template_special_tokens())
    try:
        return _compile_template(source).render(**values)
    except jinja2.exceptions.TemplateError as exc:
        raise ValueError(f"chat template failed: {exc}") from exc


def require_chat_template(tokenizer: Tokenizer, name: Optional[str] = None) -> None:
    select_chat_template(tokenizer, name)
