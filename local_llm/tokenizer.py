from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


ChatTemplate = Union[str, Dict[str, str]]


def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    encoder = dict(zip(values, map(chr, characters)))
    return encoder, {character: byte for byte, character in encoder.items()}


class ByteTokenizer:
    """A dependency-free, reversible UTF-8 byte tokenizer for toy models."""

    def __init__(self, byte_offset: int = 3, bos_token_id: Optional[int] = 1,
                 eos_token_id: Optional[int] = 2, pad_token_id: Optional[int] = 0) -> None:
        if byte_offset < 0:
            raise ValueError("byte_offset must be non-negative")
        self.byte_offset = byte_offset
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos_token_id, eos_token_id, pad_token_id
        self.vocab_size = byte_offset + 256

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos and self.bos_token_id is not None:
            tokens.append(self.bos_token_id)
        tokens.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos and self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        return tokens

    def token_bytes(self, token: int) -> bytes:
        byte = int(token) - self.byte_offset
        return bytes([byte]) if 0 <= byte <= 255 else b""

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = {token for token in (self.bos_token_id, self.eos_token_id, self.pad_token_id) if token is not None}
        data = b"".join(self.token_bytes(token) for token in tokens
                        if not (skip_special_tokens and int(token) in specials))
        return data.decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("type") != "byte":
            raise ValueError("expected tokenizer.json with type='byte'")
        return cls(int(raw.get("byte_offset", 3)), raw.get("bos_token_id", 1),
                   raw.get("eos_token_id", 2), raw.get("pad_token_id", 0))

    def save(self, path: Path) -> None:
        raw = {"type": "byte", "byte_offset": self.byte_offset, "bos_token_id": self.bos_token_id,
               "eos_token_id": self.eos_token_id, "pad_token_id": self.pad_token_id}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2)
            handle.write("\n")


class BPETokenizer:
    """GPT-2 compatible byte-level BPE loaded from Hugging Face tokenizer.json."""

    def __init__(self, vocab: Dict[str, int], merges: Sequence[Union[str, Sequence[str]]],
                 added_tokens: Sequence[dict] = (), bos_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None,
                 pattern: Optional[str] = None, individual_digits: bool = False,
                 chat_template: Optional[ChatTemplate] = None) -> None:
        try:
            import regex
        except ImportError as exc:
            raise RuntimeError("GPT-2 tokenizers require the 'regex' package: pip install -e .") from exc
        self.vocab = {token: int(index) for token, index in vocab.items()}
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        if len(self.id_to_token) != len(self.vocab):
            raise ValueError("tokenizer vocabulary contains duplicate IDs")
        self.vocab_size = max(self.id_to_token, default=-1) + 1
        self.byte_encoder, self.byte_decoder = _bytes_to_unicode()
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos_token_id, eos_token_id, pad_token_id
        self.chat_template = chat_template
        self.special_by_text = {item["content"]: int(item["id"]) for item in added_tokens
                                if item.get("special") and isinstance(item.get("content"), str)}
        self.special_ids = set(self.special_by_text.values())
        self.ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            parts = merge.split() if isinstance(merge, str) else list(merge)
            if len(parts) != 2:
                raise ValueError(f"invalid BPE merge: {merge!r}")
            self.ranks[(parts[0], parts[1])] = rank
        digits = r"\p{N}" if individual_digits else r"\p{N}+"
        default_pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+| ?" + digits +
                           r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+")
        self.pattern = regex.compile(pattern or default_pattern)

    @staticmethod
    def _pattern_from_pre_tokenizer(raw: dict) -> Optional[str]:
        patterns: List[str] = []

        def visit(node: object) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "Split":
                pattern = node.get("pattern", {})
                if isinstance(pattern, dict) and isinstance(pattern.get("Regex"), str):
                    patterns.append(pattern["Regex"])
            for child in node.get("pretokenizers", []):
                visit(child)

        visit(raw)
        return patterns[0] if patterns else None

    @staticmethod
    def _has_individual_digits(raw: dict) -> bool:
        if not isinstance(raw, dict):
            return False
        if raw.get("type") == "Digits" and raw.get("individual_digits") is True:
            return True
        return any(BPETokenizer._has_individual_digits(child) for child in raw.get("pretokenizers", []))

    @staticmethod
    def _normalize_chat_template(value: object) -> Optional[ChatTemplate]:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            templates = {str(name): template for name, template in value.items()
                         if isinstance(template, str)}
            return templates or None
        if isinstance(value, list):
            templates = {
                str(item["name"]): item["template"]
                for item in value
                if isinstance(item, dict) and isinstance(item.get("name"), str)
                and isinstance(item.get("template"), str)
            }
            return templates or None
        return None

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        model = raw.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError("only Hugging Face BPE tokenizer.json files are supported")
        config_path, tokenizer_config = path.with_name("tokenizer_config.json"), {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                tokenizer_config = json.load(handle)

        def special_id(name: str) -> Optional[int]:
            value = tokenizer_config.get(name)
            content = value.get("content") if isinstance(value, dict) else value
            if not isinstance(content, str):
                return None
            for item in raw.get("added_tokens", []):
                if item.get("content") == content:
                    return int(item["id"])
            return model.get("vocab", {}).get(content)

        pre_tokenizer = raw.get("pre_tokenizer", {})
        return cls(model["vocab"], model.get("merges", []), raw.get("added_tokens", []),
                   special_id("bos_token"), special_id("eos_token"), special_id("pad_token"),
                   cls._pattern_from_pre_tokenizer(pre_tokenizer), cls._has_individual_digits(pre_tokenizer),
                   cls._normalize_chat_template(tokenizer_config.get("chat_template")))

    def template_special_tokens(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for name, token_id in (
            ("bos_token", self.bos_token_id),
            ("eos_token", self.eos_token_id),
            ("pad_token", self.pad_token_id),
        ):
            if token_id is not None and token_id in self.id_to_token:
                result[name] = self.id_to_token[token_id]
        return result

    @lru_cache(maxsize=65536)
    def _bpe(self, token: str) -> Tuple[str, ...]:
        word = tuple(token)
        if len(word) < 2:
            return word
        while True:
            pair = min(set(zip(word, word[1:])), key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged: List[str] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
        return word

    def _ordinary_encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in self.pattern.findall(text):
            encoded = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            for token in self._bpe(encoded):
                try:
                    ids.append(self.vocab[token])
                except KeyError as exc:
                    raise ValueError(f"BPE vocabulary cannot encode token {token!r}") from exc
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos and self.bos_token_id is not None:
            tokens.append(self.bos_token_id)
        if self.special_by_text:
            import regex
            expression = "(" + "|".join(regex.escape(value) for value in
                                          sorted(self.special_by_text, key=len, reverse=True)) + ")"
            for piece in regex.compile(expression).split(text):
                if piece in self.special_by_text:
                    tokens.append(self.special_by_text[piece])
                elif piece:
                    tokens.extend(self._ordinary_encode(piece))
        else:
            tokens.extend(self._ordinary_encode(text))
        if add_eos and self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        return tokens

    def token_bytes(self, token: int) -> bytes:
        if int(token) in self.special_ids:
            return b""
        value = self.id_to_token.get(int(token), "")
        try:
            return bytes(self.byte_decoder[character] for character in value)
        except KeyError:
            return value.encode("utf-8")

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        data = b"".join(self.token_bytes(token) for token in tokens
                        if not (skip_special_tokens and int(token) in self.special_ids))
        return data.decode("utf-8", errors="replace")


Tokenizer = Union[ByteTokenizer, BPETokenizer]


def load_tokenizer(path: Path) -> Tokenizer:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("type") == "byte":
        return ByteTokenizer.load(path)
    if raw.get("model", {}).get("type") == "BPE":
        return BPETokenizer.load(path)
    raise ValueError("unsupported tokenizer.json format")
