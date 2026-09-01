from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional


class ByteTokenizer:
    """A dependency-free, reversible UTF-8 byte tokenizer.

    IDs below ``byte_offset`` are reserved for special tokens. This deliberately
    simple tokenizer makes the runtime testable independently of SentencePiece.
    """

    def __init__(
        self,
        byte_offset: int = 3,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        pad_token_id: Optional[int] = 0,
    ) -> None:
        if byte_offset < 0:
            raise ValueError("byte_offset must be non-negative")
        self.byte_offset = byte_offset
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = byte_offset + 256

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos and self.bos_token_id is not None:
            tokens.append(self.bos_token_id)
        tokens.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos and self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        data = bytearray()
        specials = {token for token in (self.bos_token_id, self.eos_token_id, self.pad_token_id) if token is not None}
        for token in tokens:
            token = int(token)
            if skip_special_tokens and token in specials:
                continue
            byte = token - self.byte_offset
            if 0 <= byte <= 255:
                data.append(byte)
        return data.decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("type") != "byte":
            raise ValueError("v0.1 supports tokenizer.json with type='byte' only")
        return cls(
            byte_offset=int(raw.get("byte_offset", 3)),
            bos_token_id=raw.get("bos_token_id", 1),
            eos_token_id=raw.get("eos_token_id", 2),
            pad_token_id=raw.get("pad_token_id", 0),
        )

    def save(self, path: Path) -> None:
        raw = {
            "type": "byte",
            "byte_offset": self.byte_offset,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2)
            handle.write("\n")

