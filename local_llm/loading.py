import hashlib
from pathlib import Path
from typing import Tuple

from .gguf import GGUFReader, model_config, model_weights, tokenizer as gguf_tokenizer
from .model import LlamaModel
from .tokenizer import Tokenizer, load_tokenizer


def model_fingerprint(path: Path) -> Tuple[str, int]:
    """Hash model contents and directory-relative filenames without loading weights."""
    root = Path(path)
    is_directory = root.is_dir()
    if root.is_file():
        files = [root]
    elif is_directory:
        files = sorted(item for item in root.rglob("*") if item.is_file())
    else:
        raise FileNotFoundError(f"model path does not exist: {root}")
    digest = hashlib.sha256()
    total = 0
    for model_file in files:
        if is_directory:
            digest.update(str(model_file.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
        with model_file.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), total


def load_runtime(path: Path) -> Tuple[LlamaModel, Tokenizer]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"unsupported model file {path}; expected .gguf")
        reader = GGUFReader(path)
        return LlamaModel(model_config(reader), model_weights(reader)), gguf_tokenizer(reader)
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {path}")
    return LlamaModel.from_directory(path), load_tokenizer(path / "tokenizer.json")
