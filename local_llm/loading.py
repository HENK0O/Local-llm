from pathlib import Path
from typing import Tuple

from .gguf import GGUFReader, model_config, model_weights, tokenizer as gguf_tokenizer
from .model import LlamaModel
from .tokenizer import Tokenizer, load_tokenizer


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
