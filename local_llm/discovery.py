"""Bounded, read-only discovery of model files in explicit local libraries."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .config import ModelConfig
from .gguf import GGUFReader, model_config


@dataclass(frozen=True)
class DiscoveredModel:
    id: str
    name: str
    path: str
    source: str
    size_bytes: int
    compatible: bool
    reason: Optional[str]
    architecture: Optional[str]
    quantization: str

    def to_dict(self):
        return asdict(self)


def lmstudio_model_roots(home: Path) -> List[Path]:
    """Read only the configured library path, never credentials or model weights."""
    roots = [home / ".lmstudio/models", home / ".cache/lm-studio/models"]
    settings = [home / ".lmstudio/settings.json",
                home / "Library/Application Support/LM Studio/settings.json",
                home / ".config/LM Studio/settings.json"]
    if os.environ.get("APPDATA"):
        settings.append(Path(os.environ["APPDATA"]) / "LM Studio/settings.json")
    for path in settings:
        try:
            if path.stat().st_size > 1024 * 1024:
                continue
            value = json.loads(path.read_text()).get("downloadsFolder")
            if isinstance(value, str) and value.strip():
                folder = Path(value).expanduser()
                if folder.is_absolute():
                    roots.append(folder)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return list(dict.fromkeys(roots))


def default_model_roots() -> List[Path]:
    home = Path.home()
    roots = [Path.cwd() / "models", *lmstudio_model_roots(home),
             Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", home / ".cache/huggingface")) / "hub"))]
    roots.extend(Path(p).expanduser() for p in os.environ.get("LOCAL_LLM_MODEL_DIRS", "").split(os.pathsep) if p)
    return roots


def inspect_model(path: Path, source: str = "local") -> DiscoveredModel:
    path = path.resolve()
    architecture = None
    quantization = "F32/F16"
    reason = None
    size = 0
    try:
        if path.is_file():
            size = path.stat().st_size
            reader = GGUFReader(path)
            architecture = reader.metadata.get("general.architecture")
            quantization = "/".join(sorted({t.type_name for t in reader.tensors.values()}))
            if architecture != "llama":
                raise ValueError("Architecture non prise en charge : " + str(architecture))
            model_config(reader).validate()
            unsupported = {t.type_name for t in reader.tensors.values()} - {"F32", "F16", "BF16", "Q8_0", "Q4_0"}
            if unsupported:
                raise ValueError("Quantification non prise en charge : " + ", ".join(sorted(unsupported)))
            if not isinstance(reader.metadata.get("tokenizer.ggml.merges"), list):
                raise ValueError("Tokenizer BPE requis")
            if not any(k.startswith("tokenizer.chat_template") for k in reader.metadata):
                raise ValueError("Template de conversation absent")
            if reader.metadata.get("llama.rope.scaling.type", "none") != "none":
                raise ValueError("RoPE avec scaling non pris en charge")
        else:
            raw = json.loads((path / "config.json").read_text())
            architecture = raw.get("model_type", "llama")
            if architecture != "llama":
                raise ValueError("Architecture non prise en charge : " + str(architecture))
            if raw.get("rope_scaling"):
                raise ValueError("RoPE avec scaling non pris en charge")
            ModelConfig.from_dict(raw)
            weights = list(path.glob("*.safetensors")) + list(path.glob("weights.npz"))
            if not weights or not (path / "tokenizer.json").is_file():
                raise ValueError("Poids ou tokenizer manquants")
            from .tokenizer import load_tokenizer
            from .chat import require_chat_template
            require_chat_template(load_tokenizer(path / "tokenizer.json"))
            size = sum(p.stat().st_size for p in weights)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reason = str(exc)
    return DiscoveredModel(hashlib.sha256(str(path).encode()).hexdigest()[:20],
                           path.stem if path.is_file() else path.name, str(path), source,
                           size, reason is None, reason, architecture, quantization)


def discover_models(roots: Optional[Iterable[Path]] = None, limit: int = 256) -> List[DiscoveredModel]:
    found = {}
    studio_roots = {p.resolve() for p in lmstudio_model_roots(Path.home())}
    for root in roots if roots is not None else default_model_roots():
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        source = "LM Studio" if root in studio_roots or "lmstudio" in str(root) or "lm-studio" in str(root) else "Hugging Face" if "huggingface" in str(root) else "local"
        for directory, dirs, files in os.walk(root, followlinks=False):
            relative = Path(directory).relative_to(root)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"blobs", "node_modules", "__pycache__"}) if len(relative.parts) < 5 else []
            candidates = [Path(directory) / name for name in sorted(files) if name.lower().endswith(".gguf") and not name.lower().startswith("mmproj")]
            if "config.json" in files and "tokenizer.json" in files:
                candidates.append(Path(directory))
            for candidate in candidates:
                canonical = str(candidate.resolve())
                if canonical not in found:
                    item = inspect_model(candidate, source)
                    found[canonical] = item
                if len(found) >= limit:
                    return list(found.values())
    return sorted(found.values(), key=lambda m: (not m.compatible, "Q8_0" not in m.quantization, m.size_bytes, m.name.lower()))
