from __future__ import annotations

import hashlib
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .generation import generate
from .loading import load_runtime


@dataclass(frozen=True)
class MetricSummary:
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: int
    model_sha256: str
    model_bytes: int
    model_format: str
    architecture: Dict[str, int]
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    requested_tokens: int
    runs: int
    prefill_tokens_per_second: MetricSummary
    decode_tokens_per_second: MetricSummary
    kv_cache_bytes: int
    environment: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _model_files(path: Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(f"model path does not exist: {path}")


def model_fingerprint(path: Path) -> tuple[str, int]:
    """Hash filenames and contents so two benchmark reports use identical weights."""
    root = Path(path)
    digest = hashlib.sha256()
    total = 0
    for model_file in _model_files(root):
        if root.is_dir():
            digest.update(str(model_file.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
        with model_file.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), total


def _summary(values: List[float]) -> MetricSummary:
    return MetricSummary(statistics.median(values), min(values), max(values))


def run_benchmark(path: Path, prompt: str, tokens: int, runs: int) -> BenchmarkReport:
    if runs <= 0 or tokens <= 1:
        raise ValueError("benchmark requires runs > 0 and tokens > 1")
    fingerprint, model_bytes = model_fingerprint(path)
    model, tokenizer = load_runtime(path)
    prompt_tokens = tokenizer.encode(prompt)
    results = [generate(model, prompt_tokens, tokens) for _ in range(runs)]
    generated = results[0].token_ids
    if any(result.token_ids != generated for result in results[1:]):
        raise RuntimeError("greedy benchmark produced different tokens between runs")
    stats = [result.stats for result in results]
    config = model.config
    return BenchmarkReport(
        schema_version=1,
        model_sha256=fingerprint,
        model_bytes=model_bytes,
        model_format="gguf" if Path(path).is_file() else "directory",
        architecture={
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
        },
        prompt=prompt,
        prompt_token_ids=prompt_tokens,
        generated_token_ids=generated,
        requested_tokens=tokens,
        runs=runs,
        prefill_tokens_per_second=_summary(
            [item.prefill_tokens_per_second for item in stats]
        ),
        decode_tokens_per_second=_summary(
            [item.decode_tokens_per_second for item in stats]
        ),
        kv_cache_bytes=stats[0].cache_bytes,
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    )


def save_report(report: BenchmarkReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_report(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1:
        raise ValueError("unsupported benchmark report schema")
    return report


def compare_report(report: BenchmarkReport, baseline: Dict[str, Any]) -> Dict[str, float]:
    checks = {
        "model_sha256": report.model_sha256,
        "prompt_token_ids": report.prompt_token_ids,
        "generated_token_ids": report.generated_token_ids,
        "requested_tokens": report.requested_tokens,
        "runs": report.runs,
        "environment": report.environment,
    }
    for name, current in checks.items():
        if baseline.get(name) != current:
            raise ValueError(f"benchmark is not comparable: {name} differs from baseline")

    def change(metric: str) -> float:
        old = float(baseline[metric]["median"])
        new = float(getattr(report, metric).median)
        return ((new / old) - 1.0) * 100.0 if old else float("inf")

    old_cache = int(baseline["kv_cache_bytes"])
    cache_change = ((report.kv_cache_bytes / old_cache) - 1.0) * 100.0 if old_cache else 0.0
    return {
        "prefill_percent": change("prefill_tokens_per_second"),
        "decode_percent": change("decode_tokens_per_second"),
        "kv_cache_percent": cache_change,
    }
