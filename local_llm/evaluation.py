from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .chat import ChatMessage, format_chat
from .loading import load_runtime, model_fingerprint
from .model import LlamaModel


@dataclass(frozen=True)
class LogitComparison:
    max_absolute_error: float
    mean_absolute_error: float
    greedy_tokens_identical: bool
    within_tolerance: bool


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    model_sha256: str
    reference: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    reference_token_ids: list[int]
    cached_vs_uncached: LogitComparison
    runtime_vs_reference: Optional[LogitComparison]
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    kv_cache_bytes: int
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogitTrace:
    prompt_token_ids: np.ndarray
    prefill_logits: np.ndarray
    decision_logits: np.ndarray
    generated_token_ids: np.ndarray
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    kv_cache_bytes: int = 0


def capture_trace(
    model: LlamaModel, prompt_token_ids: list[int], tokens: int, *, use_cache: bool
) -> LogitTrace:
    if not prompt_token_ids:
        raise ValueError("prompt must contain at least one token")
    if tokens <= 0:
        raise ValueError("evaluation requires --tokens > 0")
    if len(prompt_token_ids) + tokens - 1 > model.config.max_position_embeddings:
        raise ValueError("prompt and generated tokens exceed the model context length")

    sequence = list(prompt_token_ids)
    capacity = min(model.config.max_position_embeddings, len(sequence) + tokens)
    cache = model.new_cache(capacity) if use_cache else None
    start = time.perf_counter()
    prefill = model.forward(np.asarray(sequence, dtype=np.int64), cache=cache)
    prefill_seconds = time.perf_counter() - start
    current = prefill[-1]
    decisions = []
    generated = []
    decode_seconds = 0.0

    for index in range(tokens):
        decisions.append(np.asarray(current, dtype=np.float32).copy())
        token = int(np.argmax(current))
        generated.append(token)
        if model.config.eos_token_id is not None and token == model.config.eos_token_id:
            break
        if index + 1 == tokens:
            break
        sequence.append(token)
        start = time.perf_counter()
        if use_cache:
            current = model.forward(np.asarray([token], dtype=np.int64), cache=cache)[-1]
        else:
            current = model.forward(np.asarray(sequence, dtype=np.int64))[-1]
        decode_seconds += time.perf_counter() - start

    return LogitTrace(
        prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int64),
        prefill_logits=np.asarray(prefill, dtype=np.float32),
        decision_logits=np.stack(decisions),
        generated_token_ids=np.asarray(generated, dtype=np.int64),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        kv_cache_bytes=cache.nbytes if cache is not None else 0,
    )


def compare_traces(
    current: LogitTrace, reference: LogitTrace, atol: float, rtol: float
) -> LogitComparison:
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be non-negative")
    same_tokens = np.array_equal(current.generated_token_ids, reference.generated_token_ids)
    if (current.prompt_token_ids.shape != reference.prompt_token_ids.shape or
            not np.array_equal(current.prompt_token_ids, reference.prompt_token_ids)):
        raise ValueError("reference prompt tokens do not match the current prompt")
    if (current.prefill_logits.shape != reference.prefill_logits.shape or
            current.decision_logits.shape != reference.decision_logits.shape):
        return LogitComparison(float("inf"), float("inf"), same_tokens, False)

    differences = np.concatenate((
        np.abs(current.prefill_logits - reference.prefill_logits).ravel(),
        np.abs(current.decision_logits - reference.decision_logits).ravel(),
    ))
    within = (
        np.allclose(current.prefill_logits, reference.prefill_logits, atol=atol, rtol=rtol)
        and np.allclose(current.decision_logits, reference.decision_logits,
                        atol=atol, rtol=rtol)
    )
    return LogitComparison(
        max_absolute_error=float(np.max(differences, initial=0.0)),
        mean_absolute_error=float(np.mean(differences)) if differences.size else 0.0,
        greedy_tokens_identical=bool(same_tokens),
        within_tolerance=bool(within),
    )


def save_trace(trace: LogitTrace, path: Path, model_sha256: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"schema_version": 1, "model_sha256": model_sha256})
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        prompt_token_ids=trace.prompt_token_ids,
        prefill_logits=trace.prefill_logits,
        decision_logits=trace.decision_logits,
        generated_token_ids=trace.generated_token_ids,
    )


def load_trace(path: Path, expected_model_sha256: str) -> LogitTrace:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"metadata", "prompt_token_ids", "prefill_logits",
                    "decision_logits", "generated_token_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError("invalid reference trace; missing " + ", ".join(sorted(missing)))
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported reference trace schema")
        if metadata.get("model_sha256") != expected_model_sha256:
            raise ValueError("reference trace was produced with different model files")
        return LogitTrace(
            prompt_token_ids=archive["prompt_token_ids"].copy(),
            prefill_logits=archive["prefill_logits"].copy(),
            decision_logits=archive["decision_logits"].copy(),
            generated_token_ids=archive["generated_token_ids"].copy(),
        )


def _load_baguette_module(repo: Path):
    model_file = Path(repo) / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"Baguette model.py not found in {repo}")
    name = "_local_llm_baguette_reference"
    spec = importlib.util.spec_from_file_location(name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Baguette reference from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_baguette_reference(
    checkpoint: Path, repo: Path, prompt_token_ids: list[int], tokens: int
) -> LogitTrace:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("a .pt reference requires PyTorch") from exc

    module = _load_baguette_module(repo)
    data = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not isinstance(data.get("model_cfg"), dict):
        raise ValueError("invalid Baguette checkpoint")
    config = module.ModelConfig.from_dict(data["model_cfg"])
    reference_model = module.build_model(config)
    reference_model.load_state_dict(data["model"])
    reference_model.eval()

    generated = []
    decisions = []
    tensor = torch.tensor([prompt_token_ids], dtype=torch.long)
    max_len = min(config.max_seq_len, len(prompt_token_ids) + tokens)
    with torch.inference_mode():
        # The public forward gives us every prefill logit for layer-by-layer
        # correctness. Generation itself uses Baguette's native KV cache so its
        # timing remains a meaningful reference for the optimized runtime.
        logits, _, _ = reference_model(tensor, diagnostics=False)
        prefill = logits[0].float().cpu().numpy().copy()
        dtype = next(reference_model.parameters()).dtype
        caches = reference_model._alloc_caches(1, max_len, tensor.device, dtype)
        start = time.perf_counter()
        current = reference_model._forward_cached(tensor, caches, 0)[0, -1]
        prefill_seconds = time.perf_counter() - start
        decode_seconds = 0.0
        position = len(prompt_token_ids)
        for index in range(tokens):
            array = current.float().cpu().numpy()
            decisions.append(array.copy())
            token = int(np.argmax(array))
            generated.append(token)
            if token == 2 or index + 1 == tokens:
                break
            next_id = torch.tensor([[token]], dtype=torch.long)
            start = time.perf_counter()
            current = reference_model._forward_cached(next_id, caches, position)[0, -1]
            decode_seconds += time.perf_counter() - start
            position += 1

    return LogitTrace(
        prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int64),
        prefill_logits=np.asarray(prefill, dtype=np.float32),
        decision_logits=np.stack(decisions).astype(np.float32, copy=False),
        generated_token_ids=np.asarray(generated, dtype=np.int64),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        kv_cache_bytes=sum(
            value.numel() * value.element_size()
            for cache in caches for value in cache.values()
            if hasattr(value, "numel")
        ),
    )


def capture_external_reference(
    model_path: Path,
    reference: Path,
    reference_repo: Optional[Path],
    prompt_token_ids: list[int],
    tokens: int,
    model_sha256: Optional[str] = None,
) -> tuple[LogitTrace, str]:
    """Load a configured reference trace without reloading the local runtime."""
    reference = Path(reference)
    suffix = reference.suffix.lower()
    if suffix == ".npz":
        if model_sha256 is None:
            model_sha256, _ = model_fingerprint(model_path)
        return load_trace(reference, model_sha256), f"Trace NumPy: {reference.name}"
    if suffix not in {".pt", ".pth"}:
        raise ValueError("reference must be a .npz trace or Baguette .pt checkpoint")

    repo = Path(reference_repo) if reference_repo is not None else reference.parent
    conversion = Path(model_path) / "conversion.json"
    if conversion.is_file():
        with conversion.open("r", encoding="utf-8") as handle:
            source_sha256 = json.load(handle).get("source_sha256")
        if source_sha256 and source_sha256 != model_fingerprint(reference)[0]:
            raise ValueError("reference checkpoint differs from the converted source")
    trace = capture_baguette_reference(reference, repo, prompt_token_ids, tokens)
    return trace, f"Baguette PyTorch: {reference.name}"


def evaluate_runtime(
    model_path: Path,
    prompt: str,
    tokens: int,
    reference: Optional[Path] = None,
    reference_repo: Optional[Path] = None,
    atol: float = 2e-4,
    rtol: float = 2e-4,
    save_reference: Optional[Path] = None,
    chat: bool = False,
    system_prompt: Optional[str] = None,
) -> EvaluationReport:
    model_sha256, _ = model_fingerprint(model_path)
    model, tokenizer = load_runtime(model_path)
    if system_prompt is not None and not chat:
        raise ValueError("a system prompt requires chat mode")
    model_prompt = (
        format_chat([ChatMessage("user", prompt)], tokenizer, system_prompt=system_prompt)
        if chat else prompt
    )
    prompt_ids = tokenizer.encode(model_prompt)
    cached = capture_trace(model, prompt_ids, tokens, use_cache=True)
    uncached = capture_trace(model, prompt_ids, tokens, use_cache=False)
    cache_comparison = compare_traces(cached, uncached, atol, rtol)

    if save_reference is not None:
        save_trace(cached, save_reference, model_sha256)

    reference_trace = None
    reference_name = "none"
    if reference is not None:
        reference_trace, reference_name = capture_external_reference(
            model_path, reference, reference_repo, prompt_ids, tokens, model_sha256
        )

    external = (
        compare_traces(cached, reference_trace, atol, rtol)
        if reference_trace is not None else None
    )
    decoded = max(0, len(cached.generated_token_ids) - 1)
    passed = (
        cache_comparison.within_tolerance
        and cache_comparison.greedy_tokens_identical
        and (external is None or (
            external.within_tolerance and external.greedy_tokens_identical
        ))
    )
    return EvaluationReport(
        schema_version=1,
        model_sha256=model_sha256,
        reference=reference_name,
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        generated_token_ids=cached.generated_token_ids.tolist(),
        reference_token_ids=(reference_trace.generated_token_ids.tolist()
                             if reference_trace is not None else []),
        cached_vs_uncached=cache_comparison,
        runtime_vs_reference=external,
        prefill_tokens_per_second=(
            len(prompt_ids) / cached.prefill_seconds if cached.prefill_seconds else float("inf")
        ),
        decode_tokens_per_second=(
            decoded / cached.decode_seconds if cached.decode_seconds else float("inf")
        ),
        kv_cache_bytes=cached.kv_cache_bytes,
        passed=passed,
    )
