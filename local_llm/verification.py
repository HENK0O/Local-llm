from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .generation import generate
from .loading import load_runtime


@dataclass(frozen=True)
class TensorComparison:
    name: str
    max_absolute_error: float
    mean_absolute_error: float
    within_tolerance: bool


def compare_reference(model_dir: Path, reference_path: Path,
                      atol: float = 2e-4, rtol: float = 2e-4) -> List[TensorComparison]:
    model, _ = load_runtime(model_dir)
    with np.load(reference_path, allow_pickle=False) as archive:
        if "input_ids" not in archive:
            raise ValueError("reference archive is missing input_ids")
        input_ids = np.asarray(archive["input_ids"], dtype=np.int64).reshape(-1)
        _, actual = model.forward_with_activations(input_ids)
        comparisons: List[TensorComparison] = []
        for name in ("embeddings", *(f"layer.{i}" for i in range(model.config.num_hidden_layers)),
                     "norm", "logits"):
            if name not in archive:
                continue
            expected = np.asarray(archive[name], dtype=np.float32)
            value = actual[name]
            if expected.shape != value.shape:
                raise ValueError(f"{name}: reference shape {expected.shape}, runtime shape {value.shape}")
            difference = np.abs(value - expected)
            comparisons.append(TensorComparison(
                name=name,
                max_absolute_error=float(difference.max(initial=0.0)),
                mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
                within_tolerance=bool(np.allclose(value, expected, atol=atol, rtol=rtol)),
            ))
        if "greedy_tokens" in archive:
            expected_tokens = np.asarray(archive["greedy_tokens"], dtype=np.int64).reshape(-1)
            actual_tokens = np.asarray(generate(model, input_ids.tolist(), len(expected_tokens)).token_ids)
            exact = np.array_equal(actual_tokens, expected_tokens)
            comparisons.append(TensorComparison(
                name="greedy_tokens",
                max_absolute_error=0.0 if exact else 1.0,
                mean_absolute_error=0.0 if exact else 1.0,
                within_tolerance=exact,
            ))
    return comparisons
