"""Compare cached CPU projections on exactly the same decoding workload."""
from __future__ import annotations

import statistics
import time
from typing import List

import numpy as np

from .gguf import Q4Matrix, Q8Matrix, q4_backend_name, q8_backend_name
from .model import LlamaModel


class PortableMatrix:
    """Reuse packed weights without native projection kernels or duplicate storage."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __getitem__(self, item):
        return self.matrix[item]

    def matmul(self, x):
        return self.matrix.matmul_numpy(x)


def portable_model(model: LlamaModel) -> LlamaModel:
    return LlamaModel(model.config, {
        name: PortableMatrix(w) if isinstance(w, (Q8Matrix, Q4Matrix)) else w
        for name, w in model.weights.items()
    })


def replay(model: LlamaModel, prompt: List[int], emitted: List[int]):
    cache = model.new_cache(min(model.config.max_position_embeddings, len(prompt) + len(emitted)))
    model.forward(np.asarray(prompt), cache, last_logits_only=True)
    seconds = 0.0
    logits = []
    for token in emitted[:-1]:
        start = time.perf_counter()
        current = model.forward(np.asarray([token]), cache)
        seconds += time.perf_counter() - start
        logits.append(current[-1])
    return seconds, np.asarray(logits)


def compare_cached(model: LlamaModel, prompt: List[int], emitted: List[int], steps: int = 8):
    emitted = emitted[:steps + 1]
    decoded = len(emitted) - 1
    if decoded < 1:
        raise ValueError("La réponse contient trop peu de tokens pour mesurer le décodage")
    specialized = any((isinstance(w, Q8Matrix) and q8_backend_name() == "native-cpp") or
                      (isinstance(w, Q4Matrix) and q4_backend_name() == "native-cpp")
                      for w in model.weights.values())
    baseline = portable_model(model)
    measurements = {"local": [], "baseline": []}
    errors = []
    within = True
    # Warm the model and BLAS outside the timed decode; alternate run order.
    for run in range(3):
        outputs = {}
        variants = [("local", model), ("baseline", baseline)]
        for label, runtime in variants[::1 if run % 2 == 0 else -1]:
            seconds, logits = replay(runtime, prompt, emitted)
            measurements[label].append(seconds)
            outputs[label] = logits
        errors.append(float(np.max(np.abs(outputs["local"] - outputs["baseline"]))))
        within = within and bool(np.allclose(outputs["local"], outputs["baseline"], rtol=2e-4, atol=2e-4))
    local_speed = decoded / statistics.median(measurements["local"])
    baseline_speed = decoded / statistics.median(measurements["baseline"])
    return {
        "kind": "numpy_cached", "reference_name": "CPU NumPy · projections · cache KV",
        "local_tokens_per_second": local_speed, "baseline_tokens_per_second": baseline_speed,
        "delta_tokens_per_second": None,
        "diagnostic_delta_tokens_per_second": local_speed - baseline_speed if within and specialized else None,
        "speedup": None,
        "diagnostic_ratio": local_speed / baseline_speed if within and specialized else None,
        "scope": "projection_diagnostic", "validated_engine_gain": False,
        "reference_is_standard_engine": False,
        "comparable": within, "same_weights": True, "specialized_projections": specialized,
        "within_tolerance": within, "max_absolute_error": max(errors),
        "decode_steps": decoded, "runs": 3, "measurements_seconds": measurements,
        "method": ("Rejeu des mêmes tokens, cache KV des deux côtés. Médiane de 3 mesures, hors prefill et sampling. Les autres opérations sont partagées : on isole les projections natives face aux projections NumPy. Ce témoin ne représente pas llama.cpp ou un GPU."
                   if specialized else "Les deux chemins utilisent les mêmes projections NumPy/BLAS. Aucun gain attribuable à des projections natives n’est affiché ; les différences de débit reflètent le bruit de mesure."),
    }
