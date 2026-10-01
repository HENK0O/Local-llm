"""Compare the working runtime with a git revision using identical loaded weights.

Run from the repository root: python scripts/benchmark_inference.py --help
The selected revision is executed as Python code; use a trusted local revision.
"""
import argparse
import json
import os
import platform
import numpy as np
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm.generation import generate
from local_llm.loading import load_runtime
from local_llm.benchmark import model_fingerprint


def historical_module(ref, name):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:local_llm/{name}.py"], text=True
    )
    module = types.ModuleType(f"local_llm._baseline_{name}")
    module.__package__ = "local_llm"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:{name}.py", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.tokens < 2:
        parser.error("runs must be positive and tokens at least 2")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"], text=True
    ).strip()
    model, tokenizer = load_runtime(args.model)
    old_model = historical_module(revision, "model").LlamaModel(model.config, model.weights)
    old_generate = historical_module(revision, "generation").generate
    fingerprint, size = model_fingerprint(args.model)
    report = {"baseline_commit": revision, "model_sha256": fingerprint,
              "model_bytes": size, "runs": args.runs,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(),
                              "threads": os.environ.get("LOCAL_LLM_THREADS", "auto")},
              "scope": "Historical model.py and generation.py; shared current loader and kernels",
              "cases": []}
    # Warm both implementations before alternating measurements to reduce drift.
    for runtime, fn in ((old_model, old_generate), (model, generate)):
        fn(runtime, tokenizer.encode("Hello"), 2)
    for repetitions in (4, 16, 32):
        prompt = tokenizer.encode("Explain how a computer works. " * repetitions)
        timings = {"before": [], "after": []}
        expected = None
        for run in range(args.runs):
            variants = [("before", old_model, old_generate), ("after", model, generate)]
            for label, runtime, fn in variants[::1 if run % 2 == 0 else -1]:
                start = time.perf_counter()
                result = fn(runtime, prompt, args.tokens)
                elapsed = time.perf_counter() - start
                if expected is None:
                    expected = result.token_ids
                if result.token_ids != expected:
                    raise RuntimeError("Greedy tokens differ; refusing to report a speedup")
                timings[label].append({"prefill_seconds": result.stats.prefill_seconds,
                                       "decode_seconds": result.stats.decode_seconds,
                                       "total_seconds": elapsed})
        medians = {label: {metric: statistics.median(r[metric] for r in runs)
                           for metric in runs[0]} for label, runs in timings.items()}
        case = {"prompt_tokens": len(prompt), "generated_tokens": expected,
                "measurements": timings, "medians": medians,
                "prefill_speedup": medians["before"]["prefill_seconds"] /
                                   medians["after"]["prefill_seconds"]}
        report["cases"].append(case)
        print(f"{len(prompt)} prompt tokens: prefill {case['prefill_speedup']:.2f}x; identical tokens", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
