from __future__ import annotations

import argparse
import codecs
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

from .generation import GenerationStats, generate, generate_tokens
from .gguf import GGUFReader
from .loading import load_runtime
from .model import LlamaModel
from .tokenizer import Tokenizer
from .toy import create_toy_model
from .verification import compare_reference


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _print_stats(stats: GenerationStats) -> None:
    print(
        "\n"
        f"[prefill {stats.prefill_tokens_per_second:.1f} tok/s | "
        f"decode {stats.decode_tokens_per_second:.1f} tok/s | "
        f"KV cache {_format_bytes(stats.cache_bytes)}]",
        file=sys.stderr,
    )


def _run_once(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: Optional[int],
) -> None:
    prompt_tokens = tokenizer.encode(prompt, add_bos=True)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    stats = None
    for token, maybe_stats in generate_tokens(
        model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
    ):
        token_data = tokenizer.token_bytes(token)
        if token_data:
            text = decoder.decode(token_data, final=False)
            print(text, end="", flush=True)
        stats = maybe_stats or stats
    print(decoder.decode(b"", final=True), end="", flush=True)
    if stats is not None:
        _print_stats(stats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-llm", description="Minimal NumPy Llama runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate text from a local model directory")
    run.add_argument("model_positional", nargs="?", type=Path, help="model directory")
    run.add_argument("--model", dest="model_option", type=Path, help="model directory")
    run.add_argument("--prompt", help="input prompt; omit with --interactive")
    run.add_argument("--interactive", action="store_true", help="read prompts until EOF or /quit")
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-k", type=int)
    run.add_argument("--top-p", type=float)
    run.add_argument("--seed", type=int)

    toy = subparsers.add_parser("create-toy", help="create a deterministic tiny model")
    toy.add_argument("output", type=Path)
    toy.add_argument("--seed", type=int, default=42)

    verify = subparsers.add_parser("verify", help="compare runtime activations with a reference NPZ")
    verify.add_argument("--model", required=True, type=Path)
    verify.add_argument("--reference", required=True, type=Path)
    verify.add_argument("--atol", type=float, default=2e-4)
    verify.add_argument("--rtol", type=float, default=2e-4)

    inspect = subparsers.add_parser("inspect", help="display GGUF metadata and tensor inventory")
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--tensors", action="store_true", help="list every tensor")

    benchmark = subparsers.add_parser("benchmark", help="benchmark prefill and cached decoding")
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--prompt", default="Bonjour, comment ça va ?")
    benchmark.add_argument("--tokens", type=int, default=32)
    benchmark.add_argument("--runs", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "create-toy":
        path = create_toy_model(args.output, args.seed)
        print(f"Toy model written to {path}")
        return 0
    if args.command == "verify":
        comparisons = compare_reference(args.model, args.reference, args.atol, args.rtol)
        if not comparisons:
            print("No matching activation tensors found in the reference", file=sys.stderr)
            return 2
        print(f"{'tensor':<18} {'max abs':>12} {'mean abs':>12}  status")
        for result in comparisons:
            status = "OK" if result.within_tolerance else "FAIL"
            print(f"{result.name:<18} {result.max_absolute_error:>12.4e} "
                  f"{result.mean_absolute_error:>12.4e}  {status}")
        logits = next((item for item in comparisons if item.name == "logits"), None)
        return 0 if logits is not None and all(item.within_tolerance for item in comparisons) else 1
    if args.command == "inspect":
        reader = GGUFReader(args.model)
        print(f"GGUF v{reader.version} | {len(reader.metadata)} metadata | "
              f"{len(reader.tensors)} tensors | alignment {reader.alignment}")
        for key in ("general.name", "general.architecture", "general.file_type",
                    "llama.block_count", "llama.context_length", "llama.embedding_length"):
            if key in reader.metadata:
                print(f"{key}: {reader.metadata[key]}")
        counts = {}
        for info in reader.tensors.values():
            counts[info.type_name] = counts.get(info.type_name, 0) + 1
        print("tensor types: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
        if args.tensors:
            for info in reader.tensors.values():
                print(f"{info.name:<42} {str(info.shape):<20} {info.type_name}")
        return 0
    if args.command == "benchmark":
        if args.runs <= 0 or args.tokens <= 1:
            parser.error("benchmark requires --runs > 0 and --tokens > 1")
        model, tokenizer = load_runtime(args.model)
        prompt_tokens = tokenizer.encode(args.prompt)
        results = [generate(model, prompt_tokens, args.tokens).stats for _ in range(args.runs)]
        prefill = [item.prefill_tokens_per_second for item in results]
        decode = [item.decode_tokens_per_second for item in results]
        print(f"runs: {args.runs} | prompt: {len(prompt_tokens)} tokens | decode: {args.tokens} tokens")
        print(f"prefill median: {statistics.median(prefill):.1f} tok/s "
              f"(min {min(prefill):.1f}, max {max(prefill):.1f})")
        print(f"decode median:  {statistics.median(decode):.1f} tok/s "
              f"(min {min(decode):.1f}, max {max(decode):.1f})")
        print(f"KV cache: {_format_bytes(results[0].cache_bytes)}")
        return 0

    model_path = args.model_option or args.model_positional
    if model_path is None:
        parser.error("run requires a model path (positional or --model)")
    if args.prompt is None and not args.interactive:
        parser.error("run requires --prompt or --interactive")
    model, tokenizer = load_runtime(model_path)
    if tokenizer.vocab_size != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the model")

    if args.prompt is not None:
        _run_once(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed)
    if args.interactive:
        while True:
            try:
                prompt = input("\n> ")
            except EOFError:
                break
            if prompt.strip() in {"/quit", "/exit"}:
                break
            _run_once(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
