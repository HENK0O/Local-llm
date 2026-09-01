from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path
from typing import Optional, Sequence

from .generation import GenerationStats, generate_tokens
from .model import LlamaModel
from .tokenizer import ByteTokenizer
from .toy import create_toy_model


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
    tokenizer: ByteTokenizer,
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
        byte = token - tokenizer.byte_offset
        if 0 <= byte <= 255:
            text = decoder.decode(bytes([byte]), final=False)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "create-toy":
        path = create_toy_model(args.output, args.seed)
        print(f"Toy model written to {path}")
        return 0

    model_path = args.model_option or args.model_positional
    if model_path is None:
        parser.error("run requires a model path (positional or --model)")
    if args.prompt is None and not args.interactive:
        parser.error("run requires --prompt or --interactive")
    model = LlamaModel.from_directory(model_path)
    tokenizer = ByteTokenizer.load(model_path / "tokenizer.json")
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

