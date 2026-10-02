from __future__ import annotations

import argparse
import codecs
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from .chat import ChatMessage, format_chat, require_chat_template
from .benchmark import compare_report, load_report, run_benchmark, save_report
from .converters import convert_baguette
from .evaluation import evaluate_runtime
from .generation import GenerationStats, generate_tokens
from .gguf import GGUFReader, q4_backend_name, q8_backend_name
from .loading import load_runtime
from .model import LlamaModel
from .server import serve as serve_http
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
) -> str:
    # ByteTokenizer adds BOS by default; pretrained BPE tokenizers generally do
    # not. In chat mode the embedded template supplies the correct beginning.
    prompt_tokens = tokenizer.encode(prompt)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    stats = None
    generated: List[int] = []
    for token, maybe_stats in generate_tokens(
        model, prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
    ):
        generated.append(token)
        token_data = tokenizer.token_bytes(token)
        if token_data:
            text = decoder.decode(token_data, final=False)
            print(text, end="", flush=True)
        stats = maybe_stats or stats
    print(decoder.decode(b"", final=True), end="", flush=True)
    if stats is not None:
        _print_stats(stats)
    return tokenizer.decode(generated)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-llm", description="Local inference, GPU calibration and transparent native CPU runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate text from a local model directory")
    run.add_argument("model_positional", nargs="?", type=Path, help="model directory")
    run.add_argument("--model", dest="model_option", type=Path, help="model directory")
    run.add_argument("--prompt", help="input prompt; omit with --interactive")
    run.add_argument("--interactive", action="store_true", help="read prompts until EOF or /quit")
    run.add_argument("--chat", action="store_true", help="use the model's embedded chat template")
    run.add_argument("--chat-template", help="named embedded template (implies --chat)")
    run.add_argument("--system", help="system prompt (implies --chat)")
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
    benchmark.add_argument("--output", type=Path, help="save a reproducible JSON report")
    benchmark.add_argument("--compare", type=Path, help="compare with a saved JSON baseline")
    benchmark.add_argument("--json", action="store_true", help="print the report as JSON")

    calibrate = subparsers.add_parser('calibrate', help='calibrate the direct llama.cpp runtime on this computer')
    calibrate.add_argument('model', type=Path, help='installed GGUF target model')
    calibrate.add_argument('--draft', type=Path, help='optional smaller GGUF with exactly the same tokenizer')
    calibrate.add_argument('--output', type=Path, help='export measured profile as JSON')

    evaluate = subparsers.add_parser(
        "evaluate", help="check logits, greedy tokens, KV cache and runtime regressions"
    )
    evaluate.add_argument("model", type=Path, help="converted model directory or GGUF file")
    evaluate.add_argument("--prompt", default="Bonjour, comment vas-tu ?")
    evaluate.add_argument("--tokens", type=int, default=8)
    evaluate.add_argument("--chat", action="store_true", help="apply the embedded chat template")
    evaluate.add_argument("--system", help="system prompt (requires --chat)")
    evaluate.add_argument(
        "--reference", type=Path,
        help="saved .npz trace or original Baguette .pt checkpoint",
    )
    evaluate.add_argument(
        "--reference-repo", type=Path,
        help="Baguette repository containing model.py (defaults to checkpoint directory)",
    )
    evaluate.add_argument(
        "--save-reference", type=Path,
        help="save the current known-good runtime trace as a compressed .npz",
    )
    evaluate.add_argument("--output", type=Path, help="save the evaluation summary as JSON")
    evaluate.add_argument("--atol", type=float, default=2e-4)
    evaluate.add_argument("--rtol", type=float, default=2e-4)
    evaluate.add_argument("--json", action="store_true", help="print the summary as JSON")

    profile = subparsers.add_parser(
        "profile", help="measure time spent in each Transformer operation"
    )
    profile.add_argument("model", type=Path)
    profile.add_argument("--prompt", default="Bonjour, comment vas-tu ?")
    profile.add_argument("--tokens", type=int, default=16)
    profile.add_argument("--chat", action="store_true", help="apply the embedded chat template")
    profile.add_argument("--system", help="system prompt (requires --chat)")
    profile.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="start a local HTTP chat server")
    serve.add_argument("model", nargs="?", type=Path, help="model path; omit to discover local models")
    serve.add_argument("--model-dir", action="append", type=Path, default=[],
                       help="additional model library to scan (repeatable)")
    serve.add_argument("--lm-studio", default="http://127.0.0.1:1234",
                       help="loopback LM Studio server for discovery and comparisons")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument('--engine', choices=['auto', 'gpu', 'native'], default='auto',
                       help='auto prefers installed llama.cpp for GGUF; native retains the CPU runtime')
    serve.add_argument("--max-tokens", type=int, default=128,
                       help="default maximum generated tokens per request")
    serve.add_argument("--max-request-tokens", type=int, default=512,
                       help="hard generation limit for the native HTTP backend")
    serve.add_argument("--max-lmstudio-tokens", type=int, default=8192,
                       help="hard generation limit for LM Studio, including reasoning")
    serve.add_argument("--max-connections", type=int, default=8,
                       help="maximum simultaneous HTTP connections")
    serve.add_argument("--allow-remote", action="store_true",
                       help="allow binding to a non-loopback address (no authentication)")
    serve.add_argument("--reference", type=Path,
                       help="Baguette .pt, Hugging Face directory, or .npz benchmark trace")
    serve.add_argument("--reference-repo", type=Path,
                       help="repository containing model.py for a Baguette checkpoint")

    convert = subparsers.add_parser(
        "convert-baguette", help="convert a Baguette .pt checkpoint for local-llm"
    )
    convert.add_argument("checkpoint", type=Path)
    convert.add_argument("--tokenizer", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'calibrate':
        from .accelerator import Accelerator
        from .discovery import inspect_model
        from .telemetry import SystemTelemetry
        runtime = Accelerator()
        telemetry = SystemTelemetry()
        def available_memory():
            memory = telemetry.snapshot()
            total, used = memory.get('memory_total_bytes'), memory.get('memory_used_bytes')
            return min(total, max(0, total - used) + (runtime._rss() or 0)) if total is not None and used is not None else None
        runtime.memory_probe = available_memory
        try:
            runtime.load(inspect_model(args.model), available_memory())
            runtime.optimize(args.draft)
            last = None
            while runtime.job['state'] == 'running':
                status = str(runtime.job['progress']) + '% · ' + runtime.job['message']
                if status != last:
                    print(status, file=sys.stderr, flush=True)
                    last = status
                time.sleep(0.5)
            if runtime.job['state'] != 'complete':
                raise ValueError(runtime.job['message'])
            report = runtime.profile
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        except KeyboardInterrupt:
            runtime.cancelled.set()
            return 130
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        finally:
            runtime.close()
            telemetry.close()
    if args.command == "create-toy":
        path = create_toy_model(args.output, args.seed)
        print(f"Toy model written to {path}")
        return 0
    if args.command == "convert-baguette":
        try:
            path = convert_baguette(args.checkpoint, args.tokenizer, args.output)
        except (FileExistsError, FileNotFoundError, KeyError, RuntimeError,
                TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Baguette model written to {path}")
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
        if counts.get("Q4_0"):
            print(f"Q4 backend: {q4_backend_name()}")
        if counts.get("Q8_0"):
            print(f"Q8 backend: {q8_backend_name()}")
        template_names = reader.metadata.get("tokenizer.chat_templates", [])
        if "tokenizer.chat_template" in reader.metadata or template_names:
            names = (["default"] if "tokenizer.chat_template" in reader.metadata else [])
            if isinstance(template_names, list):
                names.extend(str(name) for name in template_names)
            print("chat templates: " + ", ".join(names))
        if args.tensors:
            for info in reader.tensors.values():
                print(f"{info.name:<42} {str(info.shape):<20} {info.type_name}")
        return 0
    if args.command == "benchmark":
        try:
            report = run_benchmark(args.model, args.prompt, args.tokens, args.runs)
            comparison = compare_report(report, load_report(args.compare)) if args.compare else None
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.output:
            save_report(report, args.output)
        if args.json:
            output = report.to_dict()
            if comparison is not None:
                output["comparison"] = comparison
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return 0
        prefill = report.prefill_tokens_per_second
        decode = report.decode_tokens_per_second
        print(f"model: {report.model_sha256[:12]} | {report.model_format} | "
              f"{_format_bytes(report.model_bytes)}")
        print(f"backend: {report.backend}")
        print(f"runs: {report.runs} | prompt: {len(report.prompt_token_ids)} tokens | "
              f"generated: {len(report.generated_token_ids)}/{report.requested_tokens} tokens")
        print(f"prefill median: {prefill.median:.1f} tok/s "
              f"(min {prefill.minimum:.1f}, max {prefill.maximum:.1f})")
        print(f"decode median:  {decode.median:.1f} tok/s "
              f"(min {decode.minimum:.1f}, max {decode.maximum:.1f})")
        print(f"KV cache: {_format_bytes(report.kv_cache_bytes)}")
        if args.output:
            print(f"report: {args.output}")
        if comparison is not None:
            print("comparison: "
                  f"prefill {comparison['prefill_percent']:+.1f}% | "
                  f"decode {comparison['decode_percent']:+.1f}% | "
                  f"KV cache {comparison['kv_cache_percent']:+.1f}%")
        return 0
    if args.command == "evaluate":
        try:
            report = evaluate_runtime(
                model_path=args.model,
                prompt=args.prompt,
                tokens=args.tokens,
                reference=args.reference,
                reference_repo=args.reference_repo,
                atol=args.atol,
                rtol=args.rtol,
                save_reference=args.save_reference,
                chat=args.chat,
                system_prompt=args.system,
            )
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            cache = report.cached_vs_uncached
            print(f"model: {report.model_sha256[:12]} | prompt: "
                  f"{len(report.prompt_token_ids)} tokens | generated: "
                  f"{len(report.generated_token_ids)} tokens")
            print("cache KV vs recalcul complet: "
                  f"max {cache.max_absolute_error:.4e} | "
                  f"moyenne {cache.mean_absolute_error:.4e} | "
                  f"tokens {'identiques' if cache.greedy_tokens_identical else 'DIFFERENTS'}")
            if report.runtime_vs_reference is not None:
                reference = report.runtime_vs_reference
                print("runtime vs reference: "
                      f"max {reference.max_absolute_error:.4e} | "
                      f"moyenne {reference.mean_absolute_error:.4e} | "
                      f"tokens {'identiques' if reference.greedy_tokens_identical else 'DIFFERENTS'}")
            print(f"prefill: {report.prefill_tokens_per_second:.1f} tok/s | "
                  f"decode: {report.decode_tokens_per_second:.1f} tok/s | "
                  f"KV cache: {_format_bytes(report.kv_cache_bytes)}")
            print("resultat: " + ("OK" if report.passed else "ECHEC"))
            if args.save_reference:
                print(f"reference sauvegardee: {args.save_reference}")
            if args.output:
                print(f"rapport: {args.output}")
        return 0 if report.passed else 1
    if args.command == "profile":
        if args.tokens <= 0:
            parser.error("--tokens must be positive")
        if args.system is not None and not args.chat:
            parser.error("--system requires --chat")
        try:
            model, tokenizer = load_runtime(args.model)
            model_prompt = (
                format_chat([ChatMessage("user", args.prompt)], tokenizer,
                            system_prompt=args.system)
                if args.chat else args.prompt
            )
            prompt_tokens = tokenizer.encode(model_prompt)
            profiler = model.start_profiling()
            generated = []
            final_stats = None
            for token, stats in generate_tokens(model, prompt_tokens, args.tokens):
                generated.append(token)
                if stats is not None:
                    final_stats = stats
            model.stop_profiling()
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if final_stats is None:
            parser.error("profiling produced no generation statistics")
        output = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated),
            "prefill_tokens_per_second": final_stats.prefill_tokens_per_second,
            "decode_tokens_per_second": final_stats.decode_tokens_per_second,
            "kv_cache_bytes": final_stats.cache_bytes,
            **profiler.to_dict(),
        }
        if args.json:
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return 0
        print(f"prompt: {len(prompt_tokens)} tokens | generated: {len(generated)} tokens")
        print(f"prefill: {final_stats.prefill_tokens_per_second:.1f} tok/s | "
              f"decode: {final_stats.decode_tokens_per_second:.1f} tok/s | "
              f"KV cache: {_format_bytes(final_stats.cache_bytes)}")
        print(f"{'operation':<24} {'calls':>7} {'total ms':>11} {'ms/call':>10} {'part':>8}")
        for entry in profiler.entries():
            print(f"{entry.operation:<24} {entry.calls:>7} {entry.seconds * 1000:>11.2f} "
                  f"{entry.milliseconds_per_call:>10.3f} {entry.percent:>7.1f}%")
        return 0
    if args.command == "serve":
        if not 0 <= args.port <= 65535:
            parser.error("--port must be between 0 and 65535")
        if args.max_tokens <= 0:
            parser.error("--max-tokens must be positive")
        if not 1 <= args.max_request_tokens:
            parser.error("--max-request-tokens must be positive")
        if args.max_tokens > args.max_request_tokens:
            parser.error("--max-tokens must not exceed --max-request-tokens")
        if args.max_lmstudio_tokens < 1:
            parser.error("--max-lmstudio-tokens must be positive")
        if not 1 <= args.max_connections <= 128:
            parser.error("--max-connections must be between 1 and 128")
        try:
            serve_http(args.model, args.host, args.port, args.max_tokens,
                       args.reference, args.reference_repo, args.max_request_tokens,
                       args.max_connections, args.allow_remote, args.model_dir, args.lm_studio,
                       args.max_lmstudio_tokens, args.engine)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    model_path = args.model_option or args.model_positional
    if model_path is None:
        parser.error("run requires a model path (positional or --model)")
    if args.prompt is None and not args.interactive:
        parser.error("run requires --prompt or --interactive")
    model, tokenizer = load_runtime(model_path)
    if tokenizer.vocab_size != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the model")

    chat_mode = args.chat or args.system is not None or args.chat_template is not None
    if chat_mode:
        try:
            require_chat_template(tokenizer, args.chat_template)
        except ValueError as exc:
            parser.error(str(exc))

    history: List[ChatMessage] = []

    def answer(prompt: str) -> None:
        if chat_mode:
            history.append(ChatMessage("user", prompt))
            model_prompt = format_chat(history, tokenizer, system_prompt=args.system,
                                       template_name=args.chat_template)
            response = _run_once(model, tokenizer, model_prompt, args.max_new_tokens,
                                 args.temperature, args.top_k, args.top_p, args.seed)
            history.append(ChatMessage("assistant", response))
        else:
            _run_once(model, tokenizer, prompt, args.max_new_tokens, args.temperature,
                      args.top_k, args.top_p, args.seed)

    if args.prompt is not None:
        answer(args.prompt)
    if args.interactive:
        while True:
            try:
                prompt = input("\n> ")
            except EOFError:
                break
            if prompt.strip() in {"/quit", "/exit"}:
                break
            answer(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
