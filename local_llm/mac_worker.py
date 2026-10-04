"""Offline JSON-lines worker in an existing Python 3.11+ MLX environment.

No HTTP listener, package installation, model conversion or fan control. MTPLX
is an optional external dependency, used through its Python runtime API. Its
Sustained profile is explicit; results do not represent its app's Turbo mode.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

from .mlx_experiment import validate_model


def probe():
    packages = {}
    for package, module in (('mlx', 'mlx'), ('mlx-lm', 'mlx_lm'), ('mtplx', 'mtplx')):
        try:
            if importlib.util.find_spec(module) is not None:
                packages[package] = importlib.metadata.version(package)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            pass
    apple = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return {'python': platform.python_version(), 'packages': packages,
            'available': apple and sys.version_info >= (3, 11) and 'mlx' in packages and 'mlx-lm' in packages}


class ReasoningSplit:
    """Separate think markers even when they arrive across text segments."""
    def __init__(self, thinking=False):
        self.thinking = thinking
        self.pending = ''

    def feed(self, text, final=False):
        self.pending += text
        result = {'content': '', 'reasoning_content': ''}
        while self.pending:
            marker = '</think>' if self.thinking else '<think>'
            index = self.pending.find(marker)
            key = 'reasoning_content' if self.thinking else 'content'
            if index >= 0:
                result[key] += self.pending[:index]
                self.pending = self.pending[index + len(marker):]
                self.thinking = not self.thinking
                continue
            keep = 0
            if not final:
                for n in range(1, min(len(marker), len(self.pending) + 1)):
                    if self.pending.endswith(marker[:n]):
                        keep = n
            result[key] += self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ''
            break
        return {key: value for key, value in result.items() if value}


class Worker:
    def __init__(self, root, engine, context, memory_limit):
        import mlx.core as mx
        self.mx, self.root, self.engine, self.context_length = mx, validate_model(root), engine, context
        if memory_limit:
            # MLX treats this as a guideline, not a hard allocation cap. The
            # parent also preflights RAM and stops on non-normal OS pressure.
            mx.set_memory_limit(memory_limit)
            mx.set_cache_limit(min(256 * 1024 ** 2, memory_limit // 16))
        self.runtime = None
        if engine == 'mlx':
            from mlx_lm import load
            self.model, self.tokenizer = load(str(self.root), tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
            mx.eval(self.model.parameters())
        elif engine == 'mtplx':
            from mtplx.profiles import apply_profile_env
            apply_profile_env('sustained')
            from mtplx.runtime import load
            self.runtime = load(self.root, mtp=True)
            self.model, self.tokenizer = self.runtime.model, self.runtime.tokenizer
        else:
            raise ValueError('Unknown Mac engine')
        if not self.tokenizer.chat_template:
            raise ValueError('Template de conversation local absent.')

    def context(self, messages, thinking=None):
        options = {} if thinking is None else {'enable_thinking': thinking}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return {'prompt': prompt, 'prompt_tokens': len(ids), 'context_length': self.context_length,
                'compression': 'none', 'kind': 'preview',
                'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}, ids

    def generate(self, body, send):
        preview, prompt = self.context(body['messages'], body.get('enable_thinking'))
        limit = body['max_tokens']
        if len(prompt) + limit > self.context_length:
            raise ValueError('Le contexte et la réponse dépassent la capacité choisie. Aucun message n’a été supprimé.')
        self.mx.reset_peak_memory()
        if body.get('seed') is not None:
            self.mx.random.seed(body['seed'])
        split = ReasoningSplit(preview['prompt'].rstrip().endswith('<think>'))
        started = time.perf_counter()
        first = None
        token_ids = []

        def emit(text):
            nonlocal first
            if first is None:
                first = time.perf_counter() - started
            delta = split.feed(text)
            # Emit token progress even for an empty detokenizer segment.
            send({'event': 'chunk', 'token_progress': True, 'choices': [{'index': 0, 'delta': delta}]})

        prefill, decode, finish = None, None, 'stop'
        if self.engine == 'mlx':
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
            eos = set(self.tokenizer.eos_token_ids)
            last_count = 0
            last = None
            for row in stream_generate(self.model, self.tokenizer, prompt, max_tokens=limit,
                                       sampler=make_sampler(temp=body.get('temperature', 0), top_p=body.get('top_p', 1), top_k=body.get('top_k', 0)),
                                       prefill_step_size=body.get('prefill_step_size', 2048)):
                # The final response can repeat the last token. EOS is excluded
                # from the equivalence hash on both backends.
                if row.generation_tokens > last_count and row.token not in eos:
                    token_ids.append(int(row.token))
                last_count = row.generation_tokens
                emit(row.text)
                last = row
            if last is None:
                raise ValueError('Aucun token MLX généré.')
            prefill = last.prompt_tokens / last.prompt_tps if last.prompt_tps > 0 else None
            decode = last.generation_tokens / last.generation_tps if last.generation_tps > 0 else None
            finish = last.finish_reason or 'stop'
        else:
            from mtplx.generation import generate_ar, generate_mtpk
            from mtplx.sampling import SamplerConfig
            detokenizer = self.tokenizer.detokenizer
            detokenizer.reset()

            def callback(ids):
                # MTPLX calls back with newly committed tokens, in batches.
                for token in ids:
                    detokenizer.add_token(token)
                    emit(detokenizer.last_segment)

            options = dict(max_tokens=limit, sampler=SamplerConfig(temperature=body.get('temperature', 0),
                           top_p=body.get('top_p', 1), top_k=body.get('top_k', 0)),
                           seed=body.get('seed', 0), token_callback=callback)
            depth = body.get('depth', 1)
            output = generate_mtpk(self.runtime, prompt, speculative_depth=depth, **options) if depth else generate_ar(self.runtime, prompt, **options)
            detokenizer.finalize()
            emit(detokenizer.last_segment)
            token_ids = [int(t) for t in output.tokens if t not in set(self.tokenizer.eos_token_ids)]
            stats = output.stats.to_dict()
            prefill = stats.get('prompt_eval_time_s')
            decode = stats.get('decode_elapsed_s')
            finish = output.finish_reason or ('length' if len(token_ids) >= limit else 'stop')
        tail = split.feed('', final=True)
        if tail:
            send({'event': 'chunk', 'choices': [{'index': 0, 'delta': tail}]})
        elapsed = time.perf_counter() - started
        # Normalize token accounting/rates to non-EOS committed tokens. All
        # generated reasoning tokens remain included. No cache is reused here.
        rate = len(token_ids) / decode if decode and decode > 0 else None
        send({'event': 'done', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
              'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(token_ids)},
              'timings': {'predicted_per_second': rate, 'predicted_ms': decode * 1000 if decode is not None else None,
                          'prompt_ms': prefill * 1000 if prefill is not None else None, 'cache_n': 0,
                          'first_token_seconds': first, 'request_seconds': elapsed,
                          'peak_gpu_bytes': int(self.mx.get_peak_memory())},
              'sample': {'seconds': elapsed, 'decode_tps': rate, 'prefill_seconds': prefill,
                         'generated_tokens': len(token_ids), 'first_token_seconds': first,
                         'input_tokens': len(prompt), 'output_limit': limit,
                         'process_rss_bytes': None, 'peak_gpu_bytes': int(self.mx.get_peak_memory()),
                         'output_sha256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                         'prompt_sha256': preview['prompt_sha256'], 'finish_reason': finish}})
        self.mx.clear_cache()


def main():
    output = sys.stdout
    def send(value):
        output.write(json.dumps(value, ensure_ascii=False) + '\n')
        output.flush()
    with contextlib.redirect_stdout(sys.stderr):
        if sys.argv[1:] == ['--probe']:
            send(probe())
            return
        try:
            worker = Worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
            send({'event': 'ready', **probe()})
            for line in sys.stdin:
                request = json.loads(line)
                try:
                    if request['op'] == 'context':
                        preview, _ = worker.context(request['messages'], request.get('enable_thinking'))
                        send({'event': 'done', 'context': preview})
                    elif request['op'] == 'generate':
                        worker.generate(request, send)
                    else:
                        raise ValueError('Unknown worker operation')
                except Exception as exc:
                    send({'event': 'error', 'error': str(exc)})
        except Exception as exc:
            send({'event': 'error', 'error': str(exc)})
            sys.exit(1)


if __name__ == '__main__':
    main()
