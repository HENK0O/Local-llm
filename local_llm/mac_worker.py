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
from .mlx_cache import ConversationCache


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


class StreamBuffer:
    """Keep first-token/text delivery immediate; coalesce subsequent IPC writes."""
    def __init__(self, send, clock=time.perf_counter):
        self.send, self.clock = send, clock
        self.pending = {}
        self.count = 0
        self.last_flush = None
        self.sent_text = False

    def push(self, delta, token_progress=True):
        for key, text in delta.items():
            self.pending[key] = self.pending.get(key, '') + text
        self.count += int(token_progress)
        now = self.clock()
        if (self.last_flush is None or delta.get('content') and not self.sent_text
                or self.count >= 8 or now - self.last_flush >= .012):
            self.flush()

    def flush(self):
        if self.count or self.pending:
            self.send({'event': 'chunk', 'token_progress': bool(self.count),
                       'choices': [{'index': 0, 'delta': self.pending}]})
            self.sent_text = self.sent_text or bool(self.pending.get('content'))
            self.pending, self.count, self.last_flush = {}, 0, self.clock()


class Worker:
    def __init__(self, root, engine, context, memory_limit, cache_budget=0):
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
        self.conversations = None
        self._prepared = None
        if engine == 'mlx':
            from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache
            self.conversations = ConversationCache(lambda: make_prompt_cache(self.model),
                can_trim_prompt_cache, trim_prompt_cache, cache_budget)

    def cache_summary(self):
        pool = getattr(self, 'conversations', None)
        return pool.summary() if pool is not None else {'supported': False, 'entries': 0, 'bytes': 0}

    def _prefill_prefix(self, prompt, layers, step):
        # Keep the last prompt token for stream_generate, which produces logits
        # and starts decoding. No change of cache size or precision is requested.
        from mlx_lm.generate import generation_stream
        with self.mx.stream(generation_stream):
            for offset in range(0, len(prompt), step):
                self.model(self.mx.array(prompt[offset:offset + step])[None], cache=layers)
                self.mx.eval([layer.state for layer in layers])
                self.mx.clear_cache()

    def context(self, messages, thinking=None):
        key = (tuple((m['role'], m['content']) for m in messages), thinking)
        prepared = getattr(self, '_prepared', None)
        if prepared is None or prepared[0] != key:
            options = {} if thinking is None else {'enable_thinking': thinking}
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            self._prepared = (key, prompt, ids, digest)
        else:
            _, prompt, ids, digest = prepared
        return {'prompt': prompt, 'prompt_tokens': len(ids), 'context_length': self.context_length,
                'compression': 'none', 'kind': 'preview',
                'prompt_sha256': digest}, ids

    def generate(self, body, send):
        preview, prompt = self.context(body['messages'], body.get('enable_thinking'))
        limit = body['max_tokens']
        if not prompt or type(limit) is not int or limit <= 0:
            raise ValueError('Un prompt non vide et une capacité de réponse positive sont requis.')
        if len(prompt) + limit > self.context_length:
            raise ValueError('Le contexte et la réponse dépassent la capacité choisie. Aucun message n’a été supprimé.')
        self.mx.reset_peak_memory()
        if body.get('seed') is not None:
            self.mx.random.seed(body['seed'])
        split = ReasoningSplit(preview['prompt'].rstrip().endswith('<think>'))
        started = time.perf_counter()
        first = None
        token_ids = []
        reused, cache_bytes, processed = 0, None, len(prompt)
        stream = StreamBuffer(send)

        def emit(text):
            nonlocal first
            if first is None:
                first = time.perf_counter() - started
            delta = split.feed(text)
            stream.push(delta)

        prefill, decode, finish = None, None, 'stop'
        if self.engine == 'mlx':
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
            eos = set(self.tokenizer.eos_token_ids)
            pool = self.conversations
            key = body.get('conversation')
            use_cache = bool(pool and pool.max_bytes and key and body.get('use_cache', True))
            layers = checkpoint = None
            generation_prompt = prompt
            prefix_seconds = 0
            if use_cache:
                layers, reused = pool.prepare(key, prompt)
                prefix_started = time.perf_counter()
                self._prefill_prefix(prompt[reused:-1], layers, body.get('prefill_step_size', 2048))
                prefix_seconds = time.perf_counter() - prefix_started
                if not pool.can_trim(layers):
                    checkpoint = pool.checkpoint(layers, prompt[:-1])
                generation_prompt = prompt[-1:]
                processed = len(prompt) - reused
            last_count = 0
            last = None
            raw_tokens = []
            for row in stream_generate(self.model, self.tokenizer, generation_prompt, max_tokens=limit,
                                       sampler=make_sampler(temp=body.get('temperature', 0), top_p=body.get('top_p', 1), top_k=body.get('top_k', 0)),
                                       prompt_cache=layers,
                                       prefill_step_size=body.get('prefill_step_size', 2048)):
                # The final response can repeat the last token. EOS is excluded
                # from the equivalence hash on both backends.
                if row.generation_tokens > last_count:
                    raw_tokens.append(int(row.token))
                    if row.token not in eos:
                        token_ids.append(int(row.token))
                last_count = row.generation_tokens
                emit(row.text)
                last = row
            if last is None:
                raise ValueError('Aucun token MLX généré.')
            prefill = prefix_seconds + last.prompt_tokens / last.prompt_tps if last.prompt_tps > 0 else None
            decode = last.generation_tokens / last.generation_tps if last.generation_tps > 0 else None
            finish = last.finish_reason or 'stop'
            if use_cache:
                cache_bytes = pool.size_bytes(layers)
                if pool.can_trim(layers):
                    known = prompt + raw_tokens
                    position = pool.position(layers)
                    # SDK versions may pre-evaluate the last generated token or
                    # leave it pending. Never bind unknown/advanced state.
                    if position in (len(known), len(known) - 1):
                        pool.store(key, known[:position], layers)
                elif checkpoint is not None:
                    pool.store(key, checkpoint.tokens, checkpoint.layers)
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
            stream.push(tail, token_progress=False)
        stream.flush()
        elapsed = time.perf_counter() - started
        # Normalize token accounting/rates to non-EOS committed tokens. All
        # generated reasoning tokens remain included. Cold calibration omits a
        # conversation key and follows the unchanged SDK generation path.
        rate = len(token_ids) / decode if decode and decode > 0 else None
        send({'event': 'done', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
              'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(token_ids)},
              'timings': {'predicted_per_second': rate, 'predicted_ms': decode * 1000 if decode is not None else None,
                          'prompt_ms': prefill * 1000 if prefill is not None else None, 'cache_n': reused,
                          'prompt_n': processed, 'kv_cache_bytes': cache_bytes,
                          'first_token_seconds': first, 'request_seconds': elapsed,
                          'peak_gpu_bytes': int(self.mx.get_peak_memory())},
              'conversation_cache': self.cache_summary(),
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
            worker = Worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                            int(sys.argv[5]) if len(sys.argv) > 5 else 0)
            send({'event': 'ready', **probe(), 'conversation_cache': worker.cache_summary()})
            for line in sys.stdin:
                request = json.loads(line)
                try:
                    if request['op'] == 'context':
                        preview, _ = worker.context(request['messages'], request.get('enable_thinking'))
                        send({'event': 'done', 'context': preview})
                    elif request['op'] == 'configure' and worker.engine == 'mlx':
                        capacity = request['context']
                        if type(capacity) is not int or not worker.context_length <= capacity <= 8192:
                            raise ValueError('Capacité MLX invalide.')
                        worker.context_length = capacity
                        worker.mx.set_memory_limit(request['memory_limit'])
                        worker.conversations.resize(request['cache_budget'])
                        send({'event': 'done', 'conversation_cache': worker.cache_summary()})
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
