"""Managed, loopback-only llama.cpp execution and reproducible local calibration.

No model downloads, shell commands, external providers or persisted chat text.
Only child processes created by this instance are stopped by close().
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from .gguf import GGUFReader
from .lmstudio import LMStudioClient
from .loading import model_fingerprint
from .calibration import (TRAIN_PROMPTS, VALIDATION_PROMPTS, CATEGORIES, OUTPUT_LIMITS,
                          TRAIN_PASSES, VALIDATION_PASSES, assess_candidate, summarize,
                          select_winner, memory_plan, shortlist, verified_profiles, USAGE_PROFILES, speculation_summary)


PROTOCOL = 5


@dataclass(frozen=True)
class ExecutionConfig:
    threads: int = 0
    batch: int = 2048
    flash: str = "auto"
    context: int = 4096
    slots: int = 2
    speculative: str = "none"
    draft_path: Optional[str] = None
    draft_tokens: int = 4
    kv_type: str = "f16"
    threads_batch: int = 0
    ubatch: Optional[int] = None
    backend_sampling: bool = False
    ngram_lookup: int = 12
    cache_ram_mib: int = 0

    def __post_init__(self):
        if any(type(value) is not int for value in (self.threads, self.threads_batch, self.batch, self.context, self.slots, self.draft_tokens, self.ngram_lookup)):
            raise ValueError("Execution counts must be integers")
        if not 0 <= self.threads <= 256 or self.batch not in {256, 512, 1024, 2048}:
            raise ValueError("Invalid execution configuration")
        if self.flash not in {"auto", "on", "off"} or not 512 <= self.context <= 32768 or not 1 <= self.slots <= 4:
            raise ValueError("Invalid context or attention configuration")
        if self.kv_type not in {"f16", "q8_0"} or self.draft_tokens not in {2, 4, 8, 16, 32, 48, 64}:
            raise ValueError("Invalid cache precision or speculative depth")
        if self.speculative not in {"none", "ngram-simple", "ngram-map-k", "ngram-mod", "draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark"}:
            raise ValueError("Unsupported speculative configuration")
        if self.speculative.startswith("draft-") != bool(self.draft_path):
            raise ValueError("A draft model is required for this speculative method")
        if (not 0 <= self.threads_batch <= 256 or
                (self.ubatch is not None and (self.ubatch not in {128, 256, 512, 1024, 2048} or self.ubatch > self.batch)) or
                type(self.backend_sampling) is not bool or self.ngram_lookup not in {4, 8, 12, 24} or
                type(self.cache_ram_mib) is not int or not 0 <= self.cache_ram_mib <= 256):
            raise ValueError("Invalid prefill, sampling or RAM cache configuration")


def draft_compatible(target: Path, draft: Path) -> bool:
    """Exact token IDs and special tokens, never display-name or family guesses."""
    target, draft = Path(target), Path(draft)
    if target.resolve() == draft.resolve() or draft.stat().st_size >= target.stat().st_size:
        return False
    a, b = GGUFReader(target).metadata, GGUFReader(draft).metadata
    if b.get('general.architecture') in {'dflash', 'dspark', 'bert', 'nomic-bert'}:
        return False
    keys = ('tokenizer.ggml.model', 'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges', 'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
            'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.add_eos_token')
    return bool(a.get('tokenizer.ggml.tokens')) and all(a.get(k) == b.get(k) for k in keys)



def draft_method(target: Path, draft: Path) -> Optional[str]:
    """Specialized heads require an explicit, hash-bound trained target binding.

    Same vocabulary is insufficient for EAGLE/DFlash/DSpark. The sidecar is a
    user declaration of the upstream training pair, not a quality certificate;
    normal independent output and timing gates still apply.
    """
    target, draft = Path(target), Path(draft)
    if target.resolve() == draft.resolve() or draft.stat().st_size >= target.stat().st_size:
        return None
    sidecar = draft.with_suffix('.local-llm-draft.json')
    if sidecar.is_file():
        if sidecar.stat().st_size > 8192:
            return None
        binding = json.loads(sidecar.read_text())
        if (binding.get('method') not in {'draft-eagle3', 'draft-dflash', 'draft-dspark'} or
                not isinstance(binding.get('source'), str) or not binding['source'].startswith('https://') or
                binding.get('target_sha256') != model_fingerprint(target)[0] or
                binding.get('draft_sha256') != model_fingerprint(draft)[0]):
            return None
        return binding['method']
    return 'draft-simple' if draft_compatible(target, draft) else None



class SlotPool:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.entries = OrderedDict()

    def acquire(self, conversation):
        if conversation in self.entries:
            slot = self.entries.pop(conversation)
            self.entries[conversation] = slot
            return slot, False
        if len(self.entries) < self.capacity:
            slot = next(i for i in range(self.capacity) if i not in self.entries.values())
        else:
            _, slot = self.entries.popitem(last=False)
        self.entries[conversation] = slot
        return slot, True

    def invalidate(self, conversation):
        self.entries.pop(conversation, None)


class Accelerator:
    def __init__(self, executable=None, state_dir=None):
        self.executable = executable or os.environ.get('LOCAL_LLM_LLAMA_SERVER') or shutil.which('llama-server')
        self.state_dir = Path(state_dir or os.environ.get('LOCAL_LLM_STATE_DIR', Path.home() / '.cache/local-llm'))
        self.lock = threading.RLock()
        self.process = self.client = self.log = None
        self.path = self.model_id = self.model_name = None
        self.config = ExecutionConfig()
        self.profile = None
        self.usage_profile = "balanced"
        self.last_profile_switch_seconds = None
        self.memory = None
        self.memory_probe = None
        self.slots = SlotPool(self.config.slots)
        self.job = None
        self.cancelled = threading.Event()
        self.capabilities = None
        self.closed = False

    def available(self):
        if self.capabilities is None:
            if not self.executable:
                self.capabilities = {'available': False, 'error': 'Installez llama.cpp : brew install llama.cpp sur macOS. Puis relancez local-llm.'}
            else:
                try:
                    version = subprocess.check_output([self.executable, '--version'], stderr=subprocess.STDOUT, text=True, timeout=30)
                    help_text = subprocess.check_output([self.executable, '--help'], stderr=subprocess.DEVNULL, text=True, timeout=30)
                    required = ('--cache-ram', '--spec-type', '--no-context-shift', '--no-webui', '--cache-type-k', '--spec-ngram-simple-size-m')
                    if any(flag not in help_text for flag in required):
                        raise ValueError('Version llama.cpp trop ancienne. Mettez-la à jour puis relancez local-llm.')
                    devices = subprocess.check_output([self.executable, '--list-devices'], stderr=subprocess.STDOUT, text=True, timeout=30)
                    version = next((line for line in version.splitlines() if line.startswith('version:')), version.strip())
                    # Free device memory fluctuates; it is not hardware identity.
                    devices = '\n'.join(re.sub(r'\s*\([^)]*(?:memory|free)[^)]*\)', '', line.strip())
                                        for line in devices.splitlines() if ': ' in line and 'srv ' not in line)
                    self.capabilities = {'available': True, 'version': version,
                                         'optional_flags': [flag for flag in ('--backend-sampling', '--spec-ngram-map-k-size-m', '--spec-ngram-mod-n-max') if flag in help_text],
                                         'specialized_methods': [method for method in ('draft-eagle3', 'draft-dflash', 'draft-dspark') if method in help_text],
                                         'devices': devices, 'gpu': any(x in devices for x in ('MTL', 'CUDA', 'Vulkan', 'ROCm', 'SYCL'))}
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    self.capabilities = {'available': False, 'error': str(exc)}
        return dict(self.capabilities)

    def describe(self):
        running = self.process is not None and self.process.poll() is None
        return {**(self.capabilities or {'available': bool(self.executable)}), 'loaded': running,
                'model_id': self.model_id, 'model_name': self.model_name,
                'config': asdict(self.config), 'context_length': self.config.context,
                'profile': self.profile, 'usage_profile': self.usage_profile,
                'profile_switch_seconds': self.last_profile_switch_seconds, 'memory_plan': self.memory, 'cached_conversations': len(self.slots.entries),
                'job': dict(self.job) if self.job else None}

    def _stop(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            else:
                process.wait()
        self.client = None
        if self.log:
            self.log.close()
            self.log = None
        self.slots = SlotPool(self.config.slots)
        self._dirty_conversations = set()

    def close(self):
        self.cancelled.set()
        with self.lock:
            self.closed = True
            self._stop()
            self.path = self.model_id = self.model_name = None
            self.memory = None

    def unload(self):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Arrêtez la calibration avant de décharger le modèle.')
        with self.lock:
            self._stop()
            self.path = self.model_id = self.model_name = None
            self.memory = None
            self.profile = None
            self.job = None

    def _start(self, config):
        self._stop()
        if self.closed:
            raise ValueError('Runtime fermé')
        # A fresh private loopback port; authentication also protects a port race.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        token = secrets.token_hex(32)
        command = [self.executable, '--model', str(self.path), '--alias', self.model_id,
                   '--host', '127.0.0.1', '--port', str(port), '--api-key', token,
                   '--n-gpu-layers', 'all', '--ctx-size', str(config.context * config.slots),
                   '--parallel', str(config.slots), '--batch-size', str(config.batch),
                   '--ubatch-size', str(config.ubatch or min(512, config.batch)), '--flash-attn', config.flash,
                   '--cache-type-k', config.kv_type, '--cache-type-v', config.kv_type, '--cache-ram', str(config.cache_ram_mib),
                   '--no-context-shift', '--no-webui', '--jinja', '--spec-type', config.speculative]
        if config.threads:
            command += ['--threads', str(config.threads)]
        if config.threads_batch:
            command += ['--threads-batch', str(config.threads_batch)]
        if config.backend_sampling:
            command += ['--backend-sampling']
        if config.draft_path:
            command += ['--spec-draft-model', config.draft_path, '--spec-draft-ngl', 'all', '--spec-draft-n-max', str(config.draft_tokens)]
        if config.speculative == 'ngram-simple':
            command += ['--spec-ngram-simple-size-m', str(config.draft_tokens), '--spec-ngram-simple-size-n', str(config.ngram_lookup)]
        elif config.speculative == 'ngram-map-k':
            command += ['--spec-ngram-map-k-size-m', str(config.draft_tokens), '--spec-ngram-map-k-size-n', str(config.ngram_lookup)]
        elif config.speculative == 'ngram-mod':
            command += ['--spec-ngram-mod-n-max', str(config.draft_tokens), '--spec-ngram-mod-n-min', str(min(16, config.draft_tokens)), '--spec-ngram-mod-n-match', str(config.ngram_lookup)]
        self.log = tempfile.TemporaryFile(mode='w+b')
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_') and k != 'LM_STUDIO_API_TOKEN'}
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, env=env)
            self.client = LMStudioClient('http://127.0.0.1:' + str(port), token=token)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(max(0, self.log.tell() - 4096))
                    raise ValueError('Chargement llama.cpp échoué : ' + self.log.read().decode('utf-8', 'replace')[-2000:])
                healthy = False
                try:
                    healthy = self.client._request('/health', timeout=1).get('status') == 'ok'
                except (OSError, ValueError):
                    pass
                if healthy:
                    props = self.client._request('/props')
                    actual_context = props.get('default_generation_settings', {}).get('n_ctx', config.context)
                    if props.get('total_slots', config.slots) != config.slots or actual_context != config.context:
                        raise ValueError('La capacité de contexte du runtime diffère de la configuration demandée.')
                    self.config = config
                    self.slots = SlotPool(config.slots)
                    return
                if self.cancelled.is_set() and self.job and self.job['state'] == 'running':
                    raise ValueError('Calibration interrompue')
                time.sleep(0.1)
            raise ValueError('Délai de chargement llama.cpp dépassé')
        except BaseException:
            self._stop()
            raise

    def load(self, item, memory_available=None):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Calibration en cours')
        if not self.available()['available']:
            raise ValueError(self.available()['error'])
        path = Path(item.path)
        if path.suffix.lower() != '.gguf' or item.architecture in {'dflash', 'dspark', 'bert', 'nomic-bert', None}:
            raise ValueError('Un modèle de discussion GGUF est requis ; les modèles auxiliaires ne répondent pas seuls.')
        with self.lock:
            if self.model_id == item.id and self.process is not None and self.process.poll() is None:
                return self.describe()
            metadata = GGUFReader(path).metadata
            plan = memory_plan(metadata, path.stat().st_size, memory_available)
            self.path, self.model_id, self.model_name = path, item.id, item.name
            self.profile = None
            self.usage_profile = "balanced"
            self.last_profile_switch_seconds = None
            self.job = None
            self.memory = plan
            try:
                self._start(ExecutionConfig(context=plan["context"], slots=plan["slots"], cache_ram_mib=plan.get("cache_ram_mib", 0)))
                self._restore_profile()
            except BaseException:
                self._stop()
                raise
            return self.describe()

    def _restore_profile(self):
        fingerprint, _ = model_fingerprint(self.path)
        source = self.state_dir / (fingerprint + '.json')
        try:
            if not source.is_file() or source.stat().st_size > 2 * 1024 * 1024:
                return
            report = json.loads(source.read_text())
            hardware = {'system': platform.system(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()}
            if (report['protocol'] != PROTOCOL or report['model_sha256'] != fingerprint or
                    report['hardware'] != hardware or report['runtime']['version'] != self.available()['version'] or
                    report['runtime']['devices'] != self.available()['devices'] or
                    report['context_length'] != self.config.context or report['slots'] != self.config.slots):
                return
            config = ExecutionConfig(**report['config'])
            _, summaries = select_winner(report['trials'])
            validation = report['validation']
            profiles = verified_profiles(report['trials'], validation['trials'])
            expected_winner = profiles['balanced']['winner']
            verified = {name: summarize(trial['samples']) for name, trial in validation['trials'].items() if not trial.get('error')}
            if (profiles != report['profiles'] or expected_winner != report['winner'] or
                    summaries != report['training_summaries'] or verified != report['summaries'] or
                    asdict(config) != report['trials'][expected_winner]['config'] or
                    config.cache_ram_mib != self.config.cache_ram_mib or
                    any(len(t['samples']) != VALIDATION_PASSES * len(CATEGORIES) or
                        any(row.get('passes') != VALIDATION_PASSES for row in t['samples'])
                        for t in validation['trials'].values() if not t.get('error'))):
                return
            # Bind every selectable profile, not only the balanced winner.
            for profile in profiles.values():
                item_config = ExecutionConfig(**profile['config'])
                if item_config.draft_path:
                    path = Path(item_config.draft_path)
                    if (draft_method(self.path, path) != item_config.speculative or
                            model_fingerprint(path)[0] != report['draft_fingerprints'].get(str(path))):
                        return
            self._start(config)
            self.profile = report
            self.usage_profile = "balanced"
        except (OSError, ValueError, KeyError, TypeError):
            self.profile = None
            if self.client is not None and self.process is not None and self.process.poll() is None:
                return
            self._start(replace(self.config, threads=0, threads_batch=0, ubatch=None, backend_sampling=False, batch=2048, flash="auto", speculative="none", draft_path=None, kv_type="f16"))

    def set_usage_profile(self, usage):
        if usage not in USAGE_PROFILES:
            raise ValueError('Profil inconnu')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou un chargement est en cours.')
        try:
            if not self.profile or self.job and self.job['state'] == 'running':
                raise ValueError('Calibrez ce modèle avant de choisir un profil.')
            config = ExecutionConfig(**self.profile['profiles'][usage]['config'])
            started = time.perf_counter()
            if config != self.config:
                original = self.config
                try:
                    self._start(config)
                except Exception:
                    self._start(original)
                    raise
            self.usage_profile = usage
            self.last_profile_switch_seconds = time.perf_counter() - started
            return self.describe()
        finally:
            self.lock.release()

    def active_measurement(self):
        return self.profile['profiles'][self.usage_profile] if self.profile else None

    def context(self, messages):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Calibration en cours ; le contexte sera disponible à la fin.')
        with self.lock:
            if self.job and self.job['state'] == 'running':
                raise ValueError('Calibration en cours ; le contexte sera disponible à la fin.')
            if not self.client:
                raise ValueError('Chargez un modèle dans le moteur GPU.')
            prompt = self.client._request('/apply-template', {'messages': messages})['prompt']
            tokens = self.client._request('/tokenize', {'content': prompt, 'add_special': False})['tokens']
            return {'prompt': prompt, 'prompt_tokens': len(tokens), 'context_length': self.config.context,
                    'model': self.model_name, 'compression': 'none', 'kind': 'preview'}

    def iter_chat(self, payload, conversation):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Calibration en cours')
        with self.lock:
            if not self.client or self.process.poll() is not None:
                raise ValueError('Le moteur GPU n’est pas chargé.')
            if payload.get('model') != self.model_id:
                raise ValueError('Le modèle a changé ; renvoyez la requête.')
            preview = self.context(payload['messages'])
            if preview['prompt_tokens'] + payload['max_tokens'] > self.config.context:
                required = preview['prompt_tokens'] + payload['max_tokens']
                available = self.memory_probe() if self.memory_probe else None
                plan = memory_plan(GGUFReader(self.path).metadata, self.path.stat().st_size, available, required)
                original = self.config
                try:
                    self._start(replace(original, context=plan['context'], slots=plan['slots'], cache_ram_mib=plan.get('cache_ram_mib', 0)))
                except Exception:
                    self._start(original)
                    raise
                self.memory = plan
                # Measurements made at another context capacity are not transferable.
                self.profile = None
                self.usage_profile = "balanced"
            slot, reset = self.slots.acquire(conversation)
            # With a bounded host cache, llama.cpp saves/restores exact token-prefix
            # states itself. After an interrupted stream we still force a cold
            # request, so a dirty partial generation cannot masquerade as reused.
            cached = not reset
            if self.config.cache_ram_mib and conversation not in getattr(self, '_dirty_conversations', set()):
                cached = True
            body = dict(payload, cache_prompt=cached, id_slot=slot, repeat_penalty=1.0)
            upstream = self.client.iter_chat(body)
            completed = False
            try:
                for chunk in upstream:
                    yield chunk
                completed = True
                if hasattr(self, '_dirty_conversations'):
                    self._dirty_conversations.discard(conversation)
            finally:
                upstream.close()
                if not completed:
                    self.slots.invalidate(conversation)
                    if not hasattr(self, '_dirty_conversations'):
                        self._dirty_conversations = set()
                    self._dirty_conversations.add(conversation)
                    # The next request assigned this slot uses cache_prompt=False.

    def _rss(self):
        if platform.system() in {'Darwin', 'Linux'} and self.process:
            try:
                return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(self.process.pid)], text=True, timeout=2).strip()) * 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return None

    def _sample(self, prompt, limit=128):
        formatted = self.client._request('/apply-template', {'messages': [{'role': 'user', 'content': prompt}]})['prompt']
        started = time.perf_counter()
        result = self.client._request('/completion', {'prompt': formatted, 'n_predict': limit,
            'temperature': 0, 'seed': 42, 'repeat_penalty': 1.0, 'cache_prompt': False,
            'return_tokens': True, 'id_slot': 0, 'stream': False}, timeout=180)
        timings = result['timings']
        output = json.dumps(result.get('tokens', result['content']), ensure_ascii=False).encode()
        return {'seconds': time.perf_counter() - started, 'decode_tps': timings['predicted_per_second'],
                'prefill_seconds': timings['prompt_ms'] / 1000, 'generated_tokens': timings['predicted_n'],
                'output_sha256': hashlib.sha256(output).hexdigest(), 'process_rss_bytes': self._rss(),
                'timings': timings}

    def optimize(self, draft=None, drafts=None):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou un chargement est en cours.')
        try:
            if not self.model_id or not self.client:
                raise ValueError('Chargez un modèle dans le moteur GPU avant de l’optimiser.')
            if self.job and self.job['state'] == 'running':
                raise ValueError('Calibration déjà en cours')
            candidates = [Path(draft)] if draft else [Path(p) for p in (drafts or [])][:2]
            methods = [draft_method(self.path, p) for p in candidates]
            if any(method is None for method in methods):
                raise ValueError("Modèle auxiliaire incompatible")
            if any(method != 'draft-simple' and method not in self.available().get('specialized_methods', []) for method in methods):
                raise ValueError('La méthode de cet auxiliaire spécialisé n’est pas proposée par le build llama.cpp installé.')
            self.cancelled.clear()
            self.job = {'id': secrets.token_hex(8), 'state': 'running', 'progress': 0,
                        'message': 'Préparation des essais comparables', 'result': None}
            threading.Thread(target=self._calibrate, args=(candidates,), daemon=True).start()
            return dict(self.job)
        finally:
            self.lock.release()

    def _latency_sample(self, prompt, cached):
        started = time.perf_counter()
        first = first_text = None
        timings = {}
        stream = self.client.iter_chat({'model': self.model_id, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 32, 'temperature': 0, 'seed': 42, 'repeat_penalty': 1.0,
            'cache_prompt': cached, 'id_slot': 0})
        try:
            for chunk in stream:
                if self.cancelled.is_set():
                    raise ValueError('Calibration interrompue')
                timings = chunk.get('timings') or timings
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    now = time.perf_counter()
                    if first is None and any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning')):
                        first = now - started
                    if first_text is None and delta.get('content'):
                        first_text = now - started
        finally:
            stream.close()
        return {'first_token_seconds': first, 'first_text_seconds': first_text,
                'seconds': time.perf_counter() - started,
                'prefill_seconds': timings.get('prompt_ms', 0) / 1000 if timings else None,
                'cached_tokens': timings.get('cache_n')}

    def _cache_benchmark(self):
        pairs = []
        for i in range(3):
            prompt = VALIDATION_PROMPTS[2] + '\nComparison run ' + str(i) + ': give a short answer.'
            cold = self._latency_sample(prompt, False)
            warm = self._latency_sample(prompt, True)
            pairs.append({'cold': cold, 'warm': warm})
        import statistics
        def median(field, kind):
            values = [pair[kind][field] for pair in pairs]
            return statistics.median(values) if all(v is not None for v in values) else None
        cold_prefill, warm_prefill = median('prefill_seconds', 'cold'), median('prefill_seconds', 'warm')
        valid = all(isinstance(p['warm']['cached_tokens'], int) and p['warm']['cached_tokens'] > 0 for p in pairs)
        return {'pairs': pairs, 'cache_verified': valid,
                'cold_first_token_seconds': median('first_token_seconds', 'cold'),
                'warm_first_token_seconds': median('first_token_seconds', 'warm'),
                'prefill_seconds_saved': cold_prefill - warm_prefill if valid and cold_prefill is not None and warm_prefill is not None else None,
                'scope': 'same_prompt_cold_then_warm_three_pairs'}

    def _run_trials(self, configs, prompts, passes, phase, trials):
        total = len(configs) * passes * len(prompts)
        done = 0
        for round_index in range(passes):
            # Reverse, then rotate the order to limit systematic order effects.
            order = list(configs) if round_index % 2 == 0 else list(reversed(configs))
            if round_index == 2:
                order = order[1:] + order[:1]
            for name in order:
                if self.cancelled.is_set():
                    raise ValueError('Calibration interrompue')
                if trials[name].get('error'):
                    done += len(prompts)
                    continue
                self.job['message'] = phase + ' · ' + name + ' · passage ' + str(round_index + 1) + '/' + str(passes)
                try:
                    self._start(configs[name])
                    self._sample('Describe a sunny day in a few complete sentences.', 32)
                    for i, prompt in enumerate(prompts):
                        if self.cancelled.is_set():
                            raise ValueError('Calibration interrompue')
                        row = self._sample(prompt, 64 if phase == 'Présélection' else OUTPUT_LIMITS[i])
                        row.update(category=CATEGORIES[i], passes=passes, workload=i)
                        trials[name]['samples'].append(row)
                        done += 1
                        offset, span = {'Présélection': (0, 30), 'Sélection': (30, 30), 'Vérification indépendante': (60, 30)}[phase]
                        self.job['progress'] = round(offset + span * done / total)
                except (OSError, ValueError) as exc:
                    if name == 'standard' or self.cancelled.is_set():
                        raise
                    trials[name]['error'] = str(exc)

    def _candidate_configs(self, original, drafts):
        base = replace(original, threads=0, threads_batch=0, ubatch=None, backend_sampling=False,
                       batch=2048, flash='auto', speculative='none', draft_path=None, kv_type='f16')
        half = max(1, (os.cpu_count() or 4) // 2)
        tuned = replace(base, threads=half, threads_batch=os.cpu_count() or 4, batch=1024, flash='on')
        configs = {'standard': base, 'réglages-1024': tuned, 'réglages-256': replace(tuned, batch=256),
                   'prefill-256': replace(base, ubatch=256), 'prefill-1024': replace(base, ubatch=1024),
                   'threads-décodage': replace(base, threads=half),
                   'threads-préparation': replace(base, threads_batch=half),
                   'cache-q8': replace(tuned, kv_type='q8_0')}
        capabilities = self.available()
        flags = capabilities.get('optional_flags', [])
        if '--backend-sampling' in flags:
            configs['sampling-gpu'] = replace(base, backend_sampling=True)
        for depth in (4, 16, 48):
            configs['motifs-' + str(depth)] = replace(tuned, speculative='ngram-simple', draft_tokens=depth)
        configs['motifs-courts'] = replace(tuned, speculative='ngram-simple', draft_tokens=32, ngram_lookup=4)
        if '--spec-ngram-map-k-size-m' in flags:
            configs['motifs-map-32'] = replace(tuned, speculative='ngram-map-k', draft_tokens=32, ngram_lookup=8)
        if '--spec-ngram-mod-n-max' in flags:
            configs['motifs-adaptatifs-64'] = replace(tuned, speculative='ngram-mod', draft_tokens=64, ngram_lookup=24)
        paths = [drafts] if isinstance(drafts, (str, Path)) else drafts or []
        for index, path in enumerate(paths):
            method = draft_method(self.path, Path(path))
            if not method or method != 'draft-simple' and method not in capabilities.get('specialized_methods', []):
                continue
            for depth in (2, 4, 8, 16):
                configs['auxiliaire-' + str(index + 1) + '-' + str(depth)] = replace(tuned,
                    speculative=method, draft_path=str(path), draft_tokens=depth)
        return configs

    def _calibrate(self, drafts):
        with self.lock:
            original, original_profile, original_usage = self.config, self.profile, self.usage_profile
            chosen, terminal = original, 'complete'
            try:
                initial_fingerprint, _ = model_fingerprint(self.path)
                initial_drafts = {str(path): model_fingerprint(Path(path))[0] for path in (drafts or [])}
                configs = self._candidate_configs(original, drafts)
                base = configs['standard']
                screening = {name: {'config': asdict(config), 'samples': []} for name, config in configs.items()}
                self._run_trials(configs, TRAIN_PROMPTS, 1, 'Présélection', screening)
                names = shortlist(screening)
                configs = {name: configs[name] for name in names}
                trials = {name: {'config': asdict(config), 'samples': []} for name, config in configs.items()}
                self._run_trials(configs, TRAIN_PROMPTS, TRAIN_PASSES, 'Sélection', trials)
                candidate, training_summaries = select_winner(trials)
                decisions = {name: assess_candidate(trials['standard'], trial) for name, trial in trials.items() if name != 'standard'}
                finalists = {'standard': base}
                for category in USAGE_PROFILES.values():
                    name, _ = select_winner(trials, category=category)
                    finalists[name] = configs[name]
                validation = {name: {'config': asdict(config), 'samples': []} for name, config in finalists.items()}
                self._run_trials(finalists, VALIDATION_PROMPTS, VALIDATION_PASSES, 'Vérification indépendante', validation)
                profiles = verified_profiles(trials, validation)
                balanced = profiles['balanced']
                decision, winner = balanced['decision'], balanced['winner']
                if candidate != 'standard':
                    decisions[candidate] = dict(decision, stage='validation indépendante')
                summaries = {name: summarize(trial['samples']) for name, trial in validation.items() if not trial.get('error')}
                chosen = configs[winner]
                self.job.update(progress=90, message='Mesure du premier token et du cache de contexte')
                self._start(chosen)
                cache = self._cache_benchmark()
                fingerprint, _ = model_fingerprint(self.path)
                if fingerprint != initial_fingerprint or any(model_fingerprint(Path(path))[0] != digest for path, digest in initial_drafts.items()):
                    original_profile = None
                    original = replace(original, speculative='none', draft_path=None, kv_type='f16')
                    raise ValueError('Les poids ont changé pendant la calibration ; aucun gain n’est validé.')
                report = {'protocol': PROTOCOL, 'model_sha256': fingerprint, 'model_id': self.model_id,
                          'hardware': {'system': platform.system(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()},
                          'runtime': self.available(), 'context_length': base.context, 'slots': base.slots,
                          'memory_plan': self.memory, 'cache_benchmark': cache, 'draft_search': self.job.get('draft_search'),
                          'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'winner': winner, 'candidate': candidate, 'config': asdict(chosen), 'trials': trials,
                          'training_summaries': training_summaries, 'summaries': summaries, 'decisions': decisions,
                          'screening': screening, 'profiles': profiles,
                          'draft_fingerprints': initial_drafts,
                          'speculation': {name: speculation_summary(trial['samples']) for name, trial in validation.items()},
                          'validation': {'trials': validation, 'decision': decision, 'passes': VALIDATION_PASSES},
                          'draft_sha256': model_fingerprint(Path(chosen.draft_path))[0] if chosen.draft_path else None,
                          'gain_percent': 100 * (summaries['standard']['seconds'] / summaries[winner]['seconds'] - 1),
                          'decode_gain_percent': 100 * (summaries[winner]['decode_tps'] / summaries['standard']['decode_tps'] - 1),
                          'scope': 'independent_chat_code_long_context_3_passes',
                          'kv_precision_changed': chosen.kv_type != base.kv_type,
                          'outputs_identical_on_benchmark': True}
                self.state_dir.mkdir(parents=True, exist_ok=True)
                target = self.state_dir / (fingerprint + '.json')
                fd, filename = tempfile.mkstemp(dir=self.state_dir, suffix='.tmp')
                try:
                    with os.fdopen(fd, 'w') as handle:
                        json.dump(report, handle, ensure_ascii=False, indent=2)
                    os.replace(filename, target)
                finally:
                    if os.path.exists(filename):
                        os.unlink(filename)
                self.profile = report
                self.usage_profile = "balanced"
                self.job.update(progress=100, message='Configuration validée sur des prompts indépendants' if winner != 'standard' else 'La configuration standard reste la meilleure', result=report)
            except Exception as exc:
                self.profile, chosen, self.usage_profile = original_profile, original, original_usage
                terminal = 'cancelled' if self.cancelled.is_set() else 'failed'
                self.job.update(message=str(exc))
            finally:
                self.cancelled.clear()
                try:
                    self._start(chosen)
                except Exception as exc:
                    terminal = 'failed'
                    self.job.update(message='Restauration échouée : ' + str(exc))
                self.slots = SlotPool(self.config.slots)
                self.job['state'] = terminal
