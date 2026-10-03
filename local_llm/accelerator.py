"""Managed, loopback-only llama.cpp execution and reproducible local calibration.

No model downloads or external providers. Local KV snapshots contain chat state.
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
from .discovery import mtp_head_count
from .lmstudio import LMStudioClient
from .loading import model_fingerprint
from .memory import allocations
from .telemetry import macos_memory_pressure
from .recommendations import detect_hardware
from .kv_store import KVStore, digest
from .calibration import (TRAIN_PROMPTS, VALIDATION_PROMPTS, CATEGORIES, OUTPUT_LIMITS,
                          TRAIN_PASSES, VALIDATION_PASSES, assess_candidate, summarize,
                          select_winner, memory_plan, controlled_memory_plan, shortlist, verified_profiles, USAGE_PROFILES, speculation_summary)


PROTOCOL = 7


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
    draft_p_min: float = 0.0

    def __post_init__(self):
        if any(type(value) is not int for value in (self.threads, self.threads_batch, self.batch, self.context, self.slots, self.draft_tokens, self.ngram_lookup)):
            raise ValueError("Execution counts must be integers")
        if not 0 <= self.threads <= 256 or self.batch not in {256, 512, 1024, 2048}:
            raise ValueError("Invalid execution configuration")
        if self.flash not in {"auto", "on", "off"} or not 512 <= self.context <= 32768 or not 1 <= self.slots <= 4:
            raise ValueError("Invalid context or attention configuration")
        if self.kv_type not in {"f16", "q8_0"} or self.draft_tokens not in {2, 3, 4, 6, 8, 12, 16, 32, 48, 64}:
            raise ValueError("Invalid cache precision or speculative depth")
        if self.speculative not in {"none", "ngram-simple", "ngram-map-k", "ngram-mod", "draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark", "draft-mtp"}:
            raise ValueError("Unsupported speculative configuration")
        if (self.speculative.startswith("draft-") and self.speculative != "draft-mtp") != bool(self.draft_path):
            raise ValueError("A draft model is required for this speculative method")
        if type(self.draft_p_min) not in (int, float) or not 0 <= self.draft_p_min <= .95:
            raise ValueError('Invalid speculative confidence threshold')
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
        self.kv_store = KVStore(self.state_dir)
        self.cache_binding = None
        self.runtime_memory = None
        self._dirty_conversations = set()
        self._fingerprints = {}
        self._pressure_stop = threading.Event()
        self._pressure_thread = None
        self._memory_abort = None

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
                                         'optional_flags': [flag for flag in ('--backend-sampling', '--spec-ngram-map-k-size-m', '--spec-ngram-mod-n-max', '--spec-draft-p-min', '--slot-save-path') if flag in help_text],
                                         'specialized_methods': [method for method in ('draft-eagle3', 'draft-dflash', 'draft-dspark', 'draft-mtp') if method in help_text],
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
                'job': dict(self.job) if self.job else None,
                'runtime_memory': self.runtime_memory,
                'persistent_cache': dict(self.kv_store.describe(), supported='--slot-save-path' in (self.capabilities or {}).get('optional_flags', []))}

    def _stop(self):
        self._pressure_stop.set()
        if self._pressure_thread is not None:
            self._pressure_thread.join(timeout=2)
            self._pressure_thread = None
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
        self.cache_binding = None
        self.runtime_memory = None
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
        if config.speculative == 'draft-mtp' and self.memory_probe:
            available = self.memory_probe()
            if available is not None:
                plan = memory_plan(GGUFReader(self.path).metadata, self.path.stat().st_size, available, required=config.context)
                # MTP shares weights but allocates another context and compute buffers.
                needed = (plan['estimated_bytes'] + config.context * plan.get('mtp_kv_bytes_per_token', plan['kv_bytes_per_token']) +
                          plan.get('recurrent_state_bytes', 0) * config.draft_tokens + 256 * 1024 ** 2)
                if needed > available - (plan.get('reserve_bytes') or 0):
                    raise ValueError('RAM disponible insuffisante pour le contexte MTP supplémentaire ; standard conservé.')
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
                   '--no-context-shift', '--no-webui', '--jinja', '--log-verbosity', '4', '--spec-type', config.speculative]
        persistent = '--slot-save-path' in (self.capabilities or self.available()).get('optional_flags', [])
        if persistent:
            try:
                directory = self.kv_store.prepare()
                command += ['--slot-save-path', str(directory)]
            except (OSError, ValueError) as exc:
                persistent = False
                self.kv_store.error = str(exc)
        if config.draft_p_min:
            if '--spec-draft-p-min' not in self.available().get('optional_flags', []):
                raise ValueError('Ce runtime ne permet pas de régler la confiance spéculative.')
            command += ['--spec-draft-p-min', str(config.draft_p_min)]
        if config.threads:
            command += ['--threads', str(config.threads)]
        if config.threads_batch:
            command += ['--threads-batch', str(config.threads_batch)]
        if config.backend_sampling:
            command += ['--backend-sampling']
        if config.draft_path:
            command += ['--spec-draft-model', config.draft_path, '--spec-draft-ngl', 'all', '--spec-draft-n-max', str(config.draft_tokens)]
        if config.speculative == 'draft-mtp':
            command += ['--spec-draft-n-max', str(config.draft_tokens)]
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
            self._memory_abort = None
            if (self.memory or {}).get('controlled_attempt'):
                self._watch_memory(self.process)
            self.client = LMStudioClient('http://127.0.0.1:' + str(port), token=token)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    if self._memory_abort:
                        raise ValueError(self._memory_abort)
                    self.log.seek(max(0, self.log.tell() - 4096))
                    raise ValueError('Chargement llama.cpp échoué : ' + self.log.read().decode('utf-8', 'replace')[-2000:])
                healthy = False
                try:
                    healthy = self.client._request('/health', timeout=1).get('status') == 'ok'
                except (OSError, ValueError):
                    pass
                if healthy:
                    if self._memory_abort:
                        raise ValueError(self._memory_abort)
                    props = self.client._request('/props')
                    actual_context = props.get('default_generation_settings', {}).get('n_ctx', config.context)
                    if props.get('total_slots', config.slots) != config.slots or actual_context != config.context:
                        raise ValueError('La capacité de contexte du runtime diffère de la configuration demandée.')
                    self.config = config
                    self.slots = SlotPool(config.slots)
                    raw = os.pread(self.log.fileno(), min(os.fstat(self.log.fileno()).st_size, 4 * 1024 ** 2), 0) if hasattr(os, 'pread') else b''
                    self.runtime_memory = allocations(raw.decode('utf-8', 'replace'))
                    # Bind once at load, never hash gigabytes per request. File-stat
                    # identity is checked again before each snapshot operation.
                    if persistent and self.path.is_file():
                        fingerprint = self._cached_fingerprint(self.path)
                        draft_sha = self._cached_fingerprint(Path(config.draft_path)) if config.draft_path else None
                        self.cache_binding = digest({'protocol': PROTOCOL, 'model': fingerprint,
                            'draft': draft_sha, 'config': asdict(config), 'runtime': self.available()})
                        self._cache_stat = self.path.stat()
                    return
                if self.cancelled.is_set() and self.job and self.job['state'] == 'running':
                    raise ValueError('Calibration interrompue')
                time.sleep(0.1)
            raise ValueError('Délai de chargement llama.cpp dépassé')
        except BaseException:
            self._stop()
            raise

    def _watch_memory(self, process):
        stop = self._pressure_stop = threading.Event()
        def watch():
            while not stop.is_set() and process.poll() is None:
                pressure = macos_memory_pressure()
                if stop.is_set():
                    return
                if pressure in (None, 'critical'):
                    self._memory_abort = ('Tentative arrêtée : pression mémoire macOS critique.' if pressure else
                                          'Tentative arrêtée : surveillance mémoire macOS indisponible.')
                    # Only this instance's private child; never another engine.
                    if not stop.is_set() and process.poll() is None:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    return
                stop.wait(1)
        self._pressure_thread = threading.Thread(target=watch, daemon=True)
        self._pressure_thread.start()

    def _plan_memory(self, metadata, weight_bytes, available, required=512):
        try:
            return memory_plan(metadata, weight_bytes, available, required)
        except ValueError:
            pressure = macos_memory_pressure()
            if pressure != 'normal':
                raise
            plan = controlled_memory_plan(metadata, weight_bytes, available,
                detect_hardware().get('memory_bytes'), pressure, required)
            if plan is None:
                raise
            return plan

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
            plan = self._plan_memory(metadata, path.stat().st_size, memory_available)
            self.path, self.model_id, self.model_name = path, item.id, item.name
            self.profile = None
            self.usage_profile = "balanced"
            self.last_profile_switch_seconds = None
            self.job = None
            self.memory = plan
            try:
                config = ExecutionConfig(context=plan['context'], slots=plan['slots'], cache_ram_mib=plan.get('cache_ram_mib', 0),
                    batch=256 if plan.get('controlled_attempt') else 2048,
                    ubatch=128 if plan.get('controlled_attempt') else None)
                while True:
                    try:
                        self._start(config)
                        break
                    except ValueError as exc:
                        allocation_failure = any(term in str(exc).lower() for term in ('failed to allocate', 'out of memory', 'insufficient memory'))
                        if not allocation_failure or config.context <= 512:
                            raise
                        # Initial load has no user messages yet. Retry only explicit
                        # allocation failures, at smaller capacity and no host cache.
                        config = replace(config, context=max(512, config.context // 2), cache_ram_mib=0)
                        self.memory = dict(plan, context=config.context, cache_ram_mib=0,
                            context_bytes=config.context * plan['kv_bytes_per_token'],
                            estimated_bytes=plan['estimated_bytes'] - (plan['context'] - config.context) * plan['kv_bytes_per_token'] - plan.get('cache_ram_mib', 0) * 1024 ** 2,
                            fallback_reason='Allocation refusée par le runtime ; capacité réduite avant la première requête.')
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
                    report.get('workload_manifest') != self._manifest(self._benchmark_workloads(VALIDATION_PROMPTS)) or
                    report.get('training_manifest') != self._manifest(self._benchmark_workloads(TRAIN_PROMPTS)) or
                    any(len(t['samples']) != VALIDATION_PASSES * 6 or
                        any(row.get('passes') != VALIDATION_PASSES or row.get('workloads_count') != 6 for row in t['samples'])
                        for t in validation['trials'].values() if not t.get('error'))):
                return
            for name, manifest in (('trials', report['training_manifest']), ('validation', report['workload_manifest'])):
                rows = report['trials'] if name == 'trials' else validation['trials']
                for trial in rows.values():
                    if trial.get('error'):
                        continue
                    for i, sample in enumerate(trial['samples']):
                        expected = manifest[i % 6]
                        if any(sample.get(key) != expected[key] for key in ('category', 'input_tokens', 'output_limit')) or sample.get('workload') != expected['id']:
                            return
            # Bind every selectable profile, not only the balanced winner.
            for profile in profiles.values():
                item_config = ExecutionConfig(**profile['config'])
                if (item_config.context != self.config.context or item_config.slots != self.config.slots or
                        item_config.cache_ram_mib != self.config.cache_ram_mib):
                    return
                if item_config.speculative == 'draft-mtp' and ('draft-mtp' not in self.available().get('specialized_methods', []) or not mtp_head_count(GGUFReader(self.path))):
                    return
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
            self._start(replace(self.config, threads=0, threads_batch=0, ubatch=None, backend_sampling=False, batch=2048, flash="auto", speculative="none", draft_path=None, kv_type="f16", draft_p_min=0.0))

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
                metadata, weight_bytes = GGUFReader(self.path).metadata, self.path.stat().st_size
                plan = self._plan_memory(metadata, weight_bytes, available, required)
                original = self.config
                previous_memory = self.memory
                try:
                    self.memory = plan
                    self._start(replace(original, context=plan['context'], slots=plan['slots'], cache_ram_mib=plan.get('cache_ram_mib', 0),
                        batch=256 if plan.get('controlled_attempt') else original.batch,
                        ubatch=128 if plan.get('controlled_attempt') else original.ubatch))
                except Exception:
                    self.memory = previous_memory
                    self._start(original)
                    raise
                self.memory = plan
                # Measurements made at another context capacity are not transferable.
                self.profile = None
                self.usage_profile = "balanced"
            slot, reset = self.slots.acquire(conversation)
            restored = False
            if reset and conversation not in self._dirty_conversations and self._cache_compatible():
                restored = self.kv_store.restore(self.client, slot, conversation, self.cache_binding, payload['messages'])
            else:
                self.kv_store.last = {'restored': False, 'restore_seconds': None}
            # With a bounded host cache, llama.cpp saves/restores exact token-prefix
            # states itself. After an interrupted stream we still force a cold
            # request, so a dirty partial generation cannot masquerade as reused.
            cached = not reset or restored
            if self.config.cache_ram_mib and conversation not in getattr(self, '_dirty_conversations', set()):
                cached = True
            body = dict(payload, cache_prompt=cached, id_slot=slot, repeat_penalty=1.0)
            upstream = self.client.iter_chat(body)
            completed = False
            timings = {}
            try:
                for chunk in upstream:
                    if self._memory_abort:
                        raise ValueError(self._memory_abort)
                    timings = chunk.get('timings') or timings
                    yield chunk
                completed = True
                if hasattr(self, '_dirty_conversations'):
                    self._dirty_conversations.discard(conversation)
                if self._cache_compatible():
                    tokens = preview['prompt_tokens'] + payload['max_tokens']
                    expected = (self.memory or {}).get('kv_bytes_per_token', 256 * 1024) * tokens + (self.memory or {}).get('recurrent_budget_bytes', 0)
                    if expected <= self.kv_store.budget:
                        self.kv_store.save(self.client, slot, conversation, self.cache_binding, payload['messages'], timings)
            except Exception:
                if self._memory_abort:
                    raise ValueError(self._memory_abort) from None
                raise
            finally:
                upstream.close()
                if not completed:
                    if self.cache_binding:
                        self.kv_store.remove(conversation, self.cache_binding)
                    self.slots.invalidate(conversation)
                    if not hasattr(self, '_dirty_conversations'):
                        self._dirty_conversations = set()
                    self._dirty_conversations.add(conversation)
                    # The next request assigned this slot uses cache_prompt=False.

    def _cached_fingerprint(self, path):
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
        if identity not in self._fingerprints:
            self._fingerprints[identity] = model_fingerprint(path)[0]
        return self._fingerprints[identity]

    def _cache_compatible(self):
        if not self.cache_binding or not self.path:
            return False
        try:
            stat = self.path.stat()
        except OSError:
            return False
        old = self._cache_stat
        return (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns) == (old.st_size, old.st_mtime_ns, old.st_ino, old.st_ctime_ns)

    def configure_cache(self, enabled=None, clear=False):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou calibration est en cours.')
        try:
            self.kv_store.configure(enabled, clear)
            return self.describe()
        finally:
            self.lock.release()

    def _rss(self):
        if platform.system() in {'Darwin', 'Linux'} and self.process:
            try:
                return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(self.process.pid)], text=True, timeout=2).strip()) * 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return None

    def _sample(self, prompt, limit=128):
        formatted = self.client._request('/apply-template', {'messages': [{'role': 'user', 'content': prompt}]})['prompt']
        count = len(self.client._request('/tokenize', {'content': formatted, 'add_special': False})['tokens'])
        if count + limit > self.config.context:
            raise ValueError('Le workload public dépasse le contexte ; aucune troncature implicite.')
        started = time.perf_counter()
        first, timings, tokens = None, {}, []
        # Sample RSS during generation; it is a sampled process peak, not exclusive RAM.
        stop, peak = threading.Event(), [self._rss()]
        def monitor():
            while not stop.wait(.1):
                value = self._rss()
                if value is not None:
                    peak[0] = max(value, peak[0] or 0)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        upstream = self.client.iter_completion({'prompt': formatted, 'n_predict': limit,
            'temperature': 0, 'seed': 42, 'repeat_penalty': 1.0, 'cache_prompt': False,
            'return_tokens': True, 'id_slot': 0, 'stream': True})
        try:
            for chunk in upstream:
                if self.cancelled.is_set():
                    raise ValueError('Calibration interrompue')
                if first is None and (chunk.get('tokens') or chunk.get('content')):
                    first = time.perf_counter() - started
                tokens.extend(chunk.get('tokens') or [])
                timings = chunk.get('timings') or timings
            finished = time.perf_counter()
        finally:
            upstream.close()
            stop.set()
            thread.join(timeout=3)
        elapsed = finished - started
        if not tokens or first is None or not timings.get('predicted_n'):
            raise ValueError('Le runtime ne fournit pas de génération mesurable avec IDs de tokens.')
        output = json.dumps(tokens).encode()
        return {'seconds': elapsed, 'decode_tps': timings['predicted_per_second'],
                'prefill_seconds': timings['prompt_ms'] / 1000, 'generated_tokens': timings['predicted_n'],
                'first_token_seconds': first, 'input_tokens': count, 'output_limit': limit,
                'output_sha256': hashlib.sha256(output).hexdigest(), 'process_rss_bytes': self._rss(),
                'process_rss_peak_bytes': max(peak[0] or 0, self._rss() or 0) or None, 'timings': timings}

    def _benchmark_workloads(self, prompts):
        # Two input/output shapes per category; public data only. Synthetic log
        # length is bounded explicitly against this model's actual tokenizer.
        extended = (
            prompts[0] + '\nWrite a detailed practical guide with examples, tradeoffs and a checklist.',
            prompts[1] + '\nInclude tests, document invariants, and discuss alternative implementations in detail.',
            prompts[2] + '\n' + '\n'.join('Additional observation %d: queue %d, memory %d MiB, latency %.2f seconds.' %
                (i, i % 9, 800 + i * 13, .1 + i % 7 * .15) for i in range(96)))
        workloads = []
        for i, prompt in enumerate(tuple(prompts) + extended):
            limit = min((128, 192, 128, 512, 512, 256)[i], self.config.context // 4)
            lines = prompt.splitlines()
            while True:
                formatted = self.client._request('/apply-template', {'messages': [{'role': 'user', 'content': '\n'.join(lines)}]})['prompt']
                count = len(self.client._request('/tokenize', {'content': formatted, 'add_special': False})['tokens'])
                if count + limit <= self.config.context:
                    break
                if len(lines) <= 2:
                    raise ValueError('Contexte trop petit pour la suite de calibration.')
                lines.pop()
            workloads.append({'id': i, 'category': CATEGORIES[i % 3], 'prompt': '\n'.join(lines),
                              'input_tokens': count, 'output_limit': limit})
        return workloads

    @staticmethod
    def _manifest(workloads):
        return [dict(id=w['id'], category=w['category'], input_tokens=w['input_tokens'],
                     output_limit=w['output_limit'], prompt_sha256=hashlib.sha256(w['prompt'].encode()).hexdigest()) for w in workloads]

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
                        workload = prompt if isinstance(prompt, dict) else {'prompt': prompt, 'id': i, 'category': CATEGORIES[i], 'output_limit': 64}
                        row = self._sample(workload['prompt'], workload['output_limit'])
                        row.update(category=workload['category'], passes=passes, workload=workload['id'], workloads_count=len(prompts))
                        trials[name]['samples'].append(row)
                        done += 1
                        offset, span = {'Présélection': (0, 30), 'Sélection': (30, 30), 'Vérification indépendante': (60, 30)}[phase]
                        self.job['progress'] = round(offset + span * done / total)
                except (OSError, ValueError) as exc:
                    if name == 'standard' or self.cancelled.is_set():
                        raise
                    trials[name]['error'] = str(exc)

    def _candidate_configs(self, original, drafts):
        compact = (self.memory or {}).get('controlled_attempt')
        base = replace(original, threads=0, threads_batch=0, ubatch=None, backend_sampling=False,
                       batch=2048, flash='auto', speculative='none', draft_path=None, kv_type='f16', draft_p_min=0.0)
        if compact:
            base = replace(base, batch=256, ubatch=128, cache_ram_mib=0)
        half = max(1, (os.cpu_count() or 4) // 2)
        tuned = replace(base, threads=half, threads_batch=os.cpu_count() or 4, batch=1024, flash='on')
        configs = {'standard': base, 'réglages-1024': tuned, 'réglages-256': replace(tuned, batch=256),
                   'prefill-256': replace(base, ubatch=256), 'prefill-1024': replace(base, batch=max(1024, base.batch), ubatch=1024),
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
        if self.path and 'draft-mtp' in capabilities.get('specialized_methods', []) and mtp_head_count(GGUFReader(self.path)):
            for depth in (2, 3, 4, 6, 8, 12, 16):
                configs['mtp-intégré-' + str(depth)] = replace(tuned, speculative='draft-mtp', draft_tokens=depth)
            if '--spec-draft-p-min' in flags:
                for threshold in (.5, .8):
                    configs['mtp-confiance-' + str(threshold)] = replace(tuned, speculative='draft-mtp', draft_tokens=12, draft_p_min=threshold)
        paths = [drafts] if isinstance(drafts, (str, Path)) else drafts or []
        for index, path in enumerate(paths):
            method = draft_method(self.path, Path(path))
            if not method or method != 'draft-simple' and method not in capabilities.get('specialized_methods', []):
                continue
            for depth in (2, 3, 4, 6, 8, 12, 16):
                configs['auxiliaire-' + str(index + 1) + '-' + str(depth)] = replace(tuned,
                    speculative=method, draft_path=str(path), draft_tokens=depth)
            if '--spec-draft-p-min' in flags and method == 'draft-simple':
                configs['auxiliaire-' + str(index + 1) + '-confiance'] = replace(tuned, speculative=method, draft_path=str(path), draft_tokens=12, draft_p_min=.8)
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
                training_workloads = self._benchmark_workloads(TRAIN_PROMPTS)
                validation_workloads = self._benchmark_workloads(VALIDATION_PROMPTS)
                screening_workloads = [dict(w, output_limit=min(w['output_limit'], 128)) for w in training_workloads]
                screening = {name: {'config': asdict(config), 'samples': []} for name, config in configs.items()}
                self._run_trials(configs, screening_workloads, 1, 'Présélection', screening)
                names = shortlist(screening)
                configs = {name: configs[name] for name in names}
                trials = {name: {'config': asdict(config), 'samples': []} for name, config in configs.items()}
                self._run_trials(configs, training_workloads, TRAIN_PASSES, 'Sélection', trials)
                candidate, training_summaries = select_winner(trials)
                decisions = {name: assess_candidate(trials['standard'], trial) for name, trial in trials.items() if name != 'standard'}
                finalists = {'standard': base}
                for category in USAGE_PROFILES.values():
                    name, _ = select_winner(trials, category=category)
                    finalists[name] = configs[name]
                validation = {name: {'config': asdict(config), 'samples': []} for name, config in finalists.items()}
                self._run_trials(finalists, validation_workloads, VALIDATION_PASSES, 'Vérification indépendante', validation)
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
                    original = replace(original, speculative='none', draft_path=None, kv_type='f16', draft_p_min=0.0)
                    raise ValueError('Les poids ont changé pendant la calibration ; aucun gain n’est validé.')
                report = {'protocol': PROTOCOL, 'model_sha256': fingerprint, 'model_id': self.model_id,
                          'hardware': {'system': platform.system(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()},
                          'runtime': self.available(), 'context_length': base.context, 'slots': base.slots,
                          'memory_plan': self.memory, 'runtime_memory': self.runtime_memory,
                          'workload_manifest': self._manifest(validation_workloads),
                          'training_manifest': self._manifest(training_workloads), 'cache_benchmark': cache, 'draft_search': self.job.get('draft_search'),
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
                          'scope': 'independent_6_workloads_3_passes',
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
                self.job.update(progress=100, message='Configuration validée sur des prompts indépendants' if winner != 'standard' else 'Aucun gain validé sur la suite complète ; réglage standard conservé.', result=report)
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
