"""Managed, loopback-only llama.cpp execution and reproducible local calibration.

No model downloads, shell commands, external providers or persisted chat text.
Only child processes created by this instance are stopped by close().
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import socket
import statistics
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


PROTOCOL = 3
PROMPTS = (
    "Explain how a bicycle works in detail, using clear complete sentences.",
    "Write a Python function to merge two sorted lists, then explain the algorithm.",
    "Résume les observations suivantes puis propose un plan :\n" + "Une équipe teste une application locale. Elle mesure la vitesse, la mémoire et la stabilité à chaque essai.\n" * 32,
)


@dataclass(frozen=True)
class ExecutionConfig:
    threads: int = 0
    batch: int = 2048
    flash: str = "auto"
    context: int = 4096
    slots: int = 2
    speculative: str = "none"
    draft_path: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.threads <= 256 or self.batch not in {256, 512, 1024, 2048}:
            raise ValueError("Invalid execution configuration")
        if self.flash not in {"auto", "on", "off"} or not 512 <= self.context <= 32768 or self.slots != 2:
            raise ValueError("Invalid context or attention configuration")
        if self.speculative not in {"none", "ngram-simple", "draft-simple"}:
            raise ValueError("Unsupported speculative configuration")
        if (self.speculative == "draft-simple") != bool(self.draft_path):
            raise ValueError("A draft model is required for draft-simple")


def draft_compatible(target: Path, draft: Path) -> bool:
    """Exact token IDs and special tokens, never display-name or family guesses."""
    target, draft = Path(target), Path(draft)
    if target.resolve() == draft.resolve() or draft.stat().st_size >= target.stat().st_size:
        return False
    a, b = GGUFReader(target).metadata, GGUFReader(draft).metadata
    if b.get('general.architecture') in {'dflash', 'bert', 'nomic-bert'}:
        return False
    keys = ('tokenizer.ggml.model', 'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges', 'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
            'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.add_eos_token')
    return bool(a.get('tokenizer.ggml.tokens')) and all(a.get(k) == b.get(k) for k in keys)


def summarize(samples):
    if not samples or any(not math.isfinite(s['seconds']) or s['seconds'] <= 0 or
                          not math.isfinite(s['decode_tps']) or s['decode_tps'] <= 0 for s in samples):
        raise ValueError("Invalid benchmark timings")
    # Compare complete workloads, not a median mixing short and long prompts.
    if len(samples) != 2 * len(PROMPTS):
        raise ValueError('Two complete benchmark passes are required')
    rounds = [samples[r:r + len(PROMPTS)] for r in range(0, len(samples), len(PROMPTS))]
    return {'seconds': statistics.median(sum(s['seconds'] for s in group) for group in rounds),
            'decode_tps': statistics.median(
                sum(max(1, s['generated_tokens'] - 1) for s in group) /
                sum(max(1, s['generated_tokens'] - 1) / s['decode_tps'] for s in group)
                for group in rounds),
            'prefill_seconds': statistics.median(s['prefill_seconds'] for s in samples),
            'process_rss_bytes': max(s.get('process_rss_bytes') or 0 for s in samples)}


def select_winner(trials, baseline_name='standard'):
    """Reject changed outputs and noisy/regressive gains, including speculation."""
    base = trials[baseline_name]
    if len(base['samples']) != 2 * len(PROMPTS):
        raise ValueError('Two complete benchmark passes are required')
    reference = [s['output_sha256'] for s in base['samples']]
    summaries = {baseline_name: summarize(base['samples'])}
    winner = baseline_name
    for name, trial in trials.items():
        if name == baseline_name or [s['output_sha256'] for s in trial['samples']] != reference:
            continue
        try:
            candidate = summarize(trial['samples'])
        except (ValueError, KeyError, TypeError):
            continue
        summaries[name] = candidate
        # Improvement must be present in both repetitions, not just one outlier.
        rounds = [sum(s['seconds'] for s in trial['samples'][r:r + len(PROMPTS)]) /
                  sum(s['seconds'] for s in base['samples'][r:r + len(PROMPTS)])
                  for r in range(0, len(reference), len(PROMPTS))]
        if (all(r < 0.95 for r in rounds) and
                candidate['decode_tps'] >= summaries[baseline_name]['decode_tps'] and
                candidate['seconds'] < summaries[winner]['seconds']):
            winner = name
    return winner, summaries


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
        self.slots = SlotPool()
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
                    required = ('--cache-ram', '--spec-type', '--no-context-shift', '--no-webui')
                    if any(flag not in help_text for flag in required):
                        raise ValueError('Version llama.cpp trop ancienne. Mettez-la à jour puis relancez local-llm.')
                    devices = subprocess.check_output([self.executable, '--list-devices'], stderr=subprocess.STDOUT, text=True, timeout=30)
                    version = next((line for line in version.splitlines() if line.startswith('version:')), version.strip())
                    # Free device memory fluctuates; it is not hardware identity.
                    devices = '\n'.join(re.sub(r'\s*\([^)]*(?:memory|free)[^)]*\)', '', line.strip())
                                        for line in devices.splitlines() if ': ' in line and 'srv ' not in line)
                    self.capabilities = {'available': True, 'version': version,
                                         'devices': devices, 'gpu': any(x in devices for x in ('MTL', 'CUDA', 'Vulkan', 'ROCm', 'SYCL'))}
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    self.capabilities = {'available': False, 'error': str(exc)}
        return dict(self.capabilities)

    def describe(self):
        running = self.process is not None and self.process.poll() is None
        return {**(self.capabilities or {'available': bool(self.executable)}), 'loaded': running,
                'model_id': self.model_id, 'model_name': self.model_name,
                'config': asdict(self.config), 'context_length': self.config.context,
                'profile': self.profile, 'cached_conversations': len(self.slots.entries),
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
        self.slots = SlotPool()

    def close(self):
        self.cancelled.set()
        with self.lock:
            self.closed = True
            self._stop()
            self.path = self.model_id = self.model_name = None

    def unload(self):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Arrêtez la calibration avant de décharger le modèle.')
        with self.lock:
            self._stop()
            self.path = self.model_id = self.model_name = None
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
                   '--ubatch-size', str(512 if config.batch == 2048 else config.batch), '--flash-attn', config.flash,
                   '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--cache-ram', '0',
                   '--no-context-shift', '--no-webui', '--jinja', '--spec-type', config.speculative]
        if config.threads:
            command += ['--threads', str(config.threads), '--threads-batch', str(config.threads)]
        if config.draft_path:
            command += ['--spec-draft-model', config.draft_path, '--spec-draft-ngl', 'all', '--spec-draft-n-max', '4']
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
        if path.suffix.lower() != '.gguf' or item.architecture in {'dflash', 'bert', 'nomic-bert', None}:
            raise ValueError('Un modèle de discussion GGUF est requis ; les modèles auxiliaires ne répondent pas seuls.')
        with self.lock:
            if self.model_id == item.id and self.process is not None and self.process.poll() is None:
                return self.describe()
            if memory_available is not None and path.stat().st_size * 1.2 + 512 * 1024 ** 2 > memory_available:
                raise ValueError('Mémoire disponible insuffisante pour ce modèle et son contexte. Déchargez les modèles inutilisés dans LM Studio ou choisissez un fichier plus petit.')
            self.path, self.model_id, self.model_name = path, item.id, item.name
            self.profile = None
            self.job = None
            context = int(GGUFReader(path).metadata.get(str(item.architecture) + '.context_length', 4096))
            try:
                self._start(ExecutionConfig(context=min(4096, max(512, context))))
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
            valid = {name: trial for name, trial in report['trials'].items()
                     if len(trial['samples']) == 2 * len(PROMPTS) and not trial.get('error')}
            winner, summaries = select_winner(valid)
            if winner != report['winner'] or summaries != report['summaries'] or asdict(config) != valid[winner]['config']:
                return
            if config.draft_path:
                if not draft_compatible(self.path, Path(config.draft_path)):
                    return
                if model_fingerprint(Path(config.draft_path))[0] != report.get('draft_sha256'):
                    return
            self._start(config)
            self.profile = report
        except (OSError, ValueError, KeyError, TypeError):
            self.profile = None
            self._start(ExecutionConfig(context=self.config.context))

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
                raise ValueError('Le contexte dépasse la capacité chargée. Réduisez la longueur de réponse ou créez une conversation.')
            slot, reset = self.slots.acquire(conversation)
            body = dict(payload, cache_prompt=not reset, id_slot=slot, repeat_penalty=1.0)
            upstream = self.client.iter_chat(body)
            completed = False
            try:
                for chunk in upstream:
                    yield chunk
                completed = True
            finally:
                upstream.close()
                if not completed:
                    self.slots.invalidate(conversation)
                    # The next request assigned this slot uses cache_prompt=False.

    def _rss(self):
        if platform.system() in {'Darwin', 'Linux'} and self.process:
            try:
                return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(self.process.pid)], text=True, timeout=2).strip()) * 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return None

    def _sample(self, prompt):
        formatted = self.client._request('/apply-template', {'messages': [{'role': 'user', 'content': prompt}]})['prompt']
        started = time.perf_counter()
        result = self.client._request('/completion', {'prompt': formatted, 'n_predict': 64,
            'temperature': 0, 'seed': 42, 'repeat_penalty': 1.0, 'cache_prompt': False,
            'return_tokens': True, 'id_slot': 0, 'stream': False}, timeout=180)
        timings = result['timings']
        output = json.dumps(result.get('tokens', result['content']), ensure_ascii=False).encode()
        return {'seconds': time.perf_counter() - started, 'decode_tps': timings['predicted_per_second'],
                'prefill_seconds': timings['prompt_ms'] / 1000, 'generated_tokens': timings['predicted_n'],
                'output_sha256': hashlib.sha256(output).hexdigest(), 'process_rss_bytes': self._rss(),
                'timings': timings}

    def optimize(self, draft=None):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou un chargement est en cours.')
        try:
            if not self.model_id or not self.client:
                raise ValueError('Chargez un modèle dans le moteur GPU avant de l’optimiser.')
            if self.job and self.job['state'] == 'running':
                raise ValueError('Calibration déjà en cours')
            if draft and not draft_compatible(self.path, Path(draft)):
                raise ValueError('Modèle auxiliaire incompatible : vocabulaire identique et fichier plus petit requis.')
            self.cancelled.clear()
            self.job = {'id': secrets.token_hex(8), 'state': 'running', 'progress': 0,
                        'message': 'Préparation des essais comparables', 'result': None}
            threading.Thread(target=self._calibrate, args=(draft,), daemon=True).start()
            return dict(self.job)
        finally:
            self.lock.release()

    def _calibrate(self, draft):
        with self.lock:
            original = self.config
            original_profile = self.profile
            chosen = original
            terminal = 'complete'
            try:
                initial_fingerprint, _ = model_fingerprint(self.path)
                base = replace(original, threads=0, batch=2048, flash='auto', speculative='none', draft_path=None)
                tuned = replace(base, threads=max(1, (os.cpu_count() or 4) // 2), batch=1024, flash='on')
                configs = {'standard': base, 'réglages-1024': tuned,
                           'réglages-256': replace(tuned, batch=256),
                           'spéculation-contexte': replace(tuned, speculative='ngram-simple')}
                if draft:
                    configs['modèle-auxiliaire'] = replace(tuned, speculative='draft-simple', draft_path=str(draft))
                trials = {name: {'config': asdict(config), 'samples': []} for name, config in configs.items()}
                total = len(configs) * 2 * len(PROMPTS)
                done = 0
                # A/B then B/A reduces order/thermal bias; every load is warmed up.
                for round_index in range(2):
                    order = list(configs) if round_index == 0 else list(reversed(configs))
                    for name in order:
                        if self.cancelled.is_set():
                            raise ValueError('Calibration interrompue')
                        self.job['message'] = 'Essai ' + name + ' · passage ' + str(round_index + 1) + '/2'
                        try:
                            self._start(configs[name])
                            self._sample('Describe a sunny day in a few complete sentences.')
                            for prompt in PROMPTS:
                                if self.cancelled.is_set():
                                    raise ValueError('Calibration interrompue')
                                trials[name]['samples'].append(self._sample(prompt))
                                done += 1
                                self.job['progress'] = round(100 * done / total)
                        except (OSError, ValueError) as exc:
                            if name == 'standard' or self.cancelled.is_set():
                                raise
                            trials[name]['error'] = str(exc)
                valid = {n: t for n, t in trials.items() if len(t['samples']) == 2 * len(PROMPTS) and not t.get('error')}
                winner, summaries = select_winner(valid)
                chosen = configs[winner]
                fingerprint, _ = model_fingerprint(self.path)
                if fingerprint != initial_fingerprint:
                    original_profile = None
                    original = replace(original, speculative='none', draft_path=None)
                    raise ValueError('Les poids ont changé pendant la calibration ; aucun gain n’est validé.')
                report = {'protocol': PROTOCOL, 'model_sha256': fingerprint, 'model_id': self.model_id,
                          'hardware': {'system': platform.system(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()},
                          'runtime': self.available(), 'context_length': base.context, 'slots': base.slots,
                          'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'winner': winner, 'config': asdict(chosen), 'trials': trials, 'summaries': summaries,
                          'draft_sha256': model_fingerprint(Path(chosen.draft_path))[0] if chosen.draft_path else None,
                          'gain_percent': 100 * (summaries['standard']['seconds'] / summaries[winner]['seconds'] - 1),
                          'decode_gain_percent': 100 * (summaries[winner]['decode_tps'] / summaries['standard']['decode_tps'] - 1),
                          'scope': 'benchmark_local_3_prompts_2_passes', 'outputs_identical_on_benchmark': True}
                # Profiles contain hashes and aggregate timings, never prompts/replies.
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
                self.job.update(progress=100, message='Configuration validée' if winner != 'standard' else 'La configuration standard reste la meilleure', result=report)
            except Exception as exc:
                self.profile = original_profile
                chosen = original
                terminal = 'cancelled' if self.cancelled.is_set() else 'failed'
                self.job.update(message=str(exc))
            finally:
                # Cancellation cannot prevent restoring a working configuration.
                self.cancelled.clear()
                try:
                    self._start(chosen)
                except Exception as exc:
                    terminal = 'failed'
                    self.job.update(message='Restauration échouée : ' + str(exc))
                self.slots = SlotPool()
                self.job['state'] = terminal
