"""Managed MLX/MTPLX inference and conservative, same-artifact selection.

Workers keep weights resident until unload. They are owned children, use an
existing environment and never start the app server or modify MTPLX settings.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

from .calibration import (TRAIN_PROMPTS, VALIDATION_PROMPTS, OUTPUT_LIMITS, CATEGORIES,
                          TRAIN_PASSES, VALIDATION_PASSES, USAGE_PROFILES, verified_profiles,
                          summarize, assess_candidate)
from .loading import model_fingerprint
from .mlx_experiment import validate_model
from .telemetry import macos_memory_pressure
from .recommendations import detect_hardware


MAC_PROTOCOL = 2


def worker_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HF_', 'TRANSFORMERS_', 'MLX_', 'MTPLX_'))
           and k != 'LM_STUDIO_API_TOKEN'}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               PYTHONPYCACHEPREFIX=str(Path(tempfile.gettempdir()) / 'local-llm-mac-bytecode'),
               PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    return env


def interpreter_candidates():
    explicit = os.environ.get('LOCAL_LLM_MLX_PYTHON')
    if explicit:
        return [Path(explicit).expanduser()]
    return list(dict.fromkeys([Path(sys.executable),
        Path.home() / 'Library/Application Support/MTPLX/runtime-venv/bin/python',
        Path.home() / '.mtplx/venv/bin/python',
        Path.home() / '.local/share/uv/tools/mtplx/bin/python']))


def model_engines(path, capability):
    if not capability.get('available'):
        return []
    try:
        root = validate_model(path)
        config = json.loads((root / 'config.json').read_text())
    except (OSError, ValueError, TypeError):
        return []
    # Candidate status, never a successful-load claim. The worker checks the
    # installed implementation and refuses unsupported tensor layouts.
    engines = []
    if config.get('model_type') != 'prism_hadamard_qwen35':
        engines.append('mlx')
    if 'mtplx' in capability.get('packages', {}) and (
            (root / 'mtp.safetensors').is_file() or config.get('mtplx_mtp_contract')):
        engines.append('mtplx')
    return engines


def mac_memory_plan(root, available, total, context=4096):
    """Upper estimate for weights, attention KV and hybrid recurrent state."""
    config = json.loads((root / 'config.json').read_text())
    text = config.get('text_config') or config
    if not isinstance(text, dict):
        raise ValueError('Configuration de contexte MLX invalide.')
    def positive(name, default):
        value = text.get(name, default)
        return value if type(value) is int and value > 0 else default
    context = min(context, positive('max_position_embeddings', 4096), 8192)
    weights = sum(p.stat().st_size for p in root.glob('*.safetensors'))
    layers = positive('num_hidden_layers', 32)
    heads = positive('num_key_value_heads', positive('num_attention_heads', 32))
    head_dim = positive('head_dim', positive('hidden_size', 4096) // positive('num_attention_heads', 32))
    types = text.get('layer_types')
    full_layers = types.count('full_attention') if isinstance(types, list) and len(types) == layers else layers
    recurrent_layers = layers - full_layers
    kv = context * full_layers * heads * head_dim * 2 * 2
    recurrent = recurrent_layers * positive('linear_num_value_heads', 32) * positive('linear_key_head_dim', 128) * positive('linear_value_head_dim', 128) * 4 * 4
    compute = max(512 * 1024 ** 2, int(weights * .15))
    estimated = weights + kv + recurrent + compute
    reserve = 2 * 1024 ** 3
    if available is None or total is None:
        raise ValueError('RAM disponible inconnue : le chargement MLX attend une mesure système fiable.')
    if estimated + reserve > total or estimated > available:
        raise ValueError('RAM disponible insuffisante pour cet essai MLX : %.1f Gio disponibles, %.1f Gio estimés. Déchargez les modèles ouverts dans un autre moteur.' %
                         (available / 1024 ** 3, estimated / 1024 ** 3))
    cache_budget = min(256 * 1024 ** 2, max(0, min(available, total - reserve) - estimated) // 4)
    return {'context': context, 'estimated_bytes': estimated + cache_budget, 'available_bytes': available,
            'weight_bytes': weights, 'context_bytes': kv, 'recurrent_budget_bytes': recurrent,
            'compute_budget_bytes': compute, 'reserve_bytes': reserve,
            'conversation_cache_budget_bytes': cache_budget,
            'memory_limit_bytes': min(total - reserve, available), 'kind': 'estimate'}


@dataclass(frozen=True)
class MacConfig:
    engine: str = 'mlx'
    context: int = 4096
    prefill_step_size: int = 2048
    depth: int = 0
    speculative: str = 'none'

    def __post_init__(self):
        if (self.engine not in {'mlx', 'mtplx'} or type(self.context) is not int or not 512 <= self.context <= 8192
                or self.prefill_step_size not in {128, 512, 2048} or type(self.prefill_step_size) is not int
                or type(self.depth) is not int or not 0 <= self.depth <= 3
                or self.engine == 'mlx' and self.depth != 0
                or self.speculative != ('draft-mtp' if self.depth else 'none')):
            raise ValueError('Configuration du moteur Mac invalide.')


class MacRuntime:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.process = self.log = None
        self.model_id = self.model_name = self.path = None
        self.config = MacConfig()
        self.profile = self.job = self.memory = None
        self.memory_probe = None
        self.usage_profile = 'balanced'
        self.capability = None
        self.events = queue.Queue()
        self._reader = self._pressure_thread = None
        self._pressure_stop = threading.Event()
        self._memory_abort = None
        self.last_profile_switch_seconds = None
        self.runtime_memory = None
        self.cache_state = {'supported': False, 'entries': 0, 'bytes': 0}
        self.last_request_context = None
        self.closed = False
        self.slots = type('NoSlots', (), {'entries': {}})()

    def available(self):
        if self.capability is None:
            self.capability = {'available': False, 'error': 'Un environnement Python 3.11+ avec mlx-lm est requis. Configurez LOCAL_LLM_MLX_PYTHON, ou installez MTPLX séparément.'}
            if platform.system() != 'Darwin' or platform.machine() != 'arm64':
                return dict(self.capability)
            for python in interpreter_candidates():
                if not python.is_file():
                    continue
                try:
                    result = subprocess.run([str(python), '-m', 'local_llm.mac_worker', '--probe'],
                                            capture_output=True, text=True, timeout=10, env=worker_environment())
                    value = json.loads(result.stdout) if result.returncode == 0 else {}
                    if value.get('available'):
                        self.capability = dict(value, interpreter=str(python), gpu=True)
                        break
                except (OSError, ValueError, subprocess.SubprocessError):
                    continue
        return dict(self.capability)

    def describe(self):
        loaded = self.process is not None and self.process.poll() is None
        cache = self.cache_state if loaded else {'supported': False, 'entries': 0, 'bytes': 0}
        return {**self.available(), 'engine': self.config.engine, 'loaded': loaded,
                'model_id': self.model_id, 'model_name': self.model_name, 'config': asdict(self.config),
                'context_length': self.config.context, 'profile': self.profile, 'usage_profile': self.usage_profile,
                'profile_switch_seconds': self.last_profile_switch_seconds, 'memory_plan': self.memory,
                'cached_conversations': cache['entries'], 'conversation_cache': dict(cache),
                'job': dict(self.job) if self.job else None,
                'runtime_memory': self.runtime_memory, 'persistent_cache': {'supported': False},
                'attribution': 'Powered by MTPLX · Youssof Altoukhi' if self.config.engine == 'mtplx' else 'MLX-LM · Apple',
                'scope': 'MTPLX Python · Sustained' if self.config.engine == 'mtplx' else 'MLX-LM · génération standard'}

    def _stop(self):
        self._pressure_stop.set()
        process, self.process = self.process, None
        self.cache_state = {'supported': False, 'entries': 0, 'bytes': 0}
        self.last_request_context = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        for thread in (self._reader, self._pressure_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)
        self._reader = self._pressure_thread = None
        if self.log:
            self.log.close()
            self.log = None

    def close(self):
        self.closed = True
        self.cancelled.set()
        with self.lock:
            self._stop()

    def unload(self):
        if self.job and self.job.get('state') == 'running':
            raise ValueError('Comparaison en cours.')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou une comparaison est en cours.')
        try:
            self._stop()
            self.model_id = self.model_name = self.path = self.profile = None
            return self.describe()
        finally:
            self.lock.release()

    def _start(self, config):
        self._stop()
        if self.closed:
            raise ValueError('Le moteur Mac est fermé.')
        if macos_memory_pressure() != 'normal':
            raise ValueError('La pression mémoire du Mac est élevée ou inconnue. Libérez de la RAM avant cet essai.')
        if self.memory_probe is not None:
            self.memory = mac_memory_plan(self.path, self.memory_probe(),
                                          detect_hardware().get('memory_bytes'), config.context)
        self._memory_abort = None
        self._pressure_stop = threading.Event()
        self.config = config
        self.events = queue.Queue()
        self.log = tempfile.TemporaryFile(mode='w+b')
        self.process = subprocess.Popen([self.available()['interpreter'], '-m', 'local_llm.mac_worker', str(self.path),
            config.engine, str(config.context), str(self.memory['memory_limit_bytes']),
            str(self.memory.get('conversation_cache_budget_bytes', 0) if config.engine == 'mlx' else 0)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=worker_environment())
        process, events = self.process, self.events
        def read():
            try:
                while True:
                    line = process.stdout.readline(8 * 1024 ** 2 + 1)
                    if not line:
                        break
                    if len(line) > 8 * 1024 ** 2:
                        raise ValueError('Réponse du worker trop volumineuse.')
                    events.put(json.loads(line))
            except (OSError, ValueError) as exc:
                events.put({'event': 'error', 'error': str(exc)})
            finally:
                events.put({'event': 'error', 'error': 'Le worker Mac s’est arrêté.'})
        self._reader = threading.Thread(target=read, daemon=True)
        self._reader.start()
        def pressure():
            while not self._pressure_stop.wait(1):
                if macos_memory_pressure() != 'normal':
                    self._memory_abort = 'Essai arrêté : pression mémoire élevée ou capteur indisponible.'
                    if process.poll() is None:
                        process.terminate()
                    break
        self._pressure_thread = threading.Thread(target=pressure, daemon=True)
        self._pressure_thread.start()
        try:
            if self._next(timeout=180).get('event') != 'ready':
                raise ValueError('Chargement du worker non confirmé.')
        except BaseException:
            self._stop()
            raise

    def _next(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._memory_abort:
                raise ValueError(self._memory_abort)
            if self.job and self.job.get('state') == 'running' and self.cancelled.is_set():
                raise ValueError('Comparaison interrompue.')
            try:
                event = self.events.get(timeout=.2)
            except queue.Empty:
                continue
            if event.get('event') == 'error':
                raise ValueError(event.get('error', 'Erreur du worker Mac'))
            if event.get('conversation_cache'):
                self.cache_state = dict(event['conversation_cache'])
            return event
        self._stop()
        raise ValueError('Le worker Mac n’a pas répondu dans le délai prévu.')

    def _request(self, body):
        if self.process is None or self.process.poll() is not None:
            self._stop()
            raise ValueError('Le moteur Mac n’est plus chargé. Rechargez le modèle sélectionné.')
        completed = False
        try:
            self.process.stdin.write(json.dumps(body).encode() + b'\n')
            self.process.stdin.flush()
            while True:
                event = self._next()
                completed = event['event'] == 'done'
                yield event
                if completed:
                    return
        finally:
            # An interrupted generator must not leave an inference running or
            # let its late frames become the next request's answer.
            if not completed:
                self._stop()

    def load(self, item, available=None, engine='auto'):
        if self.closed or self.job and self.job.get('state') == 'running':
            raise ValueError('Le moteur Mac est fermé ou occupé.')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou comparaison est en cours.')
        try:
            root = validate_model(item.path)
            candidates = model_engines(root, self.available())
            if engine != 'auto' and engine not in candidates or not candidates:
                raise ValueError('Ce moteur ne peut pas charger ce checkpoint local.')
            total = detect_hardware().get('memory_bytes')
            plan = mac_memory_plan(root, available, total)
            chosen = candidates[0] if engine == 'auto' else engine
            self._stop()
            self.path, self.model_id, self.model_name = root, item.id, item.name
            self.memory, self.profile, self.job, self.usage_profile = plan, None, None, 'balanced'
            self.cancelled.clear()
            self._start(MacConfig(engine=chosen, context=plan['context']))
            if engine == 'auto':
                self._restore_profile()
            return self.describe()
        finally:
            self.lock.release()

    def context(self, messages, thinking=None):
        with self.lock:
            frames = self._request({'op': 'context', 'messages': messages, 'enable_thinking': thinking})
            try:
                result = next(frames)['context']
                # Consume completion before closing, so a completed request is
                # not mistaken for cancellation by the generator's finally.
                next(frames, None)
            finally:
                frames.close()
            return dict(result, model=self.model_name)

    def iter_chat(self, payload, conversation):
        if self.job and self.job['state'] == 'running':
            raise ValueError('Comparaison en cours.')
        with self.lock:
            if payload.get('model') != self.model_id:
                raise ValueError('Le modèle a changé ; renvoyez la requête.')
            if self.config.engine == 'mlx':
                preview = self.context(payload['messages'], payload.get('enable_thinking'))
                self._ensure_capacity(preview['prompt_tokens'] + payload['max_tokens'])
                self.last_request_context = dict(preview, context_length=self.config.context)
            upstream = self._request(dict(payload, op='generate', conversation=conversation,
                prefill_step_size=self.config.prefill_step_size, depth=self.config.depth))
            try:
                for chunk in upstream:
                    chunk = dict(chunk)
                    chunk.pop('sample', None)
                    chunk.pop('event', None)
                    yield dict(chunk, model=self.model_id)
            finally:
                upstream.close()

    def _ensure_capacity(self, required):
        """Grow MLX context in place; never reload weights or shorten messages."""
        if required <= self.config.context:
            return
        if required > 8192:
            raise ValueError('Le contexte et la réponse demandent %d tokens ; la capacité MLX actuelle est limitée à 8192. Réduisez la longueur de réponse ou ouvrez une nouvelle conversation.' % required)
        if self.memory_probe is None or macos_memory_pressure() != 'normal':
            raise ValueError('Une mesure de RAM disponible et une pression mémoire normale sont requises pour agrandir le contexte MLX.')
        capacity = min(8192, 1 << (required - 1).bit_length())
        plan = mac_memory_plan(self.path, self.memory_probe(), detect_hardware().get('memory_bytes'), capacity)
        if plan['context'] < required:
            raise ValueError('La capacité du modèle est de %d tokens ; %d sont requis. Aucun message n’a été supprimé.' % (plan['context'], required))
        budget = min(self.memory.get('conversation_cache_budget_bytes', 0), plan['conversation_cache_budget_bytes'])
        plan['conversation_cache_budget_bytes'] = budget
        frames = self._request({'op': 'configure', 'context': plan['context'],
                                'memory_limit': plan['memory_limit_bytes'], 'cache_budget': budget})
        try:
            for frame in frames:
                if frame['event'] != 'done':
                    raise ValueError('Modification du contexte non confirmée.')
        finally:
            frames.close()
        self.config = replace(self.config, context=plan['context'])
        self.memory, self.profile, self.usage_profile = plan, None, 'balanced'

    def _rss(self):
        if self.process:
            try:
                return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(self.process.pid)], text=True, timeout=2).strip()) * 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return None

    def active_measurement(self):
        return self.profile['profiles'][self.usage_profile] if self.profile else None

    def configure_cache(self, enabled=None, clear=False):
        raise ValueError('Le cache disque n’est pas encore pris en charge par ce moteur Mac.')

    def set_usage_profile(self, usage):
        if usage not in USAGE_PROFILES:
            raise ValueError('Profil inconnu.')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Une génération ou comparaison est en cours.')
        try:
            if self.profile is None:
                raise ValueError('Comparez les moteurs avant de choisir un profil.')
            started = time.perf_counter()
            self._start(MacConfig(**self.profile['profiles'][usage]['config']))
            self.usage_profile = usage
            self.last_profile_switch_seconds = time.perf_counter() - started
            return self.describe()
        finally:
            self.lock.release()

    def _binding(self):
        info = self.available()
        hardware = detect_hardware()
        return {'protocol': MAC_PROTOCOL, 'model_sha256': model_fingerprint(self.path)[0],
                'runtime': {'packages': info['packages'], 'python': info['python']},
                'hardware': {key: hardware.get(key) for key in ('cpu', 'logical_cores', 'system', 'architecture', 'memory_bytes')},
                'context_length': self.config.context,
                'scope': 'same_mlx_artifact_sustained_cold_greedy', 'cache': 'cold',
                'sampling': {'temperature': 0, 'top_k': 0, 'top_p': 1, 'seed': 42, 'enable_thinking': False}}

    def _manifest(self, prompts):
        result = []
        for i, prompt in enumerate(prompts):
            frames = self._request({'op': 'context', 'messages': [{'role': 'user', 'content': prompt}], 'enable_thinking': False})
            try:
                preview = next(frames)['context']
                next(frames, None)
            finally:
                frames.close()
            result.append({'workload': i, 'category': CATEGORIES[i], 'output_limit': OUTPUT_LIMITS[i],
                           'input_tokens': preview['prompt_tokens'], 'prompt_sha256': preview['prompt_sha256']})
        return result

    def _candidate_configs(self, context):
        engines = model_engines(self.path, self.available())
        baseline = 'mlx' if 'mlx' in engines else 'mtplx'
        configs = {'standard': MacConfig(engine=baseline, context=context)}
        if baseline == 'mlx':
            configs['mlx-prefill-512'] = MacConfig(context=context, prefill_step_size=512)
        if 'mtplx' in engines:
            for depth in (1, 2, 3):
                configs['mtplx-mtp-' + str(depth)] = MacConfig(engine='mtplx', context=context,
                    depth=depth, speculative='draft-mtp')
        return configs

    def _validate_report(self, report):
        candidates = self._candidate_configs(self.config.context)
        for phase, prompts, passes in (('training', TRAIN_PROMPTS, TRAIN_PASSES), ('validation', VALIDATION_PROMPTS, VALIDATION_PASSES)):
            manifest = self._manifest(prompts)
            if report.get(phase + '_manifest') != manifest:
                raise ValueError('Les workloads du profil ont changé.')
            if 'standard' not in report[phase]:
                raise ValueError('Référence du profil absente.')
            for name, trial in report[phase].items():
                if name not in candidates or trial['config'] != asdict(candidates[name]):
                    raise ValueError('Configuration du profil inconnue.')
                if trial.get('error'):
                    continue
                if len(trial['samples']) != passes * len(manifest):
                    raise ValueError('Passages du profil incomplets.')
                for i, sample in enumerate(trial['samples']):
                    if sample.get('passes') != passes or any(sample.get(k) != v for k, v in manifest[i % len(manifest)].items()):
                        raise ValueError('Les mesures ne correspondent pas aux workloads publics.')
        profiles = verified_profiles(report['training'], report['validation'])
        expected = {name: summarize(trial['samples']) for name, trial in report['validation'].items() if not trial.get('error')}
        if profiles != report['profiles'] or report['summaries'] != expected:
            raise ValueError('Décisions du profil non vérifiables.')
        return profiles

    def _restore_profile(self):
        original = self.config
        binding = self._binding()
        path = self.state_dir / ('mac-' + binding['model_sha256'] + '.json')
        try:
            if not path.is_file() or path.stat().st_size > 2 * 1024 ** 2:
                return
            report = json.loads(path.read_text())
            if any(report.get(k) != v for k, v in binding.items()):
                return
            profiles = self._validate_report(report)
            config = MacConfig(**profiles['balanced']['config'])
            if config.context != self.config.context or config.engine not in model_engines(self.path, self.available()):
                return
            self._start(config)
            self.profile = report
        except (OSError, ValueError, KeyError, TypeError):
            self.profile = None
            if self.process is None:
                self._start(original)

    def _sample(self, prompt, limit):
        upstream = self._request({'op': 'generate', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': limit, 'temperature': 0, 'top_k': 0, 'top_p': 1, 'seed': 42, 'enable_thinking': False,
            'prefill_step_size': self.config.prefill_step_size, 'depth': self.config.depth})
        sample = None
        try:
            for frame in upstream:
                if frame.get('sample'):
                    sample = frame['sample']
        finally:
            upstream.close()
        if sample is None or any(sample.get(k) is None for k in ('decode_tps', 'prefill_seconds')):
            raise ValueError('Le moteur n’a pas fourni des mesures complètes.')
        sample['process_rss_bytes'] = self._rss()
        return sample

    def optimize(self, draft=None, drafts=None):
        if draft or drafts:
            raise ValueError('Les modèles auxiliaires GGUF ne sont pas utilisables par MLX.')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Le moteur Mac est occupé.')
        try:
            if self.job and self.job.get('state') == 'running':
                raise ValueError('Comparaison déjà en cours.')
            if self.process is None or self.process.poll() is not None:
                raise ValueError('Chargez un modèle MLX avant la comparaison.')
            self.cancelled.clear()
            self.job = {'state': 'running', 'progress': 0, 'message': 'Comparaison des moteurs Mac sur les mêmes poids.'}
            threading.Thread(target=self._calibrate, daemon=True).start()
            return dict(self.job)
        finally:
            self.lock.release()

    def _calibrate(self):
        with self.lock:
            original, previous = self.config, self.profile
            try:
                binding = self._binding()
                configs = self._candidate_configs(original.context)
                if len(configs) < 2:
                    raise ValueError('Aucun autre moteur compatible à comparer.')

                def trials_for(candidates, prompts, passes, offset, span):
                    trials = {name: {'config': asdict(config), 'samples': []} for name, config in candidates.items()}
                    done, total = 0, len(candidates) * len(prompts) * passes
                    for round_index in range(passes):
                        order = list(candidates) if round_index % 2 == 0 else list(reversed(candidates))
                        for name in order:
                            if trials[name].get('error'):
                                continue
                            self.job['message'] = ('Sélection' if offset == 0 else 'Vérification indépendante') + ' · ' + name + ' · passage ' + str(round_index + 1)
                            try:
                                self._start(candidates[name])
                                self._sample('Describe a sunny day in a few complete sentences.', 32)
                                for i, prompt in enumerate(prompts):
                                    if self.cancelled.is_set():
                                        raise ValueError('Comparaison interrompue.')
                                    row = self._sample(prompt, OUTPUT_LIMITS[i])
                                    row.update(workload=i, category=CATEGORIES[i], passes=passes)
                                    trials[name]['samples'].append(row)
                                    done += 1
                                    self.job['progress'] = round(offset + span * done / total)
                            except (OSError, ValueError) as exc:
                                if name == 'standard' or self.cancelled.is_set():
                                    raise
                                trials[name]['error'] = str(exc)
                    return trials

                training = trials_for(configs, TRAIN_PROMPTS, TRAIN_PASSES, 0, 55)
                # Token templates must match too, not merely output hashes.
                def gate_templates(trials):
                    base = [r['prompt_sha256'] for r in trials['standard']['samples']]
                    for name, trial in trials.items():
                        if name != 'standard' and not trial.get('error') and [r['prompt_sha256'] for r in trial['samples']] != base:
                            trial['error'] = 'Templates ou tokenisation différents ; comparaison non validée.'
                gate_templates(training)
                # Disjoint validation only for winners of the selection phase.
                from .calibration import select_winner
                finalists = {'standard': configs['standard']}
                for category in USAGE_PROFILES.values():
                    name, _ = select_winner(training, category=category)
                    finalists[name] = configs[name]
                validation = trials_for(finalists, VALIDATION_PROMPTS, VALIDATION_PASSES, 55, 40)
                gate_templates(validation)
                profiles = verified_profiles(training, validation)
                if self._binding() != binding:
                    raise ValueError('Poids ou environnement modifiés pendant la comparaison.')
                report = {**binding, 'training': training, 'validation': validation, 'profiles': profiles,
                          'training_manifest': self._manifest(TRAIN_PROMPTS), 'validation_manifest': self._manifest(VALIDATION_PROMPTS),
                          'summaries': {name: summarize(trial['samples']) for name, trial in validation.items() if not trial.get('error')},
                          'decisions': {name: assess_candidate(training['standard'], trial) for name, trial in training.items() if name != 'standard'},
                          'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                self._start(MacConfig(**profiles['balanced']['config']))
                self.state_dir.mkdir(parents=True, exist_ok=True)
                destination = self.state_dir / ('mac-' + binding['model_sha256'] + '.json')
                with tempfile.NamedTemporaryFile(mode='w', dir=self.state_dir, delete=False) as handle:
                    json.dump(report, handle, ensure_ascii=False)
                    temp = Path(handle.name)
                temp.replace(destination)
                self.profile, self.usage_profile = report, 'balanced'
                self.job.update(state='done', progress=100,
                    message='Aucun gain validé ; moteur de référence conservé.' if profiles['balanced']['winner'] == 'standard' else 'Moteur retenu après vérification indépendante.', result=report)
            except Exception as exc:
                self._stop()
                self.profile = previous
                self.job.update(state='cancelled' if self.cancelled.is_set() else 'failed', message=str(exc))
                self.cancelled.clear()
                if not self.closed:
                    try:
                        self._start(original)
                    except (OSError, ValueError) as error:
                        self.job['message'] += ' Rechargez le modèle : ' + str(error)
