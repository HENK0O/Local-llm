"""Opt-in, offline MLX calibration in a separate Python environment.

This is an experiment, not a replacement for the GGUF chat backend. It measures
installed MLX weights without downloading/converting models or trusting custom
model code. No user conversation is used in the calibration.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from .calibration import (CATEGORIES, OUTPUT_LIMITS, TRAIN_PROMPTS, VALIDATION_PROMPTS,
                          TRAIN_PASSES, VALIDATION_PASSES, USAGE_PROFILES,
                          select_winner, verified_profiles)
from .loading import model_fingerprint


def validate_model(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not (root / 'config.json').is_file() or not list(root.glob('*.safetensors')):
        raise ValueError('MLX nécessite un dossier local avec config.json et poids .safetensors compatibles. Un GGUF ne suffit pas.')
    if not (root / 'tokenizer.json').is_file() or not (root / 'tokenizer_config.json').is_file():
        raise ValueError('Tokenizer local incomplet pour MLX.')
    for name in ('config.json', 'tokenizer_config.json'):
        if (root / name).stat().st_size > 1024 ** 2:
            raise ValueError('Configuration MLX trop volumineuse.')
        config = json.loads((root / name).read_text())
        if config.get('auto_map'):
            raise ValueError('Les modèles nécessitant du code distant ne sont pas pris en charge.')
    return root


def run_experiment(path, python=None):
    """Run an installed interpreter; never install dependencies automatically."""
    root = validate_model(path)
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Cette expérience MLX est réservée aux Mac Apple Silicon.')
    interpreter = str(Path(python).expanduser().absolute()) if python else sys.executable
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HF_', 'TRANSFORMERS_', 'MLX_')) and k != 'LM_STUDIO_API_TOKEN'}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    # Installed local-llm may be a source checkout, absent from the isolated venv.
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
    try:
        result = subprocess.run([interpreter, '-m', 'local_llm.mlx_experiment', str(root)],
                                env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Délai de calibration MLX dépassé ; le worker de test a été arrêté.') from exc
    if result.returncode:
        raise ValueError('Expérience MLX indisponible : ' + result.stderr[-2000:].strip())
    return json.loads(result.stdout)


def calibrate_mlx(root):
    if sys.version_info < (3, 11):
        raise ValueError('Utilisez --python avec un environnement séparé Python 3.11+ contenant mlx-lm.')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Un Mac Apple Silicon est requis.')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    initial, _ = model_fingerprint(root)
    model, tokenizer = load(str(root), tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
    if not tokenizer.chat_template:
        raise ValueError('Un template de conversation local est requis.')
    mx.eval(model.parameters())

    def sample(prompt, tokens, step):
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        generated = []
        first = last = None
        mx.reset_peak_memory()
        started = time.perf_counter()
        for response in stream_generate(model, tokenizer, formatted, max_tokens=tokens,
                                        sampler=make_sampler(temp=0), prefill_step_size=step):
            if first is None:
                first = time.perf_counter() - started
            generated.append(response.token)
            last = response
        elapsed = time.perf_counter() - started
        if last is None or not generated:
            raise ValueError('Le modèle MLX n’a généré aucun token.')
        return {'seconds': elapsed, 'decode_tps': last.generation_tps,
                'prefill_seconds': last.prompt_tokens / last.prompt_tps,
                'generated_tokens': last.generation_tokens, 'first_token_seconds': first,
                'peak_gpu_bytes': int(last.peak_memory * 1e9), 'process_rss_bytes': None,
                'output_sha256': hashlib.sha256(json.dumps(generated).encode()).hexdigest()}

    configs = {'standard': {'prefill_step_size': 2048}, 'prefill-512': {'prefill_step_size': 512},
               'prefill-128': {'prefill_step_size': 128}}
    def trials_for(configurations, prompts, passes):
        trials = {name: {'config': config, 'samples': []} for name, config in configurations.items()}
        for index in range(passes):
            order = list(configurations) if index % 2 == 0 else list(reversed(configurations))
            if index == 2:
                order = order[1:] + order[:1]
            for name in order:
                step = configurations[name]['prefill_step_size']
                sample('Describe a sunny day in a few complete sentences.', 32, step)
                for i, prompt in enumerate(prompts):
                    row = sample(prompt, OUTPUT_LIMITS[i], step)
                    row.update(category=CATEGORIES[i], workload=i, passes=passes)
                    trials[name]['samples'].append(row)
        return trials
    training = trials_for(configs, TRAIN_PROMPTS, TRAIN_PASSES)
    finalists = {'standard': configs['standard']}
    for category in USAGE_PROFILES.values():
        name, _ = select_winner(training, category=category)
        finalists[name] = configs[name]
    validation = trials_for(finalists, VALIDATION_PROMPTS, VALIDATION_PASSES)
    if model_fingerprint(root)[0] != initial:
        raise ValueError('Les poids MLX ont changé pendant les essais.')
    return {'backend': 'mlx', 'experimental': True, 'protocol': 1, 'model_sha256': initial,
            'runtime': {'mlx': version('mlx'), 'mlx_lm': version('mlx-lm'), 'python': platform.python_version()},
            'hardware': {'system': platform.system(), 'machine': platform.machine(), 'device': mx.device_info().get('device_name')},
            'training': training, 'validation': validation, 'profiles': verified_profiles(training, validation),
            'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'scope': 'same_mlx_weights_prefill_tuning_only',
            'comparison_with_llama_cpp': None}


if __name__ == '__main__':
    try:
        root = validate_model(sys.argv[1])
        # Loading libraries may emit diagnostics; reserve stdout for the report.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            report = calibrate_mlx(root)
        print(json.dumps(report, ensure_ascii=False))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
