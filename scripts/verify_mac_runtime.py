"""Reproducible offline Mac worker checks; never start local-llm serve.

Uses an installed MLX checkpoint in place. Public prompts only. Owned workers
and benchmark state are temporary, and workers are closed even on failure.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_llm.discovery import inspect_model
from local_llm.mac_runtime import MacRuntime
from local_llm.recommendations import detect_hardware
from local_llm.telemetry import SystemTelemetry


def verify(path, engines, calibrate=False, candidates=None):
    item = inspect_model(path)
    report = {'model_name': item.name, 'hardware': detect_hardware(), 'runs': [],
              'scope': 'installed_mlx_artifact_private_workers_public_prompts',
              'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    telemetry = SystemTelemetry()
    for engine in engines:
        with tempfile.TemporaryDirectory(prefix='local-llm-mac-verify-') as folder:
            runtime = MacRuntime(folder)
            def available():
                snapshot = telemetry.snapshot(refresh=True)
                total, used = snapshot.get('memory_total_bytes'), snapshot.get('memory_used_bytes')
                return max(0, total - used) + (runtime._rss() or 0) if total is not None and used is not None else None
            runtime.memory_probe = available
            try:
                runtime.load(item, available(), engine)
                print('Loaded ' + engine + ': ' + item.name, file=sys.stderr, flush=True)
                run = {'engine': engine, 'packages': runtime.available()['packages'],
                       'memory_plan': runtime.memory, 'checks': []}
                messages = [{'role': 'user', 'content': 'Réponds seulement par le mot Bonjour.'}]
                def complete(history, conversation):
                    started = time.perf_counter()
                    text, reasoning, final = '', '', None
                    for chunk in runtime.iter_chat({'model': item.id, 'messages': history, 'max_tokens': 512,
                                                   'temperature': 0, 'seed': 42}, conversation):
                        if chunk['model'] != item.id:
                            raise AssertionError('Wrong model identity')
                        for choice in chunk.get('choices', []):
                            delta = choice.get('delta') or {}
                            text += delta.get('content', '')
                            reasoning += delta.get('reasoning_content', '')
                        if chunk.get('usage'):
                            final = chunk
                    if not final or not text.strip():
                        raise AssertionError('No visible answer; reasoning only: ' + str(len(reasoning)))
                    return {'answer': text, 'reasoning_characters': len(reasoning), 'usage': final['usage'],
                            'timings': final['timings'], 'request_seconds': time.perf_counter() - started}
                first = complete(messages, 'public-chat')
                if 'bonjour' not in first['answer'].lower():
                    raise AssertionError('Unexpected answer to first public prompt')
                run['checks'].append({'name': 'visible_answer', **first})
                messages += [{'role': 'assistant', 'content': first['answer']},
                             {'role': 'user', 'content': 'Quel mot viens-tu de dire ? Répète seulement ce mot.'}]
                preview = runtime.context(messages)
                if first['answer'] not in preview['prompt'] or messages[-1]['content'] not in preview['prompt']:
                    raise AssertionError('Context lost the previous answer or the new question')
                second = complete(messages, 'public-chat')
                if 'bonjour' not in second['answer'].lower():
                    raise AssertionError('Contextual recall failed')
                run['checks'].append({'name': 'contextual_recall', 'prompt_tokens': preview['prompt_tokens'], **second})
                partial = runtime.iter_chat({'model': item.id, 'messages': [{'role': 'user', 'content': 'Explique le fonctionnement d’un vélo.'}],
                                             'max_tokens': 512, 'temperature': 0}, 'interrupted')
                try:
                    next(partial)
                finally:
                    partial.close()
                if runtime.process is not None:
                    raise AssertionError('Interrupted worker remained running')
                runtime.load(item, available(), engine)
                resumed = complete(messages, 'public-chat')
                if 'bonjour' not in resumed['answer'].lower():
                    raise AssertionError('Late frames contaminated resumed answer')
                run['checks'].append({'name': 'interrupt_reload_resume', **resumed})
                if calibrate:
                    if candidates:
                        original = runtime._candidate_configs
                        requested = set(candidates)
                        def bounded(context):
                            configs = original(context)
                            if 'standard' not in requested or requested - set(configs):
                                raise ValueError('Unknown calibration candidates or missing standard')
                            return {key: value for key, value in configs.items() if key in requested}
                        runtime._candidate_configs = bounded
                    runtime.optimize()
                    last_message = None
                    while runtime.job['state'] == 'running':
                        message = runtime.job.get('message')
                        if message != last_message:
                            print(message, file=sys.stderr, flush=True)
                            last_message = message
                        time.sleep(.5)
                    if runtime.job['state'] != 'done':
                        raise AssertionError(runtime.job['message'])
                    run['calibration'] = runtime.job['result']
                    run['saved_profile_revalidated'] = bool(runtime._validate_report(run['calibration']))
                report['runs'].append(run)
            finally:
                runtime.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--engine', choices=['mlx', 'mtplx', 'both'], default='both')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--candidates', nargs='+', help='Bounded reproducible subset; must include standard. Listed in the exported report.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.model, ['mlx', 'mtplx'] if args.engine == 'both' else [args.engine], args.calibrate, args.candidates)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print('Checks passed. Report: ' + str(args.output))
    except (AssertionError, OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
