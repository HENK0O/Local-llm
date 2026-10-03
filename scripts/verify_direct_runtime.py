"""Verify a GGUF's direct chat lifecycle and optionally three calibration candidates.

Uses public prompts and isolated temporary state. Never touches an app server,
downloads a checkpoint or stops a process it did not create. The optional
calibration exercises six workloads, disjoint holdout prompts and all normal
validation gates; its three candidates are not an exhaustive tuning search.
"""
import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm.accelerator import Accelerator
from local_llm.discovery import inspect_model
from local_llm.telemetry import SystemTelemetry
from local_llm.version import __version__


class VerificationRuntime(Accelerator):
    def _candidate_configs(self, original, drafts):
        configs = super()._candidate_configs(original, drafts)
        names = ('standard', 'réglages-1024', 'motifs-adaptatifs-64')
        return {name: configs[name] for name in names if name in configs}


def text(chunks, field='content'):
    return ''.join((choice.get('delta') or {}).get(field) or ''
                   for chunk in chunks for choice in chunk.get('choices', []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--available-gib', type=float, help='Optional ceiling; never increases measured available RAM')
    parser.add_argument('--calibrate', action='store_true', help='Compare three targeted configurations with independent validation')
    args = parser.parse_args()
    if args.available_gib is not None and (not math.isfinite(args.available_gib) or args.available_gib <= 0):
        parser.error('available-gib must be finite and positive')
    item = inspect_model(args.model)
    report = {'app_version': __version__, 'model': args.model.name,
              'scope': 'Private direct worker, public prompts, no external-engine speedup claim',
              'calibration_search': 'Three targeted configurations' if args.calibrate else None}
    with tempfile.TemporaryDirectory(prefix='local-llm-verify-') as folder:
        runtime = VerificationRuntime(state_dir=folder)
        telemetry = SystemTelemetry()
        def available():
            snapshot = telemetry.snapshot(refresh=True)
            total, used = snapshot['memory_total_bytes'], snapshot['memory_used_bytes']
            return min(total, max(0,total-used)+(runtime._rss() or 0)) if total and used is not None else None
        runtime.memory_probe = available
        try:
            budget = available()
            if args.available_gib is not None:
                if budget is None:
                    raise ValueError('Available RAM is unknown; a ceiling cannot substitute for a measurement')
                budget = min(budget, int(args.available_gib * 1024**3))
            info = runtime.load(item, budget)
            report['load'] = {'plan': info['memory_plan'], 'allocations': info['runtime_memory']}
            print('Model loaded', flush=True)
            messages = [{'role':'system', 'content':'Réponds brièvement.'}]
            report['turns'] = []
            for prompt in ('Réponds uniquement Bonjour.', 'Quel est le mot que je viens de te demander de dire ?'):
                messages.append({'role':'user', 'content':prompt})
                payload = {'model':item.id, 'messages':messages, 'max_tokens':512, 'temperature':0, 'seed':42}
                started = time.perf_counter()
                chunks = list(runtime.iter_chat(payload, 'public-verification'))
                answer = text(chunks)
                if not answer.strip():
                    raise ValueError('No visible answer within 512 output tokens; reasoning is not counted as an answer')
                if 'bonjour' not in answer.casefold():
                    raise ValueError('The public chat check did not retain the requested word Bonjour')
                if any(chunk.get('model',item.id) != item.id for chunk in chunks):
                    raise ValueError('The answering model differs from the requested checkpoint')
                report['turns'].append({'prompt':prompt, 'answer':answer, 'seconds':time.perf_counter()-started,
                                        'timings':[c['timings'] for c in chunks if c.get('timings')]})
                messages.append({'role':'assistant', 'content':answer})
                print('Chat turn verified', flush=True)
            payload = {'model':item.id, 'messages':[{'role':'user','content':'Write a long story about a bicycle.'}],
                       'max_tokens':512, 'temperature':0, 'seed':42}
            stream = runtime.iter_chat(payload,'interrupted-verification')
            try:
                next(stream)
            finally:
                stream.close()
            if 'interrupted-verification' in runtime.slots.entries:
                raise ValueError('Interrupted state remained eligible for context reuse')
            chunks = list(runtime.iter_chat(payload,'interrupted-verification'))
            if not text(chunks).strip():
                raise ValueError('No visible answer after resuming an interrupted request')
            report['interrupt_resume'] = True
            print('Interrupt and resume verified', flush=True)
            if args.calibrate:
                runtime.optimize(drafts=[])
                previous = None
                while runtime.job['state'] == 'running':
                    status = runtime.job.get('message'), runtime.job.get('progress')
                    if status != previous:
                        print(status, flush=True); previous = status
                    time.sleep(1)
                if runtime.job['state'] != 'complete':
                    raise ValueError(runtime.job.get('message','Calibration failed'))
                report['calibration'] = runtime.profile
                print('Independent validation complete', flush=True)
            report['passed'] = True
        except BaseException as exc:
            report['passed'] = False
            report['error'] = str(exc)
            raise
        finally:
            # Stop a calibration before waiting for its lock on interruption.
            runtime.cancelled.set()
            runtime.close()
            telemetry.close()
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
