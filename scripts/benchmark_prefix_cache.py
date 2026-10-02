"""Measure two conversation turns with prefix reuse enabled or disabled.

The reference is the same local-llm engine, not LM Studio or llama.cpp.
Run from the repository root: python scripts/benchmark_prefix_cache.py --help
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm.cache import PrefixCache
from local_llm.chat import ChatMessage, format_chat
from local_llm.generation import generate_tokens
from local_llm.loading import load_runtime, model_fingerprint


FIRST_QUESTION = (
    'Contexte de travail : nous préparons une petite application locale pour prendre des notes, '
    'organiser des idées et conserver plusieurs conversations privées. Les utilisateurs préfèrent une interface '
    'simple, lisible et rapide. Les textes doivent rester sur leur ordinateur. ' * 6
) + '\nRésume ce contexte en une phrase.'
FOLLOWUP = 'Où les textes des utilisateurs doivent-ils rester ?'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or not 2 <= args.tokens <= 128:
        parser.error('runs must be positive and tokens between 2 and 128')
    model, tokenizer = load_runtime(args.model)
    messages = [ChatMessage('user', FIRST_QUESTION)]
    first_ids = tokenizer.encode(format_chat(messages, tokenizer))

    def run(ids, prefix=None):
        started = time.perf_counter()
        result = list(generate_tokens(model, ids, args.tokens, prefix_cache=prefix))
        elapsed = time.perf_counter() - started
        return [token for token, _ in result], result[-1][1], elapsed

    samples = {'cold': [], 'warm': []}
    identical = True
    for trial in range(args.runs):
        prefix = PrefixCache()
        first, _, _ = run(first_ids, prefix)
        next_messages = messages + [ChatMessage('assistant', tokenizer.decode(first)),
                                    ChatMessage('user', FOLLOWUP)]
        next_ids = tokenizer.encode(format_chat(next_messages, tokenizer))
        outputs = {}
        for label in (['cold', 'warm'] if trial % 2 == 0 else ['warm', 'cold']):
            tokens, stats, elapsed = run(next_ids, prefix if label == 'warm' else None)
            samples[label].append(dict(total_seconds=elapsed, prefill_seconds=stats.prefill_seconds,
                decode_seconds=stats.decode_seconds, generated_tokens=len(tokens),
                reused_prompt_tokens=stats.reused_prompt_tokens, generated_ids=tokens))
            outputs[label] = tokens
        identical = identical and outputs['cold'] == outputs['warm']
    if not identical:
        raise RuntimeError('Greedy tokens differ; refusing to report a speedup')

    cold = statistics.median(s['prefill_seconds'] for s in samples['cold'])
    warm = statistics.median(s['prefill_seconds'] for s in samples['warm'])
    fingerprint, _ = model_fingerprint(args.model)
    report = dict(model=args.model.name, model_sha256=fingerprint,
        comparison='Même moteur local-llm, réutilisation du préfixe activée ou désactivée ; aucune comparaison à LM Studio.',
        first_question=FIRST_QUESTION, followup=FOLLOWUP,
        prompt_tokens=len(next_ids), tokens_identical=identical, alternating_trials=args.runs,
        retained_cache_limit_bytes=prefix.max_bytes,
        median_prefill_seconds=dict(cold=cold, warm=warm),
        median_total_seconds={label: statistics.median(s['total_seconds'] for s in runs)
                              for label, runs in samples.items()},
        prefill_reduction_percent=(cold - warm) / cold * 100, measurements=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'measurements'}, indent=2))


if __name__ == '__main__':
    main()
