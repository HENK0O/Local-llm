"""Fixed public workloads, conservative memory planning and benchmark decisions.

No conversation text is used or saved by calibration. Selection and validation
use disjoint prompts; validation is never used to choose a second candidate.
"""
import math
import statistics

MIB = 1024 ** 2
TRAIN_PROMPTS = (
    'Explain how a bicycle changes gears and give practical advice for climbing a hill.',
    'Write a Python function to merge two sorted lists without sorting the result. Explain edge cases and complexity.',
    'Compare these local application measurements and propose three priorities:\n' + '\n'.join(
        'Run %d: input %d tokens, output %d tokens, latency %.2f seconds, memory %d MiB, errors %d.' %
        (i, 150 + i * 37, 90 + i * 11, 0.2 + (i % 7) * .13, 700 + i * 9, i % 3)
        for i in range(1, 33)),
)
VALIDATION_PROMPTS = (
    'Une amie débute le jardinage sur un balcon ombragé. Propose un plan simple pour les premières semaines et explique tes choix.',
    'Implement an LRU cache in Python using OrderedDict. Include get and put, explain eviction, and show a short example.',
    'Identify the main risks in this project log and propose a schedule:\n' + '\n'.join(
        'Day %d: team %s tested %d cases, found %d failures, has %d hours left; next delivery is task %s.' %
        (i, ['north', 'south', 'east'][i % 3], 17 + i * 3, i % 5, 9 + i % 11, chr(65 + i % 20))
        for i in range(1, 37)),
)
CATEGORIES = ('discussion', 'code', 'contexte long')
OUTPUT_LIMITS = (128, 192, 128)
TRAIN_PASSES = 2
VALIDATION_PASSES = 3


def _rounds(samples):
    if not samples or any(not math.isfinite(s['seconds']) or s['seconds'] <= 0 or
                          not math.isfinite(s['decode_tps']) or s['decode_tps'] <= 0 for s in samples):
        raise ValueError('Invalid benchmark timings')
    # Legacy unit fixtures also represent two complete three-workload passes.
    expected = samples[0].get('passes', TRAIN_PASSES)
    if expected not in (TRAIN_PASSES, VALIDATION_PASSES) or len(samples) != expected * len(CATEGORIES):
        raise ValueError('Complete benchmark passes are required')
    return [samples[r:r + 3] for r in range(0, len(samples), 3)]


def summarize(samples):
    rounds = _rounds(samples)
    def decode(group):
        work = [max(1, s['generated_tokens'] - 1) for s in group]
        return sum(work) / sum(n / s['decode_tps'] for n, s in zip(work, group))
    totals = [sum(s['seconds'] for s in group) for group in rounds]
    categories = {}
    for i, category in enumerate(CATEGORIES):
        rows = [group[i] for group in rounds]
        categories[category] = {'seconds': statistics.median(s['seconds'] for s in rows),
                                'decode_tps': decode(rows)}
    return {'seconds': statistics.median(totals), 'decode_tps': statistics.median(decode(g) for g in rounds),
            'prefill_seconds': statistics.median(s['prefill_seconds'] for s in samples),
            'process_rss_bytes': max((s['process_rss_bytes'] for s in samples if s.get('process_rss_bytes') is not None), default=None),
            'variation_percent': 100 * (max(totals) - min(totals)) / statistics.median(totals),
            'categories': categories}


def assess_candidate(base, trial):
    if trial.get('error'):
        return {'accepted': False, 'reason': 'Échec du runtime : ' + trial['error']}
    rows = trial['samples']
    if [s['output_sha256'] for s in rows] != [s['output_sha256'] for s in base['samples']]:
        return {'accepted': False, 'reason': 'Sorties différentes de la référence ; qualité non validée.'}
    try:
        reference, candidate = summarize(base['samples']), summarize(rows)
        ratios = [sum(s['seconds'] for s in a) / sum(s['seconds'] for s in b)
                  for a, b in zip(_rounds(rows), _rounds(base['samples']))]
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return {'accepted': False, 'reason': 'Mesures incomplètes ou invalides.'}
    if not all(r < .95 for r in ratios):
        reason = 'Gain inférieur à 5 % ou instable entre les passages.'
    elif candidate['decode_tps'] < reference['decode_tps']:
        reason = 'Le décodage ralentit malgré le gain global.'
    elif any(candidate['categories'][c]['seconds'] > reference['categories'][c]['seconds'] * 1.1 for c in CATEGORIES):
        reason = 'Une catégorie de requêtes ralentit de plus de 10 %.'
    else:
        return {'accepted': True, 'reason': 'Sorties identiques et gain stable sur chaque passage.',
                'round_gain_percent': [100 * (1 / r - 1) for r in ratios]}
    return {'accepted': False, 'reason': reason, 'round_gain_percent': [100 * (1 / r - 1) for r in ratios]}


def select_winner(trials, baseline_name='standard'):
    base = trials[baseline_name]
    summaries = {baseline_name: summarize(base['samples'])}
    winner = baseline_name
    for name, trial in trials.items():
        if name == baseline_name:
            continue
        decision = assess_candidate(base, trial)
        if [s['output_sha256'] for s in trial['samples']] == [s['output_sha256'] for s in base['samples']]:
            try:
                summaries[name] = summarize(trial['samples'])
            except (ValueError, KeyError, TypeError):
                continue
        if decision['accepted'] and summaries[name]['seconds'] < summaries[winner]['seconds']:
            winner = name
    return winner, summaries


def memory_plan(metadata, weight_bytes, available=None, required=512):
    """Bound weights + conservative KV estimate + buffers; leave an OS margin.

    Hybrid/recurrent and MLA architectures use a deliberately conservative
    estimate and one slot, since their state does not follow dense KV geometry.
    Estimates are not advertised as actual memory measurements.
    """
    architecture = str(metadata.get('general.architecture', ''))
    def integer(key, default):
        value = metadata.get(architecture + '.' + key, default)
        return value if isinstance(value, int) and value > 0 else default
    trained = integer('context_length', 4096)
    maximum = min(32768, trained)
    if required > maximum:
        raise ValueError('Le contexte requis dépasse la capacité du modèle (aucun message supprimé).')
    layers = integer('block_count', 32)
    heads = integer('attention.head_count', 32)
    kv_heads = integer('attention.head_count_kv', heads)
    embedding = integer('embedding_length', 4096)
    key = integer('attention.key_length', embedding // heads)
    value = integer('attention.value_length', embedding // heads)
    dense = architecture in {'llama', 'qwen2', 'mistral', 'gemma', 'gemma2', 'gemma3', 'phi3'}
    per_layer = metadata.get(architecture + '.attention.head_count_kv')
    head_total = (sum(per_layer) if isinstance(per_layer, list) and len(per_layer) == layers and
                  all(isinstance(n, int) and n >= 0 for n in per_layer) else layers * kv_heads)
    kv_per_token = head_total * (key + value) * 2
    if not dense:
        kv_per_token = max(256 * 1024, kv_per_token * 2)
    fixed = int(weight_bytes * 1.2) + 512 * MIB
    if available is None:
        context, slots = min(4096, maximum), 2 if dense else 1
        if required > context:
            raise ValueError('RAM disponible inconnue : impossible d’agrandir le contexte automatiquement.')
        return {'context': max(512, context), 'slots': slots, 'available_bytes': None,
                'estimated_bytes': fixed + context * slots * kv_per_token, 'kv_bytes_per_token': kv_per_token,
                'reserve_bytes': None, 'conservative': not dense}
    reserve = max(512 * MIB, min(2 * 1024 ** 3, int(available * .1)))
    budget = max(0, available - reserve - fixed)
    # Give useful context priority, then keep up to four independent histories.
    for context in ((32768,) if required > 16384 else ()) + (16384, 8192, 4096, 2048, 1024, 512):
        context = min(context, maximum)
        if context < required or context * kv_per_token > budget:
            continue
        slots = min(4 if dense else 1, int(budget // (context * kv_per_token)))
        return {'context': context, 'slots': slots, 'available_bytes': available,
                'estimated_bytes': fixed + context * slots * kv_per_token,
                'kv_bytes_per_token': kv_per_token, 'reserve_bytes': reserve, 'conservative': not dense}
    raise ValueError('Mémoire disponible insuffisante pour les poids et ce contexte. Déchargez les modèles inutilisés dans LM Studio ou choisissez un modèle plus petit.')
