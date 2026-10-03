"""Fixed public workloads, conservative memory planning and benchmark decisions.

No conversation text is used or saved by calibration. Selection and validation
use disjoint prompts; validation is never used to choose a second candidate.
"""
import math
import statistics

from .memory import geometry

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
USAGE_PROFILES = {'balanced': None, 'discussion': 'discussion', 'code': 'code', 'long_context': 'contexte long'}


def _category(row, index):
    return row.get('category', CATEGORIES[index % 3])


def _rounds(samples):
    if not samples or any(not math.isfinite(s['seconds']) or s['seconds'] <= 0 or
                          not math.isfinite(s['decode_tps']) or s['decode_tps'] <= 0 for s in samples):
        raise ValueError('Invalid benchmark timings')
    expected = samples[0].get('passes', TRAIN_PASSES)
    width = samples[0].get('workloads_count', len(CATEGORIES))
    if expected not in (TRAIN_PASSES, VALIDATION_PASSES) or width not in (3, 6) or len(samples) != expected * width:
        raise ValueError('Complete benchmark passes are required')
    groups = [samples[r:r + width] for r in range(0, len(samples), width)]
    if any('workloads_count' in s for s in samples):
        identities = [(s['workload'], s['category']) for s in groups[0]]
        if (len(set(identities)) != width or set(c for _, c in identities) != set(CATEGORIES) or
                any(s.get('passes') != expected or s.get('workloads_count') != width for s in samples) or
                any([(s['workload'], s['category']) for s in group] != identities for group in groups)):
            raise ValueError('Inconsistent workloads')
    return groups


def summarize(samples):
    rounds = _rounds(samples)
    def decode(group):
        work = [max(1, s['generated_tokens'] - 1) for s in group]
        return sum(work) / sum(n / s['decode_tps'] for n, s in zip(work, group))
    def optional_median(rows, key):
        values = [s.get(key) for s in rows]
        return statistics.median(values) if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values) else None
    totals = [sum(s['seconds'] for s in group) for group in rounds]
    categories = {}
    for category in CATEGORIES:
        groups = [[s for i, s in enumerate(group) if _category(s, i) == category] for group in rounds]
        rows = [s for group in groups for s in group]
        categories[category] = {'seconds': statistics.median(sum(s['seconds'] for s in group) for group in groups),
                                'decode_tps': decode(rows)}
    workloads = []
    for i, first in enumerate(rounds[0]):
        rows = [group[i] for group in rounds]
        workloads.append({'id': first.get('workload', i), 'category': _category(first, i),
            'input_tokens': optional_median(rows, 'input_tokens'),
            'output_tokens': optional_median(rows, 'generated_tokens'),
            'output_limit': first.get('output_limit'), 'seconds': statistics.median(s['seconds'] for s in rows),
            'first_token_seconds': optional_median(rows, 'first_token_seconds'),
            'decode_tps': decode(rows), 'prefill_seconds': optional_median(rows, 'prefill_seconds')})
    return {'seconds': statistics.median(totals), 'decode_tps': statistics.median(decode(g) for g in rounds),
            'prefill_seconds': statistics.median(s['prefill_seconds'] for s in samples),
            'first_token_seconds': optional_median(samples, 'first_token_seconds'),
            'process_rss_bytes': max((s['process_rss_bytes'] for s in samples if s.get('process_rss_bytes') is not None), default=None),
            'process_rss_peak_bytes': max((s['process_rss_peak_bytes'] for s in samples if s.get('process_rss_peak_bytes') is not None), default=None),
            'variation_percent': 100 * (max(totals) - min(totals)) / statistics.median(totals),
            'categories': categories, 'workloads': workloads}


def assess_candidate(base, trial, category=None):
    if trial.get('error'):
        return {'accepted': False, 'reason': 'Échec du runtime : ' + trial['error']}
    rows = trial['samples']
    fields = ('workload', 'category', 'workloads_count', 'input_tokens', 'output_limit')
    if [[s.get(k) for k in fields] for s in rows] != [[s.get(k) for k in fields] for s in base['samples']]:
        return {'accepted': False, 'reason': 'Workloads différents de la référence.'}
    if [s['output_sha256'] for s in rows] != [s['output_sha256'] for s in base['samples']]:
        return {'accepted': False, 'reason': 'Sorties différentes de la référence ; qualité non validée.'}
    try:
        reference, candidate = summarize(base['samples']), summarize(rows)
        def elapsed(group):
            return sum(s['seconds'] for i, s in enumerate(group) if category is None or _category(s, i) == category)
        ratios = [elapsed(a) / elapsed(b)
                  for a, b in zip(_rounds(rows), _rounds(base['samples']))]
        candidate_scope = candidate['categories'][category] if category else candidate
        reference_scope = reference['categories'][category] if category else reference
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return {'accepted': False, 'reason': 'Mesures incomplètes ou invalides.'}
    if not all(r < .95 for r in ratios):
        reason = 'Gain inférieur à 5 % ou instable entre les passages.'
    elif candidate_scope['decode_tps'] < reference_scope['decode_tps']:
        reason = 'Le décodage ralentit malgré le gain global.'
    elif any(a['seconds'] > b['seconds'] * 1.1 for a, b in zip(candidate['workloads'], reference['workloads'])
             if category is None or a['category'] == category):
        reason = 'Une catégorie ou longueur de requêtes ralentit de plus de 10 %.'
    elif any(a['first_token_seconds'] is not None and b['first_token_seconds'] is not None and
             a['first_token_seconds'] > max(b['first_token_seconds'] * 1.2, b['first_token_seconds'] + .01)
             for a, b in zip(candidate['workloads'], reference['workloads'])
             if category is None or a['category'] == category):
        reason = 'Le premier token ralentit de plus de 20 % et 10 ms sur une longueur de requête.'
    else:
        return {'accepted': True, 'reason': 'Sorties identiques et gain stable sur chaque passage.',
                'round_gain_percent': [100 * (1 / r - 1) for r in ratios]}
    return {'accepted': False, 'reason': reason, 'round_gain_percent': [100 * (1 / r - 1) for r in ratios]}


def select_winner(trials, baseline_name='standard', category=None):
    base = trials[baseline_name]
    summaries = {baseline_name: summarize(base['samples'])}
    winner = baseline_name
    for name, trial in trials.items():
        if name == baseline_name:
            continue
        decision = assess_candidate(base, trial, category)
        if [s['output_sha256'] for s in trial['samples']] == [s['output_sha256'] for s in base['samples']]:
            try:
                summaries[name] = summarize(trial['samples'])
            except (ValueError, KeyError, TypeError):
                continue
        def score(summary):
            return summary['categories'][category]['seconds'] if category else summary['seconds']
        if decision['accepted'] and score(summaries[name]) < score(summaries[winner]):
            winner = name
    return winner, summaries


def memory_plan(metadata, weight_bytes, available=None, required=512):
    """Bound weights + conservative KV estimate + buffers; leave an OS margin.

    Known hybrid/MLA architectures separate fixed recurrent state from KV.
    Unknown metadata retains a conservative fallback. One slot is reserved.
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
    shape = geometry(metadata)
    kv_per_token = shape['kv_bytes_per_token']
    # Four F32 recurrent rows conservatively cover sequence bookkeeping for
    # the single slot. Speculative rollback snapshots are budgeted separately.
    recurrent = shape['recurrent_state_bytes'] * 4
    buffers = int(weight_bytes * .2) + 512 * MIB
    fixed = weight_bytes + buffers + recurrent
    def result(context, cache_mib=0, reserve=None):
        return dict(shape, context=context, slots=1, available_bytes=available,
                    cache_ram_mib=cache_mib, reserve_bytes=reserve,
                    weight_bytes=weight_bytes, compute_budget_bytes=buffers,
                    recurrent_budget_bytes=recurrent, context_bytes=context * kv_per_token,
                    estimated_bytes=fixed + context * kv_per_token + cache_mib * MIB)
    # Start with one GPU history and at most 4096 tokens. Grow only when a
    # request needs it; extra free RAM is not a reason to reserve four KV slots.
    if available is None:
        context = min(4096, maximum)
        if required > context:
            raise ValueError('RAM disponible inconnue : impossible d’agrandir le contexte automatiquement.')
        return result(max(512, context))
    reserve = max(512 * MIB, min(2 * 1024 ** 3, int(available * .1)))
    budget = max(0, available - reserve - fixed)
    choices = [min(n, maximum) for n in (512, 1024, 2048, 4096, 8192, 16384, 32768)]
    desired = next((n for n in choices if n >= max(required, min(4096, maximum))), maximum)
    # Prefer the smallest sufficient capacity, falling back for short requests
    # when 4096 tokens cannot fit. Never drop messages to make the request fit.
    order = [n for n in choices if n >= desired] + [n for n in reversed(choices) if required <= n < desired]
    for context in dict.fromkeys(order):
        if context * kv_per_token > budget:
            continue
        cache_budget = min(available // 64, budget - context * kv_per_token)
        cache_mib = next(n for n in (256, 128, 64, 0) if n * MIB <= cache_budget)
        return result(context, cache_mib, reserve)
    minimum_context = min(n for n in choices if n >= required)
    working = fixed + minimum_context * kv_per_token
    # Solve working + the same bounded reserve used above; exclude optional cache.
    minimum = working + 512 * MIB
    if minimum > 5 * 1024 ** 3:
        minimum = working / .9
    if minimum > 20 * 1024 ** 3:
        minimum = working + 2 * 1024 ** 3
    raise ValueError('RAM disponible insuffisante : {:.1f} Gio disponibles, au moins {:.1f} Gio estimés pour ce checkpoint avec {} tokens de contexte. Déchargez les modèles ouverts dans un autre moteur ou libérez de la RAM, puis réessayez.'.format(available / 1024 ** 3, minimum / 1024 ** 3, minimum_context))


def shortlist(screening, limit=6):
    """Training-only, cheap screening. Its timings never count as verified gains."""
    base = screening['standard']['samples']
    scores = {}
    for name, trial in screening.items():
        rows = trial['samples']
        if trial.get('error') or len(rows) not in (3, 6) or len(rows) != len(base):
            continue
        if [s['output_sha256'] for s in rows] != [s['output_sha256'] for s in base]:
            continue
        if any(not math.isfinite(s['seconds']) or s['seconds'] <= 0 for s in rows):
            continue
        scores[name] = [sum(s['seconds'] for s in rows)] + [
            sum(s['seconds'] for i, s in enumerate(rows) if _category(s, i) == category)
            for category in CATEGORIES]
    selected = ['standard']
    # Include the best global and per-category candidates before filling the
    # bounded shortlist. No held-out prompt participates in this selection.
    for index in range(4):
        for name in sorted(scores, key=lambda name: scores[name][index]):
            if name != 'standard' and scores[name][index] < scores.get('standard', [float('inf')] * 4)[index]:
                if name not in selected:
                    selected.append(name)
                break
    for name in sorted(scores, key=lambda name: scores[name][0]):
        if len(selected) >= limit:
            break
        if name not in selected:
            selected.append(name)
    return selected[:limit]


def verified_profiles(trials, validation):
    """Recompute every usage decision; never search for runners-up on holdout."""
    profiles = {}
    for usage, category in USAGE_PROFILES.items():
        candidate, _ = select_winner(trials, category=category)
        base = validation['standard']
        decision = ({'accepted': True, 'reason': 'La référence est conservée.'} if candidate == 'standard'
                    else assess_candidate(base, validation[candidate], category))
        winner = candidate if decision['accepted'] else 'standard'
        reference = summarize(base['samples'])
        retained = summarize(validation[winner]['samples'])
        a, b = (reference['categories'][category], retained['categories'][category]) if category else (reference, retained)
        profiles[usage] = {'candidate': candidate, 'winner': winner, 'config': trials[winner]['config'],
                           'category': category, 'decision': decision, 'baseline': a, 'retained': b,
                           'gain_percent': 100 * (a['seconds'] / b['seconds'] - 1),
                           'decode_gain_percent': 100 * (b['decode_tps'] / a['decode_tps'] - 1)}
    return profiles


def speculation_summary(samples):
    """Actual runtime counters; missing counters are unavailable, never zero."""
    timings = [s.get('timings', {}) for s in samples]
    if not timings or not all(isinstance(t.get('draft_n'), int) and isinstance(t.get('draft_n_accepted'), int) for t in timings):
        return {'proposed_tokens': None, 'accepted_tokens': None, 'acceptance_percent': None}
    proposed = sum(t['draft_n'] for t in timings)
    accepted = sum(t['draft_n_accepted'] for t in timings)
    if proposed <= 0 or not 0 <= accepted <= proposed:
        return {'proposed_tokens': None, 'accepted_tokens': None, 'acceptance_percent': None}
    return {'proposed_tokens': proposed, 'accepted_tokens': accepted, 'acceptance_percent': 100 * accepted / proposed}
