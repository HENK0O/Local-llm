"""Offline contracts and adaptation for the optional, upstream DFlash MLX API.

No checkpoint Python is imported. Drafts share the target's embeddings/tokenizer;
shape agreement alone is not evidence that a drafter was trained for a target.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path


UPSTREAM_REVISION = '07ebd93db9f472af339b644bb70221ad8428328a'
UPSTREAM_SHA256 = '2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87'
FAMILIES = {'qwen3', 'qwen3_5', 'qwen3_5_moe'}
TARGETS = {'Qwen3-4B', 'Qwen3-8B', 'Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.5-27B',
           'Qwen3.6-27B', 'Qwen3.6-35B-A3B', 'Qwen3.8-27B'}


def read_config(root):
    path = Path(root) / 'config.json'
    if path.stat().st_size > 1024 ** 2:
        raise ValueError('Configuration DFlash trop volumineuse.')
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration DFlash invalide.')
    return config


def is_draft(config):
    names = config.get('architectures')
    return isinstance(names, list) and any(name in names for name in ('DFlashDraftModel', 'DFlash2DraftModel'))


def capability():
    """Probe source without importing MLX, or trusting an arbitrary API revision."""
    try:
        spec = importlib.util.find_spec('dflash')
        source = Path(spec.origin).parent / 'model_mlx.py' if spec and spec.origin else None
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source else None
        return {'available': digest == UPSTREAM_SHA256, 'source_sha256': digest,
                'revision': UPSTREAM_REVISION,
                'reason': None if digest == UPSTREAM_SHA256 else
                'Installez la révision DFlash documentée dans un environnement MLX séparé.'}
    except (OSError, ImportError, ValueError):
        return {'available': False, 'revision': UPSTREAM_REVISION,
                'reason': 'DFlash n’est pas installé dans cet environnement MLX.'}


def _identity(root):
    from .discovery import file_provenance
    repository, _ = file_provenance(Path(root).absolute())
    return repository.split('/')[-1] if repository else Path(root).name


def draft_contract(target, draft):
    """Return candidate metadata; actual weights/outputs still require validation."""
    target, draft = Path(target).resolve(), Path(draft).resolve()
    config, aux = read_config(target), read_config(draft)
    text = config.get('text_config') or config
    if not isinstance(text, dict) or config.get('model_type') not in FAMILIES or is_draft(config):
        raise ValueError('Cette famille n’est pas prise en charge par notre intégration DFlash.')
    if not is_draft(aux) or not list(draft.glob('*.safetensors')) or aux.get('model_file'):
        raise ValueError('Un checkpoint auxiliaire DFlash local est requis.')
    for key in ('num_hidden_layers', 'num_key_value_heads', 'head_dim', 'hidden_size', 'max_position_embeddings'):
        if type(aux.get(key)) is not int or aux[key] <= 0:
            raise ValueError('Géométrie de l’auxiliaire DFlash invalide : ' + key)
    # Anchored official model names: never silently pair a named fine-tune (e.g.
    # Heretic/Ling) with the drafter of its base model, even with identical shapes.
    name = re.sub(r'-(?:MLX-)?(?:[248]bit|bf16|mxfp4)$', '', _identity(target), flags=re.I)
    aux_name = re.sub(r'-b16$', '', _identity(draft))
    method = 'dflash2' if 'DFlash2DraftModel' in aux['architectures'] else 'dflash'
    if name not in TARGETS or aux_name != name + ('-DFlash2' if method == 'dflash2' else '-DFlash'):
        raise ValueError('Aucun couple DFlash connu pour ces noms précis. Aucun auxiliaire de modèle voisin n’a été substitué.')
    if method == 'dflash2' and name != 'Qwen3.8-27B':
        raise ValueError('DFlash2 est actuellement activé uniquement pour Qwen3.8-27B.')
    for key in ('hidden_size', 'vocab_size'):
        if type(text.get(key)) is not int or text[key] != aux.get(key):
            raise ValueError('Dimensions cible/auxiliaire incompatibles : ' + key)
    layers = text.get('num_hidden_layers')
    dc = aux.get('dflash_config', {})
    ids = dc.get('target_layer_ids', []) if isinstance(dc, dict) else []
    if (type(layers) is not int or layers != aux.get('num_target_layers') or not ids
            or any(type(i) is not int or not 0 <= i < layers for i in ids) or len(set(ids)) != len(ids)
            or type(dc.get('mask_token_id')) is not int or not 0 <= dc['mask_token_id'] < text['vocab_size']):
        raise ValueError('Contrat des couches DFlash incompatible.')
    # Current upstream loader expects full-precision draft tensors. It can be
    # quantized in memory before bind(), without quantizing target embeddings.
    if aux.get('quantization') or aux.get('quantization_config'):
        raise ValueError('Utilisez le checkpoint DFlash original ; l’app quantifie l’auxiliaire en mémoire.')
    capacity = text.get('max_position_embeddings', 4096)
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('Capacité de contexte cible invalide.')
    return {'method': method, 'target': name, 'path': str(draft),
            'size_bytes': sum(p.stat().st_size for p in draft.glob('*.safetensors')),
            'max_context': min(capacity, aux['max_position_embeddings']),
            'candidate': True}


class AdaptiveDraft:
    """Adjust between requests, never change or replay an emitted answer.

    Recent timings are heuristics, not controlled speedup measurements. Compare
    decode cost only within prompt/output/sampling buckets; retain cold baseline
    probes after backoff. No acceptance percentage is inferred from upstream's
    `accepted` field, which actually includes the bonus token and EOS.
    """
    def __init__(self):
        self.block = 3
        self.cooldown = 0
        self.reason = None
        self.rates = defaultdict(lambda: {'ar': [], 'draft': []})
        self.requests = 0

    @staticmethod
    def bucket(prompt_tokens, limit, temperature):
        return (prompt_tokens // 1024, limit // 128, temperature > 0)

    def choose(self, limit):
        if limit < 32:
            return 0, 'Réponse courte : génération classique avec cache.'
        if self.cooldown:
            self.cooldown -= 1
            return 0, self.reason
        self.requests += 1
        if self.requests % 5 == 0:
            return 0, 'Contrôle périodique du coût du mode classique sur une réponse réelle.'
        return self.block, None

    def observe(self, key, block, cycles, committed, tokens, seconds):
        if tokens < 16 or not seconds or seconds <= 0:
            return
        rates = self.rates[key]
        name = 'draft' if block else 'ar'
        rates[name] = (rates[name] + [tokens / seconds])[-3:]
        if block and cycles >= 3:
            progress = committed / cycles
            self.block = 5 if progress >= 2.5 else 2 if progress < 1.5 else 3
            if progress <= 1.15:
                self.cooldown, self.reason = 3, 'Peu de tokens validés par cycle : retour temporaire au mode classique.'
        if block and min(len(rates['ar']), len(rates['draft'])) >= 2:
            if sum(rates['draft']) / len(rates['draft']) < .9 * sum(rates['ar']) / len(rates['ar']):
                self.cooldown, self.reason = 3, 'Le mode classique est plus rapide sur les requêtes récentes (estimation).'
                rates['draft'] = []
        # Bound metadata even when many unrelated context lengths are queried.
        while len(self.rates) > 32:
            del self.rates[next(iter(self.rates))]


def load_local_draft(module, root):
    """Adapt upstream's Hub-only loader to a verified, local directory.

    This process is single-flight. Only this module's resolver is temporarily
    replaced; no global Hub patch, download, repository Python, or weight copy.
    """
    root = Path(root).resolve()
    original = module.snapshot_download
    def resolve(identifier, **kwargs):
        if identifier != str(root):
            raise ValueError('Résolution DFlash non locale refusée.')
        return str(root)
    module.snapshot_download = resolve
    try:
        return module.load_draft(str(root))
    finally:
        module.snapshot_download = original


class DFlashSession:
    def __init__(self, target, draft, model, tokenizer, context):
        if not capability()['available']:
            raise ValueError('Révision DFlash MLX non prise en charge ; consultez la documentation.')
        self.contract = draft_contract(target, draft)
        if context > self.contract['max_context']:
            raise ValueError('Contexte supérieur à la capacité de l’auxiliaire DFlash.')
        from dflash import model_mlx
        import mlx.core as mx
        import mlx.nn as nn
        self.module, self.model, self.tokenizer = model_mlx, model, tokenizer
        self.draft = load_local_draft(model_mlx, draft)
        # Before bind: target embeddings/lm_head must never be quantized twice.
        nn.quantize(self.draft, group_size=64, bits=4)
        mx.eval(self.draft.parameters())
        self.adaptive = AdaptiveDraft()
        self.last = None

    def stream(self, prompt, **options):
        layers = self.module._get_layers(self.model)
        original = list(layers)
        try:
            yield from self.module.stream_generate(self.model, self.draft, self.tokenizer, prompt, **options)
        finally:
            # Upstream leaves target hooks installed. Restore the AR path and
            # release captured hidden activations, including after cancellation.
            layers[:] = original
            if hasattr(self.model, '_hidden_states'):
                delattr(self.model, '_hidden_states')
