"""Distinguish tensor containers from runtime-specific quantization formats."""
import math


UNSUPPORTED_FORMATS = {'EXL3', 'EXL2', 'GPTQ', 'AWQ', 'BitsAndBytes'}


def directory_format(config, has_npz=False):
    text = config.get('text_config')
    for section in (config, text if isinstance(text, dict) else {}):
        for key in ('quantization_config', 'quantization'):
            quant = section.get(key)
            if not isinstance(quant, dict):
                continue
            for field in ('quant_method', 'checkpoint_format', 'format', 'method'):
                value = str(quant.get(field, '')).lower()
                if value in {'exl3', 'exllamav3'}:
                    return 'EXL3'
                if value in {'exl2', 'exllamav2'}:
                    return 'EXL2'
                if value in {'gptq', 'gptqmodel'}:
                    return 'GPTQ'
                if value == 'awq':
                    return 'AWQ'
                if value in {'bitsandbytes', 'bnb'}:
                    return 'BitsAndBytes'
    return 'NPZ' if has_npz else 'Safetensors'


def format_reason(value):
    if value == 'EXL3':
        return ('EXL3 nécessite un runtime dédié, non intégré à local-llm. '
                'Des moteurs comme MLXL3 savent le lire sur Apple Silicon. '
                'Pour notre moteur actuel, choisissez une version MLX ou GGUF compatible ; '
                'ce blocage de format n’est pas un manque de RAM.')
    if value == 'EXL2':
        return (value + ' nécessite un runtime dédié, qui n’est pas intégré à local-llm. '
                'Sur Mac Apple Silicon, choisissez une version MLX ou GGUF du même modèle. '
                'Ce format n’est pas pris en charge par notre moteur Metal ; ce n’est pas un manque de RAM.')
    return value + ' n’est pas pris en charge par les moteurs intégrés. Choisissez une version MLX ou GGUF compatible.'


def quantization_label(config, format_name):
    text = config.get('text_config')
    section = text if isinstance(text, dict) and not (config.get('quantization_config') or config.get('quantization')) else config
    quant = section.get('quantization_config') or section.get('quantization') or {}
    if isinstance(quant, dict):
        if format_name in {'EXL3', 'EXL2'}:
            for key in ('avg_bits', 'decoder_bpw', 'bits_avg', 'bits'):
                value = quant.get(key)
                if type(value) in (int, float) and math.isfinite(value) and 0 < value <= 16:
                    return '%s · %.2g bpw' % (format_name, value)
            return format_name
        if type(quant.get('bits')) is int:
            return str(quant['bits']) + '-bit' + (' MLX' if format_name == 'Safetensors' else ' ' + format_name)
    return format_name if format_name in UNSUPPORTED_FORMATS else 'F32/F16'


def model_availability(item, engines, runtimes):
    """Readiness describes detection, never promises a successful load/RAM fit."""
    fmt = getattr(item, 'format', 'unknown')
    if fmt in UNSUPPORTED_FORMATS:
        return {'code': 'unsupported_format', 'label': 'Format incompatible', 'reason': format_reason(fmt)}
    if getattr(item, 'role', 'model') == 'auxiliary':
        return {'code': 'auxiliary', 'label': 'Auxiliaire',
                'reason': 'S’utilise avec son modèle cible dans Optimisations, pas comme modèle de discussion.'}
    if engines or item.compatible:
        return {'code': 'candidate', 'label': 'À charger',
                'reason': 'La compatibilité et la mémoire disponible seront vérifiées au chargement.'}
    missing = fmt == 'GGUF' and not runtimes.get('llamacpp', {}).get('available')
    missing = missing or fmt == 'Safetensors' and not runtimes.get('mlx', {}).get('available')
    if missing:
        return {'code': 'missing_runtime', 'label': 'Moteur manquant',
                'reason': 'Installez llama.cpp pour ce GGUF.' if fmt == 'GGUF' else 'Un environnement MLX est nécessaire sur Mac Apple Silicon.'}
    return {'code': 'unavailable', 'label': 'Non pris en charge', 'reason': item.reason or 'Checkpoint incomplet ou architecture non prise en charge.'}
