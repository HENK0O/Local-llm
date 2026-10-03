"""Architecture geometry and observed llama.cpp allocations (not physical RAM).

Hybrid formulas follow llama-hparams.cpp / models/{qwen35,bailingmoe3}.cpp.
Unknown or incomplete metadata keeps the conservative fallback.
"""
import re

MIB = 1024 ** 2


def geometry(metadata):
    arch = metadata.get('general.architecture', '')
    def get(key, default=0):
        n = metadata.get(arch + '.' + key, default)
        return n if type(n) is int and n > 0 else default
    nextn = get('nextn_predict_layers')
    blocks = get('block_count', 32)
    layers = blocks - nextn if nextn < blocks else blocks
    heads = get('attention.head_count', 32)
    key = get('attention.key_length', get('embedding_length', 4096) // heads)
    value = get('attention.value_length', key)
    kv = metadata.get(arch + '.attention.head_count_kv', heads)
    per_layer = kv if isinstance(kv, list) else [kv] * layers
    valid = len(per_layer) >= layers and all(type(n) is int and n >= 0 for n in per_layer)
    dense = arch in {'llama', 'qwen2', 'mistral', 'gemma', 'gemma2', 'gemma3', 'phi3'}
    state, kind = 0, 'dense' if dense else 'unknown'
    total = sum(per_layer[:layers]) if valid else layers * heads
    per_token = total * (key + value) * 2
    if arch == 'bailingmoe3' and valid and get('kda.head_dim') and get('ssm.conv_kernel'):
        # MLA stores a compressed K only; KDA stores F32 convolution + matrix state.
        recurrent = sum(n == 0 for n in per_layer[:layers])
        dim, conv = get('kda.head_dim'), get('ssm.conv_kernel')
        state = recurrent * 4 * (3 * (conv - 1) * heads * dim + heads * dim * dim)
        per_token, kind = total * key * 2, 'hybrid-kda-mla'
    elif arch in {'qwen35', 'qwen35moe'} and all(get(k) for k in
            ('ssm.conv_kernel', 'ssm.inner_size', 'ssm.state_size', 'ssm.group_count', 'ssm.time_step_rank')):
        recurrent = metadata.get(arch + '.attention.recurrent_layers')
        if isinstance(recurrent, list) and len(recurrent) >= layers and all(type(n) in (bool, int) and n in (0, 1) for n in recurrent):
            count = sum(bool(n) for n in recurrent[:layers])
        else:
            interval = get('full_attention_interval', 4)
            count = layers - layers // interval
        d, inner, groups = get('ssm.state_size'), get('ssm.inner_size'), get('ssm.group_count')
        state = count * 4 * ((get('ssm.conv_kernel') - 1) * (inner + 2 * groups * d) + d * inner)
        per_token, kind = (layers - count) * get('attention.head_count_kv', heads) * (key + value) * 2, 'hybrid-delta'
    elif not dense:
        per_token = max(256 * 1024, per_token * 2)
    mtp = nextn * get('attention.head_count_kv', heads) * (key + value) * 2 if kind == 'hybrid-delta' else per_token
    return {'kv_bytes_per_token': per_token, 'mtp_kv_bytes_per_token': mtp, 'recurrent_state_bytes': state,
            'geometry': kind, 'conservative': kind == 'unknown'}


def allocations(log):
    """Only buffer declarations, never repeated aggregate summaries / tensor logs."""
    rows = []
    for line in log.splitlines():
        match = re.search(r':\s+(\S+)\s+(model|KV|RS|compute|output) buffer size\s*=\s*([\d.]+)\s+MiB', line)
        if match:
            device, kind, size = match.groups()
            rows.append({'device': device, 'kind': kind.lower(), 'bytes': round(float(size) * MIB)})
    return {'buffers': rows, 'declared_buffer_bytes': sum(r['bytes'] for r in rows) if rows else None,
            'kind': 'runtime_buffer_allocations', 'physical_ram': False}
