"""Opt-in actual Metal checks with generated, untrained sub-MiB checkpoints.

Set LOCAL_LLM_TEST_MAC_PYTHON and LOCAL_LLM_TEST_DFLASH_SOURCE to an installed
MLX interpreter and a checkout of the documented upstream revision. No package
installation, downloads, app server, or user model/cache is involved.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from local_llm.mac_runtime import worker_environment


CHECK = r'''
import importlib, json, sys
from pathlib import Path
import mlx.core as mx
from unittest.mock import patch
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from dflash.model_mlx import DFlashConfig, DFlashDraftModel, DFlash2DraftModel
from local_llm.mac_worker import Worker
from local_llm.dflash import capability
from local_llm.mac_runtime import MacRuntime
from local_llm.discovery import inspect_model
assert capability()['available'], capability()
root, family, method = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
name = 'Qwen3-4B' if family == 'qwen3' else 'Qwen3.8-27B' if method == 'DFlash2' else 'Qwen3.5-4B'
target, draft_root = root / name, root / (name + '-' + method)
target.mkdir(); draft_root.mkdir()
mx.random.seed(17)
config = dict(model_type=family, hidden_size=64, num_hidden_layers=2, intermediate_size=128,
              num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=128,
              rms_norm_eps=1e-5, rope_theta=10000., max_position_embeddings=8192,
              tie_word_embeddings=False, eos_token_id=0)
if family == 'qwen3_5':
    config.update(full_attention_interval=2, linear_num_value_heads=2, linear_num_key_heads=1,
                  linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
                  rope_parameters={'type':'default', 'rope_theta':10000., 'partial_rotary_factor':1.})
module = importlib.import_module('mlx_lm.models.' + family)
model = module.Model(module.ModelArgs.from_dict(config))
weights = {name: mx.ones_like(value) if 'norm.weight' in name else mx.random.normal(value.shape) * .1
           for name, value in tree_flatten(model.parameters())}
mx.save_safetensors(str(target / 'model.safetensors'), weights)
(target / 'config.json').write_text(json.dumps(config))
vocab = {'<eos>':0, '<bos>':1, '<unk>':2, '<pad>':3}
vocab.update({'tok'+str(i):i for i in range(4,128)})
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.WordPiece(prefix='##', cleanup=False)
hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
hf.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
hf.save_pretrained(target)
tokenizer_config = json.loads((target / 'tokenizer_config.json').read_text())
# Recent Transformers otherwise chooses Qwen3.5's BPE tokenizer from the model
# family, even though this synthetic fixture deliberately uses WordLevel.
tokenizer_config['tokenizer_class'] = 'PreTrainedTokenizerFast'
(target / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config))
aux = dict(config, architectures=[method + 'DraftModel'], num_target_layers=2,
           dflash_config={'target_layer_ids':[0,1], 'mask_token_id':127}, block_size=5)
aux.pop('rope_parameters', None)
dc = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
          head_dim=32, intermediate_size=128, vocab_size=128, rms_norm_eps=1e-5,
          rope_theta=10000., max_position_embeddings=8192, block_size=5,
          target_layer_ids=(0,1), num_target_layers=2, mask_token_id=127)
if method == 'DFlash2':
    extra = dict(conv_kernel_size=3, conv_group_size=16, selector_rank=64, selector_top_k=4)
    dc.update(extra); aux['dflash_config'].update(extra)
    # Exercise the actual DFlash2 sliding-cache path, including prompts longer
    # than its synthetic window and rollback on a rejected draft tail.
    dc.update(layer_types=('sliding_attention',)*2, sliding_window=8, is_causal=False)
    aux.update(layer_types=['sliding_attention']*2, sliding_window=8, is_causal=False)
draft = (DFlash2DraftModel if method == 'DFlash2' else DFlashDraftModel)(DFlashConfig(**dc))
weights = {name: mx.ones_like(value) if 'norm.weight' in name else mx.random.normal(value.shape) * .05
           for name, value in tree_flatten(draft.parameters())}
if method == 'DFlash2':
    for key in ('candidate_selector.predecessor_codebook', 'candidate_selector.successor_codebook'):
        weights[key] = weights.pop(key + '.weight')
mx.save_safetensors(str(draft_root / 'model.safetensors'), weights)
(draft_root / 'config.json').write_text(json.dumps(aux))
assert sum(p.stat().st_size for p in root.rglob('*.safetensors')) < 1024**2
ar = Worker(target, 'mlx', 4096, 128 << 20, 1 << 20)
spec = Worker(target, 'mlx', 4096, 128 << 20, 1 << 20, str(draft_root))
def generate(worker, text, block=3, temp=0):
    frames = []
    worker.generate(dict(messages=[{'role':'user', 'content':text}], max_tokens=64,
                         temperature=temp, top_p=.9 if temp else 1., top_k=8 if temp else 0,
                         seed=42, _calibration=True, block_size=block), frames.append)
    return frames[-1]
hashes = set()
for i, block in enumerate((2,3,5)):
    text = 'tok4 tok5 tok6 tok7' if i < 2 else 'tok8 tok9 tok10 tok11'
    cold, accelerated = generate(ar, text), generate(spec, text, block)
    assert cold['sample']['output_sha256'] == accelerated['sample']['output_sha256'], (family, method, block, cold, accelerated)
    assert cold['usage'] == accelerated['usage']
    assert accelerated['speculation']['last_request']['method'].startswith('dflash')
    assert accelerated['speculation']['last_request']['cycles'] > 0
    assert accelerated['timings']['cache_n'] == 0
    assert not hasattr(spec.model, '_hidden_states')
    hashes.add(cold['sample']['output_sha256'])
assert len(hashes) > 1, 'Fixture must produce input-dependent answers'
# Longer hybrid histories can hit near-ties: Metal's block GEMM and sequential
# GEMV differ numerically. Do not pretend every token sequence stays identical.
# Instead cross-check EVERY verify-block logit against a full, uncached target
# evaluation of the same prefix. A stale or wrongly rolled-back cache must fail.
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import GatedDeltaNet
emitted, prompt_ids, errors = [], [], []
original_stream = spec.dflash.stream
original_call, original_gdn = type(spec.model).__call__, GatedDeltaNet.__call__
def traced_stream(prompt, **kwargs):
    emitted.clear(); prompt_ids[:] = prompt
    for response in original_stream(prompt, **kwargs):
        emitted.extend(response.tokens)
        yield response
def checked_call(self, inputs, cache=None, **kwargs):
    logits = original_call(self, inputs, cache=cache, **kwargs)
    if self is spec.model and emitted and inputs.shape[1] > 1:
        # Avoid letting the upstream global GDN capture record oracle work.
        patched_gdn = GatedDeltaNet.__call__
        GatedDeltaNet.__call__ = original_gdn
        try:
            ids = prompt_ids + emitted[:-1] + inputs[0].tolist()
            reference = original_call(ar.model, mx.array([ids]), cache=make_prompt_cache(ar.model))[:, -inputs.shape[1]:]
            mx.eval(reference, logits)
        finally:
            GatedDeltaNet.__call__ = patched_gdn
        error = float(mx.max(mx.abs(reference - logits)))
        assert error < .01, ('cache rollback/causality error', error)
        reference_top = mx.argmax(reference, axis=-1)
        actual_top = mx.argmax(logits, axis=-1)
        sorted_values = mx.sort(reference, axis=-1)
        margins = sorted_values[..., -1] - sorted_values[..., -2]
        assert bool(mx.all((reference_top == actual_top) | (margins <= 2*error)))
        errors.append(error)
    return logits
spec.dflash.stream = traced_stream
type(spec.model).__call__ = checked_call
try:
    long_response = generate(spec, ' '.join('tok'+str(n) for n in range(8,28)), 5)
    assert errors and long_response['speculation']['last_request']['cycles'] > 0
finally:
    type(spec.model).__call__ = original_call
    spec.dflash.stream = original_stream
    assert GatedDeltaNet.__call__ is original_gdn
# Deterministically exercise partial acceptance 0/1/2/3; a random untrained
# drafter otherwise often rejects everything and would miss offset errors.
if family == 'qwen3_5':
    for accepted in range(4):
        cache = make_prompt_cache(spec.model)
        spec.model(mx.array([[4,5,6,7]]), cache=cache)
        mx.eval([layer.state for layer in cache])
        capture = spec.dflash.module._GDNStateCapture()
        try:
            spec.model(mx.array([[8,9,10,11,12]]), cache=cache)
            mx.eval([layer.state for layer in cache])
            capture.rollback(cache, accepted, 5-accepted-1)
            mx.eval([layer.state for layer in cache])
        finally:
            capture.close()
        reference = make_prompt_cache(ar.model)
        ar.model(mx.array([[4,5,6,7]]), cache=reference)
        ar.model(mx.array([[8,9,10,11,12][:accepted+1]]), cache=reference)
        mx.eval([layer.state for layer in reference])
        for actual, expected in zip(cache, reference):
            if actual.is_trimmable():
                assert actual.size() == expected.size() == 4+accepted+1
                # Allocator capacity can differ; compare the live KV region.
                for a,b in zip(actual.state, expected.state):
                    assert float(mx.max(mx.abs(a[...,:actual.size(),:] - b[...,:expected.size(),:]))) < .01
            else:
                for a,b in zip(actual.state, expected.state):
                    assert a.shape == b.shape
                    assert float(mx.max(mx.abs(a-b))) < .01
# A later AR fallback must still work with its own conversation cache.
frames = []
spec.generate(dict(messages=[{'role':'user', 'content':'tok12 tok13'}], max_tokens=16,
                   temperature=0, conversation='fallback'), frames.append)
assert frames[-1]['speculation']['last_request']['method'] == 'autoregressive'
assert frames[-1]['conversation_cache']['entries'] == 1
stochastic = generate(spec, 'tok4 tok5', temp=.7)
assert stochastic['usage']['completion_tokens'] > 0
assert stochastic['usage']['completion_tokens'] <= 64
# Verify private-worker transport and activation with the same tiny files.
# Preserve the test-only source path if DFlash came from a checkout rather than
# site-packages; production workers deliberately reset PYTHONPATH themselves.
with patch('local_llm.mac_runtime.worker_environment', return_value=dict(__import__('os').environ)):
    runtime = MacRuntime(root / 'state')
    runtime.library_paths = [str(draft_root)]
    runtime.memory_probe = lambda: 4 << 30
    try:
        runtime.load(inspect_model(target), 4 << 30, 'mlx')
        before = runtime.process.pid
        runtime.configure_speculation(draft_root)
        assert runtime.process.pid != before
        assert runtime.describe()['speculation']['enabled']
        frames = list(runtime.iter_chat(dict(model=runtime.model_id,
                      messages=[{'role':'user','content':'tok4 tok5 tok6 tok7'}],
                      max_tokens=64, temperature=0), 'ipc'))
        assert frames[-1]['speculation']['last_request']['method'].startswith('dflash')
        assert frames[-1]['usage']['completion_tokens'] > 0
        runtime.configure_speculation(None)
        assert not runtime.describe()['speculation']['enabled']
        frames = list(runtime.iter_chat(dict(model=runtime.model_id,
                      messages=[{'role':'user','content':'tok4 tok5'}],
                      max_tokens=16, temperature=0), 'ipc'))
        assert frames[-1]['conversation_cache']['entries'] == 1
    finally:
        runtime.close()
    assert runtime.process is None
print(json.dumps({'family':family, 'method':method, 'greedy_parity':True,
                  'long_verify_max_logit_error':max(errors),
                  'stochastic_smoke':True, 'fallback_cache':True}))
'''


@unittest.skipUnless(os.environ.get('LOCAL_LLM_TEST_MAC_PYTHON') and os.environ.get('LOCAL_LLM_TEST_DFLASH_SOURCE'),
                     'Set installed MLX Python and audited DFlash checkout for real Metal checks')
class RealDFlashTests(unittest.TestCase):
    def test_dflash_and_dflash2_attention_hybrid_greedy_parity_and_fallback(self):
        for family, method in (('qwen3', 'DFlash'), ('qwen3_5', 'DFlash'), ('qwen3_5', 'DFlash2')):
            with self.subTest(family=family, method=method), tempfile.TemporaryDirectory(prefix='local-llm-dflash-metal-') as folder:
                source = Path(folder) / 'check.py'; source.write_text(CHECK)
                env = worker_environment()
                env['PYTHONPATH'] += os.pathsep + os.environ['LOCAL_LLM_TEST_DFLASH_SOURCE']
                result = subprocess.run([os.environ['LOCAL_LLM_TEST_MAC_PYTHON'], str(source), folder, family, method],
                                        capture_output=True, text=True, env=env, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr[-6000:])
                self.assertTrue(json.loads(result.stdout)['greedy_parity'])


if __name__ == '__main__':
    unittest.main()
