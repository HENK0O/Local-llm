"""Opt-in real Metal test with a generated checkpoint smaller than 1 MiB.

No trained model download; this fixture tests transport/lifecycle and template
handling, never useful-model quality or performance. The main app stays closed.
"""
import json
import os
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from local_llm.discovery import inspect_model
from local_llm.mac_runtime import MacRuntime, worker_environment
from local_llm.telemetry import SystemTelemetry


CREATE_FIXTURE = '''
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
root = Path(sys.argv[1])
config = dict(model_type="llama", hidden_size=32, num_hidden_layers=1, intermediate_size=64,
              num_attention_heads=2, num_key_value_heads=1, vocab_size=16, rms_norm_eps=1e-5,
              rope_theta=10000., max_position_embeddings=8192, tie_word_embeddings=False, eos_token_id=0)
model = Model(ModelArgs.from_dict(config))
weights = {}
for name, value in tree_flatten(model.parameters()):
    weights[name] = mx.ones_like(value) if 'norm.weight' in name or name == 'model.embed_tokens.weight' else mx.zeros_like(value)
weights['lm_head.weight'][4] = mx.ones((32,))
mx.save_safetensors(str(root/'model.safetensors'), weights)
(root/'config.json').write_text(json.dumps(config))
vocab = {'<eos>':0, '<bos>':1, '<unk>':2, '<pad>':3, 'Bonjour':4}
vocab.update({'tok'+str(i):i for i in range(5,16)})
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.WordPiece(prefix='##', cleanup=False)
hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
hf.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\\n{% endfor %}assistant:"
hf.save_pretrained(root)
'''


CHECK_CACHE = '''
import json, sys
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from local_llm.mac_worker import Worker
worker = Worker(sys.argv[1], 'mlx', 4096, 64 << 20, 256 << 10)
history = [{'role':'user', 'content':'Bonjour tok5 tok6 tok7'}]
def complete(messages, key, use_cache=True):
    frames = []
    worker.generate(dict(messages=messages, conversation=key, use_cache=use_cache,
                         max_tokens=8, temperature=0, seed=42), frames.append)
    text = ''.join(f['choices'][0]['delta'].get('content','') for f in frames)
    return text, frames[-1]
text, first = complete(history, 'a')
assert first['timings']['cache_n'] == 0
history += [{'role':'assistant', 'content':text}, {'role':'user','content':'tok8 tok9 Bonjour'}]
preview, ids = worker.context(history)
layers, reused = worker.conversations.prepare('a', ids)
assert reused > 0
worker._prefill_prefix(ids[reused:-1], layers, 128)
warm_logits = worker.model(mx.array(ids[-1:])[None], cache=layers)[:, -1, :]
cold_logits = worker.model(mx.array(ids)[None], cache=make_prompt_cache(worker.model))[:, -1, :]
mx.eval(warm_logits, cold_logits)
error = float(mx.max(mx.abs(warm_logits - cold_logits)))
# Full-prompt GEMM and last-token GEMV use different Metal arithmetic. Compare
# that existing cold-path variation too; cached state must not add a large one.
chunk_cache = make_prompt_cache(worker.model)
worker._prefill_prefix(ids[:-1], chunk_cache, 128)
chunk_logits = worker.model(mx.array(ids[-1:])[None], cache=chunk_cache)[:, -1, :]
mx.eval(chunk_logits)
arithmetic_error = float(mx.max(mx.abs(chunk_logits - cold_logits)))
assert error <= max(1e-5, 4 * arithmetic_error) and error < .01, (error, arithmetic_error)
# Input dependence: a fixed-output fixture cannot validate cache correctness.
other = worker.model(mx.array([4, 5, 6])[None], cache=make_prompt_cache(worker.model))[:, -1, :]
mx.eval(other)
assert float(mx.max(mx.abs(other - cold_logits))) > 1e-5
complete(history[:1], 'a')
text, warm = complete(history, 'a')
_, cold = complete(history, 'cold', False)
assert warm['sample']['output_sha256'] == cold['sample']['output_sha256']
assert warm['timings']['cache_n'] > 0 and cold['timings']['cache_n'] == 0
assert warm['timings']['prompt_n'] + warm['timings']['cache_n'] == len(ids)
_, isolated = complete(history, 'b')
assert isolated['timings']['cache_n'] == 0
_, resumed = complete(history, 'a')
assert resumed['timings']['cache_n'] > 0
changed = [dict(m) for m in history]
changed[0]['content'] = 'Bonjour tok15 tok6 tok7'
_, edited = complete(changed, 'a')
_, edited_cold = complete(changed, 'cold', False)
assert edited['sample']['output_sha256'] == edited_cold['sample']['output_sha256']
if not worker.conversations.can_trim(make_prompt_cache(worker.model)):
    assert edited['timings']['cache_n'] == 0
print(json.dumps(dict(max_logit_error=error, cold_shape_error=arithmetic_error, reused_tokens=warm['timings']['cache_n'],
                     processed_tokens=warm['timings']['prompt_n'], input_tokens=len(ids))))
'''


@unittest.skipUnless(os.environ.get('LOCAL_LLM_TEST_MAC_PYTHON'), 'Set LOCAL_LLM_TEST_MAC_PYTHON to an installed MLX Python for tiny real Metal checks')
class RealMacTests(unittest.TestCase):
    def test_attention_and_hybrid_cache_preserve_input_dependent_logits_and_tokens(self):
        for architecture in ('llama', 'qwen3_5'):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory(prefix='local-llm-cache-metal-') as folder:
                root = Path(folder); model = root / 'model'; model.mkdir()
                creator = CREATE_FIXTURE.replace(
                    "model = Model(ModelArgs.from_dict(config))", "mx.random.seed(17)\nmodel = Model(ModelArgs.from_dict(config))")
                creator = creator.replace(
                    "mx.ones_like(value) if 'norm.weight' in name or name == 'model.embed_tokens.weight' else mx.zeros_like(value)",
                    "mx.ones_like(value) if 'norm.weight' in name else mx.random.normal(value.shape) * .1")
                creator = creator.replace("weights['lm_head.weight'][4] = mx.ones((32,))", "\nfor name in weights:\n    if name.endswith('lm_head.weight'): weights[name][0] = mx.zeros((32,))")
                if architecture == 'qwen3_5':
                    creator = creator.replace('mlx_lm.models.llama', 'mlx_lm.models.qwen3_5').replace('model_type="llama"', 'model_type="qwen3_5"')
                    creator = creator.replace('eos_token_id=0)', 'eos_token_id=0)\nconfig.update(num_hidden_layers=2, full_attention_interval=2, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, rope_parameters={"type":"default", "rope_theta":10000., "partial_rotary_factor":1.0})')
                source = root / 'create.py'; source.write_text(creator)
                python = os.environ['LOCAL_LLM_TEST_MAC_PYTHON']
                result = subprocess.run([python, str(source), str(model)], env=worker_environment(), capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                self.assertLess(inspect_model(model).size_bytes, 1024 ** 2)
                check = root / 'check.py'; check.write_text(CHECK_CACHE)
                result = subprocess.run([python, str(check), str(model)], env=worker_environment(), capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                data = json.loads(result.stdout)
                self.assertLess(data['max_logit_error'], .01)
                self.assertLessEqual(data['max_logit_error'], max(1e-5, 4 * data['cold_shape_error']))
                self.assertGreater(data['reused_tokens'], 0)
                self.assertLess(data['processed_tokens'], data['input_tokens'])

    def test_real_mlx_stream_context_interrupt_reload_and_independent_calibration(self):
        with tempfile.TemporaryDirectory(prefix='local-llm-tiny-mlx-') as folder, patch.dict(
                os.environ, {'LOCAL_LLM_MLX_PYTHON': os.environ['LOCAL_LLM_TEST_MAC_PYTHON']}):
            root = Path(folder); model = root / 'model'; model.mkdir()
            source = root / 'create.py'; source.write_text(CREATE_FIXTURE)
            subprocess.run([os.environ['LOCAL_LLM_TEST_MAC_PYTHON'], str(source), str(model)],
                           env=worker_environment(), check=True, capture_output=True, timeout=30)
            self.assertLess(inspect_model(model).size_bytes, 1024 ** 2)
            runtime = MacRuntime(root / 'state')
            telemetry = SystemTelemetry()
            def available():
                state = telemetry.snapshot(refresh=True)
                return state['memory_total_bytes'] - state['memory_used_bytes'] + (runtime._rss() or 0)
            runtime.memory_probe = available
            try:
                runtime.load(inspect_model(model), available(), 'mlx')
                messages = [{'role': 'user', 'content': 'Bonjour'}]
                payload = {'model': runtime.model_id, 'messages': messages, 'max_tokens': 16, 'temperature': 0}
                frames = list(runtime.iter_chat(payload, 'fixture'))
                text = ''.join(c.get('delta', {}).get('content', '') for f in frames for c in f.get('choices', []))
                self.assertIn('Bonjour', text)
                self.assertEqual(frames[-1]['usage']['completion_tokens'], 16)
                self.assertGreater(frames[-1]['timings']['predicted_per_second'], 0)
                self.assertEqual(frames[-1]['timings']['cache_n'], 0)
                self.assertEqual(runtime.describe()['cached_conversations'], 1)
                self.assertTrue(runtime.describe()['loaded'])
                messages += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': 'Répète le mot'}]
                preview = runtime.context(messages)
                self.assertIn(text, preview['prompt']); self.assertIn('Répète le mot', preview['prompt'])
                warm = list(runtime.iter_chat(dict(payload, messages=messages), 'fixture'))
                self.assertGreater(warm[-1]['timings']['cache_n'], 0)
                self.assertEqual(warm[-1]['timings']['prompt_n'] + warm[-1]['timings']['cache_n'],
                                 warm[-1]['usage']['prompt_tokens'])
                cold = list(runtime.iter_chat(dict(payload, messages=messages, use_cache=False), 'fixture'))
                self.assertEqual(cold[-1]['timings']['cache_n'], 0)
                self.assertEqual(''.join(c.get('delta', {}).get('content', '') for f in warm for c in f.get('choices', [])),
                                 ''.join(c.get('delta', {}).get('content', '') for f in cold for c in f.get('choices', [])))
                original_pid = runtime.process.pid
                with runtime.lock:
                    runtime._ensure_capacity(5000)
                self.assertEqual(runtime.config.context, 8192)
                self.assertEqual(runtime.process.pid, original_pid)
                self.assertGreater(runtime.describe()['cached_conversations'], 0)
                self.assertEqual(runtime.context(messages)['context_length'], 8192)
                resumed_cache = list(runtime.iter_chat(dict(payload, messages=messages), 'fixture'))
                self.assertGreater(resumed_cache[-1]['timings']['cache_n'], 0)
                partial = runtime.iter_chat(dict(payload, messages=messages), 'fixture')
                next(partial); partial.close()
                self.assertIsNone(runtime.process)
                self.assertEqual(runtime.describe()['cached_conversations'], 0)
                runtime.load(inspect_model(model), available(), 'mlx')
                self.assertIn('usage', list(runtime.iter_chat(payload, 'fixture'))[-1])
                runtime.optimize()
                deadline = time.monotonic() + 90
                while runtime.job['state'] == 'running' and time.monotonic() < deadline:
                    time.sleep(.1)
                self.assertEqual(runtime.job['state'], 'done', runtime.job)
                report = runtime.profile
                self.assertEqual(len(report['training']['standard']['samples']), 6)
                self.assertEqual(len(report['validation']['standard']['samples']), 9)
                self.assertTrue(runtime._validate_report(report))
                runtime.unload()
                runtime.load(inspect_model(model), available(), 'auto')
                self.assertIsNotNone(runtime.profile)
                self.assertEqual(runtime.profile['model_sha256'], report['model_sha256'])
            finally:
                runtime.close()
            self.assertIsNone(runtime.process)


if __name__ == '__main__':
    unittest.main()
