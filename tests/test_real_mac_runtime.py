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


@unittest.skipUnless(os.environ.get('LOCAL_LLM_TEST_MAC_PYTHON'), 'Set LOCAL_LLM_TEST_MAC_PYTHON to an installed MLX Python for tiny real Metal checks')
class RealMacTests(unittest.TestCase):
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
                self.assertTrue(runtime.describe()['loaded'])
                messages += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': 'Répète le mot'}]
                preview = runtime.context(messages)
                self.assertIn(text, preview['prompt']); self.assertIn('Répète le mot', preview['prompt'])
                partial = runtime.iter_chat(dict(payload, messages=messages), 'fixture')
                next(partial); partial.close()
                self.assertIsNone(runtime.process)
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
