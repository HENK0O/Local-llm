import json
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from local_llm.accelerator import ExecutionConfig, SlotPool
from local_llm.server import ChatService, LocalLLMHTTPServer, parse_chat_request


class FakeRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = ExecutionConfig()
        self.profile = None
        self.usage_profile = "balanced"
        self.slots = SlotPool()
        self.model_id, self.model_name, self.path = 'target', 'Target GGUF', None
        self.loaded = True
        self.job = None
        self.cancelled = threading.Event()
        self.closed = False
        self.payload = None

    def active_measurement(self): return None
    def set_usage_profile(self, usage):
        if usage not in {'balanced','discussion','code','long_context'}: raise ValueError('Profil inconnu')
        self.usage_profile = usage
        return self.describe()
    def configure_cache(self, enabled=None, clear=False):
        self.cache_settings = {'enabled':enabled, 'clear':clear}
        return self.describe()
    def available(self): return {'available': True, 'gpu': True}
    def describe(self):
        return dict(self.available(), loaded=self.loaded, model_id=self.model_id, model_name=self.model_name, job=self.job)
    def load(self, item, available):
        self.model_id, self.model_name = item.id, item.name
        self.loaded = True
        return self.describe()
    def unload(self): self.loaded = False
    def close(self): self.closed = True
    def _rss(self): return 1000
    def optimize(self, draft=None, drafts=None):
        self.job = {'state': 'running', 'progress': 0}
        return self.job
    def context(self, messages):
        return {'prompt': 'EXACT: ' + messages[0]['content'], 'prompt_tokens': 10,
                'model': self.model_name, 'context_length': 4096, 'compression': 'none'}
    def iter_chat(self, payload, conversation):
        self.payload = payload
        self.slots.acquire(conversation)
        yield {'model': self.model_id, 'choices': [{'index': 0, 'delta': {'content': 'Answer'}}]}
        yield {'model': self.model_id, 'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
               'timings': {'cache_n': 7, 'prompt_ms': 10, 'predicted_ms': 50, 'predicted_per_second': 80}}


class AcceleratorHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('local_llm.server.default_model_roots', return_value=[]): cls.service = ChatService()
        cls.runtime = cls.service.accelerator = FakeRuntime()
        cls.server = LocalLLMHTTPServer(('127.0.0.1', 0), cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(5)
        if not cls.runtime.closed: raise AssertionError('Owned runtime not closed')

    def setUp(self):
        self.runtime.loaded = True; self.runtime.model_id = 'target'; self.runtime.job = None
        self.runtime.cancelled.clear()
        self.runtime.usage_profile = 'balanced'

    def request(self, path, payload=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin: headers['Origin'] = origin
        request = Request(self.url + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        return urlopen(request, timeout=5)

    def test_stream_has_real_engine_metrics_actual_cache_work_and_exact_context(self):
        with self.request('/v1/chat/completions', {'backend': 'llamacpp', 'model': 'target', 'stream': True,
            'conversation_id': 'chat-a', 'max_tokens': 64, 'messages': [{'role': 'user', 'content': 'Question'}]}) as response:
            frames = [line[6:] for line in response.read().decode().splitlines() if line.startswith('data: ')]
        self.assertEqual(frames[-1], '[DONE]')
        data = json.loads(frames[-2]); stats = data['local_llm']
        self.assertEqual(stats['backend'], 'llamacpp')
        self.assertEqual(stats['decode_tokens_per_second'], 80)
        self.assertEqual(stats['reused_prompt_tokens'], 7)
        self.assertGreaterEqual(stats['first_token_seconds'], 0)
        self.assertGreaterEqual(stats['first_text_seconds'], stats['first_token_seconds'])
        self.assertGreater(stats['request_seconds'], stats['first_text_seconds'])
        self.assertIsNone(stats['calibration_gain_percent'])
        self.assertFalse(stats['optimized'])
        self.assertEqual(stats['optimization_profile'], 'balanced')
        self.assertIsNone(stats['draft_proposed_tokens'])
        self.assertIn('chat-a', self.runtime.slots.entries)
        with self.request('/v1/context', {'completion_id': data['id']}) as response:
            snapshot = json.load(response)
        self.assertEqual(snapshot['prompt'], 'EXACT: Question')
        self.assertEqual(snapshot['prompt_tokens'], 10)
        self.assertEqual(snapshot['model'], 'Target GGUF')
        with self.request('/v1/context', {'backend': 'llamacpp', 'model': 'target', 'stream': True,
                         'messages': [{'role': 'user', 'content': 'Preview'}]}) as response:
            self.assertEqual(json.load(response)['prompt'], 'EXACT: Preview')

    def test_usage_profile_endpoint_validates_and_applies_only_named_profiles(self):
        with self.request('/v1/accelerator/profile', {'profile':'code'}) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.runtime.usage_profile,'code')
        for payload in ({'profile':'--external'}, {}, {'profile':3}):
            with self.assertRaises(HTTPError) as error: self.request('/v1/accelerator/profile', payload)
            self.assertEqual(error.exception.code,400)

    def test_disk_cache_controls_validate_types_and_do_not_accept_paths(self):
        with self.request('/v1/accelerator/cache', {'enabled':False, 'clear':True}) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.runtime.cache_settings, {'enabled':False, 'clear':True})
        for payload in ({'enabled':1}, {'clear':'yes'}, {'path':'/private/file'}, []):
            with self.assertRaises(HTTPError) as error: self.request('/v1/accelerator/cache', payload)
            self.assertEqual(error.exception.code,400)

    def test_model_advice_accepts_only_catalog_ids_and_import_is_same_origin(self):
        selected = SimpleNamespace(id='target')
        self.service.catalog = {'target': selected}
        with patch.object(self.service.advisor, 'get', return_value={'state':'ready','model_id':'target'}) as advice:
            with self.request('/v1/model-advice?model_id=target') as response:
                self.assertEqual(json.load(response)['state'],'ready')
            self.assertIs(advice.call_args[0][0], selected)
        for query in ('', '?model_id=/private/file.gguf', '?model_id=unknown'):
            with self.assertRaises(HTTPError) as error: self.request('/v1/model-advice' + query)
            self.assertEqual(error.exception.code,400)
        with self.assertRaises(HTTPError) as error:
            self.request('/v1/local-models/import', {'path':'/private/model.gguf'}, 'https://evil.invalid')
        self.assertEqual(error.exception.code,403)

    def test_unloaded_wrong_model_nonstream_and_calibration_never_silently_route_elsewhere(self):
        payload = {'backend': 'llamacpp', 'model': 'target', 'stream': True,
                   'messages': [{'role': 'user', 'content': 'Question'}]}
        for change in ({'model': 'other'}, {'stream': False}):
            with self.assertRaises(HTTPError) as error: self.request('/v1/chat/completions', dict(payload, **change))
            self.assertEqual(error.exception.code, 400)
        self.runtime.loaded = False
        with self.assertRaises(HTTPError): self.request('/v1/chat/completions', payload)
        self.runtime.loaded = True; self.runtime.job = {'state': 'running'}
        with self.assertRaises(HTTPError): self.request('/v1/chat/completions', payload)

    def test_accelerator_endpoints_use_catalog_ids_and_protect_mutations(self):
        self.service.catalog = {'target': SimpleNamespace(id='target', name='Target GGUF', path='/tmp/a.gguf', architecture='llama')}
        with self.request('/v1/accelerator') as response: self.assertTrue(json.load(response)['available'])
        with self.request('/v1/models') as response: self.assertEqual(json.load(response)['data'][0]['id'], 'target')
        with self.request('/v1/accelerator/drafts') as response: self.assertEqual(json.load(response)['models'], [])
        with self.request('/v1/accelerator/load', {'id': 'target'}) as response: self.assertTrue(json.load(response)['loaded'])
        with self.assertRaises(HTTPError) as error: self.request('/v1/accelerator/load', {'id': '/arbitrary/model.gguf'})
        self.assertEqual(error.exception.code, 400)
        for path in ('load', 'unload', 'optimize', 'cancel', 'profile', 'cache'):
            with self.assertRaises(HTTPError) as error: self.request('/v1/accelerator/' + path, {}, 'https://evil.invalid')
            self.assertEqual(error.exception.code, 403)
        with self.request('/v1/accelerator/optimize', {}) as response:
            self.assertEqual(response.status, 202); self.assertEqual(json.load(response)['state'], 'running')
        with self.request('/v1/accelerator/cancel', {}) as response: json.load(response)
        self.assertTrue(self.runtime.cancelled.is_set())
        self.runtime.job = None
        with self.request('/v1/accelerator/unload', {}) as response: self.assertFalse(json.load(response)['loaded'])

    def test_native_default_api_does_not_spawn_external_workers(self):
        with patch('local_llm.server.default_model_roots', return_value=[]), patch('local_llm.accelerator.subprocess.Popen') as start:
            service = ChatService()
        start.assert_not_called()
        service.telemetry.close()
        with self.assertRaises(ValueError): parse_chat_request({'conversation_id': None, 'messages': [{'role': 'user', 'content': 'x'}]})

    def test_auto_boot_prefers_installed_direct_runtime_without_loading_duplicate_cpu_weights(self):
        item = SimpleNamespace(id='target', name='Target GGUF', path='/tmp/installed.gguf', architecture='llama', compatible=True, size_bytes=1)
        runtime = FakeRuntime()
        with patch('local_llm.server.default_model_roots', return_value=[]), patch('local_llm.server.discover_models', return_value=[item]), patch('local_llm.server.Accelerator', return_value=runtime), patch('local_llm.server.load_runtime') as native:
            service = ChatService(engine='auto')
        native.assert_not_called()
        self.assertIsNone(service.model)
        self.assertTrue(service.accelerator.describe()['loaded'])
        service.telemetry.close()

    def test_ram_process_scope_includes_owned_worker_and_does_not_hide_unavailable_values(self):
        with patch.object(self.service.telemetry, 'snapshot', return_value={'process_rss_bytes': 2000}):
            result = self.service.system_snapshot()
        self.assertEqual(result['process_rss_bytes'], 3000)
        self.assertEqual(result['inference_process_rss_bytes'], 1000)
        self.assertIn('partagées', result['process_note'])
        with patch.object(self.service.telemetry, 'snapshot', return_value={'process_rss_bytes': 2000}), patch.object(self.runtime, '_rss', return_value=None):
            self.assertIsNone(self.service.system_snapshot()['process_rss_bytes'])


class DraftSearchTests(unittest.TestCase):
    def test_auto_drafts_are_ranked_bounded_and_fit_spare_memory(self):
        with patch('local_llm.server.default_model_roots', return_value=[]):
            service = ChatService()
        runtime = service.accelerator = FakeRuntime()
        runtime.optimize = Mock(wraps=runtime.optimize)
        gib = 1024 ** 3
        entries = [{'id':'large','size_bytes':2*gib}, {'id':'second','size_bytes':300*1024**2},
                   {'id':'smallest','size_bytes':100*1024**2}, {'id':'third','size_bytes':400*1024**2}]
        service.catalog = {m['id']:SimpleNamespace(path='/tmp/' + m['id'] + '.gguf') for m in entries}
        try:
            with patch.object(service, 'accelerator_drafts', return_value={'models':entries}), patch.object(service.telemetry, 'snapshot', return_value={'memory_total_bytes':10*gib,'memory_used_bytes':8*gib}):
                service.optimize_accelerator({})
            runtime.optimize.assert_called_once_with(drafts=['/tmp/smallest.gguf','/tmp/second.gguf'])
            self.assertEqual(runtime.job['draft_search']['compatible'], 4)
            self.assertEqual(runtime.job['draft_search']['tested'], 2)
            runtime.optimize.reset_mock()
            with patch.object(service, 'accelerator_drafts') as discover:
                service.optimize_accelerator({'draft_mode':'off'})
            discover.assert_not_called()
            runtime.optimize.assert_called_once_with(drafts=[])
        finally:
            service.telemetry.close()


if __name__ == '__main__': unittest.main()
