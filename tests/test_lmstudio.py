import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from local_llm.lmstudio import LMStudioClient


class LMStudioTests(unittest.TestCase):
    def test_rejects_remote_urls_credentials_and_extra_paths(self):
        for url in ['https://127.0.0.1:1234', 'http://example.com', 'http://127.0.0.1.evil',
                    'http://user:password@localhost:1234', 'http://localhost:1234/path',
                    'http://localhost:1234?secret=x']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LMStudioClient(url)

    def test_lists_v1_llms_without_embeddings(self):
        client = LMStudioClient()
        with patch.object(client, '_request', return_value={'models': [
            {'type': 'llm', 'key': 'test', 'display_name': 'Test', 'quantization': {'name': 'Q8_0'}, 'loaded_instances': [{'id': 'test'}]},
            {'type': 'embedding', 'key': 'embed'},
        ]}):
            result = client.models()
        self.assertTrue(result['available'])
        self.assertEqual(len(result['models']), 1)
        self.assertTrue(result['models'][0]['loaded'])

    def test_falls_back_only_when_v1_endpoint_is_missing(self):
        client = LMStudioClient()
        with patch.object(client, '_request', side_effect=[HTTPError('', 404, '', {}, None),
            {'data': [{'type': 'llm', 'id': 'legacy', 'state': 'loaded'}]}]) as request:
            result = client.models()
        self.assertEqual(result['models'][0]['id'], 'legacy')
        self.assertEqual(request.call_args_list[1].args[0], '/api/v0/models')
        with patch.object(client, '_request', side_effect=HTTPError('', 401, 'Unauthorized', {}, None)) as request:
            self.assertFalse(client.models()['available'])
            self.assertEqual(request.call_count, 1)

    def test_offline_server_returns_actionable_state(self):
        with patch.object(LMStudioClient, '_request', side_effect=URLError('connection refused')):
            result = LMStudioClient().models()
            self.assertFalse(result['available'])
            self.assertIn('connection refused', result['error'])

    def test_completions_use_engine_stats_and_reject_missing_or_invalid_rates(self):
        client = LMStudioClient()
        with patch.object(client, '_request', return_value={'stats': {'tokens_per_second': 100}}) as request:
            client.complete_raw('test', 'raw prompt', 8)
            self.assertEqual(request.call_args.args[1]['prompt'], 'raw prompt')
            self.assertEqual(request.call_args.args[1]['temperature'], 0)
        for speed in [None, 0, -1, float('inf'), float('nan'), True]:
            with patch.object(client, '_request', return_value={'stats': {'tokens_per_second': speed}}), self.assertRaises(ValueError):
                client.complete_raw('test', 'raw prompt', 8)

    def test_token_is_not_in_request_url(self):
        client = LMStudioClient(token='test-secret')
        with patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps({'models': []}).encode())) as opener:
            client.models()
        request = opener.call_args.args[0]
        self.assertNotIn('test-secret', request.full_url)
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-secret')

class LMStudioStreamingTests(unittest.TestCase):
    def test_stream_forwards_content_and_usage_without_exposing_token(self):
        client = LMStudioClient(token='test-secret')
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"usage":{"completion_tokens":2},"choices":[]}\n\ndata: [DONE]\n\n')
        with patch.object(client.opener, 'open', return_value=stream) as opener:
            chunks = list(client.iter_chat({'model': 'ling', 'messages': []}))
        self.assertEqual(chunks[0]['choices'][0]['delta']['content'], 'Hi')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 2)
        request = opener.call_args.args[0]
        self.assertTrue(json.loads(request.data)['stream_options']['include_usage'])
        self.assertTrue(stream.closed)

    def test_truncated_or_failed_stream_does_not_report_success(self):
        for raw in [b'data: {"choices":[]}\n\n', b'data: {"error":{"message":"oops"}}\n\n']:
            client = LMStudioClient()
            with patch.object(client.opener, 'open', return_value=io.BytesIO(raw)), self.assertRaises(ValueError):
                list(client.iter_chat({'model': 'qwen'}))
