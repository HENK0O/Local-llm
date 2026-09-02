import http.client
import json
import threading
import unittest
from types import SimpleNamespace

from local_llm.cli import build_parser
from local_llm.generation import GenerationStats
from local_llm.server import (
    CompletionResult,
    LocalLLMHTTPServer,
    StreamPiece,
    parse_chat_request,
)


class StubChatService:
    model_name = "test-model"
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))

    @staticmethod
    def parse(payload):
        return parse_chat_request(payload, default_max_tokens=16)

    @staticmethod
    def complete(request):
        stats = GenerationStats(3, 2, 0.1, 0.1, 128)
        return CompletionResult("Bonjour", [7, 2], 3, stats, "stop")

    @staticmethod
    def iter_completion(request):
        yield StreamPiece("Bon", 7, None)
        yield StreamPiece("jour", 2, GenerationStats(3, 2, 0.1, 0.1, 128))


class ChatRequestTests(unittest.TestCase):
    def test_parses_sampling_options(self):
        request = parse_chat_request({
            "messages": [{"role": "user", "content": "Bonjour"}],
            "max_tokens": 12,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.9,
            "seed": 42,
            "stream": True,
        })
        self.assertEqual(request.messages[0].content, "Bonjour")
        self.assertEqual(request.max_tokens, 12)
        self.assertTrue(request.stream)

    def test_rejects_invalid_requests(self):
        invalid = [
            {},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
            {"messages": [{"role": "user", "content": "x"}], "stream": "yes"},
            {"messages": [{"role": "user", "content": "x"}], "top_p": 2},
            {"messages": [{"role": "user", "content": "x"}], "seed": -1},
            {"messages": [{"role": "user", "content": "x"}], "temperature": float("nan")},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_chat_request(payload)

    def test_cli_exposes_serve_command(self):
        args = build_parser().parse_args(["serve", "model.gguf", "--port", "9000"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.port, 9000)

    def test_cli_exposes_profile_command(self):
        args = build_parser().parse_args([
            "profile", "model.gguf", "--tokens", "4", "--json",
        ])
        self.assertEqual(args.command, "profile")
        self.assertEqual(args.tokens, 4)
        self.assertTrue(args.json)


class HTTPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = LocalLLMHTTPServer(("127.0.0.1", 0), StubChatService())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), data

    def test_health_and_models(self):
        status, content_type, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"local-llm run --local", body)
        self.assertIn(b'id="modelName"', body)
        self.assertIn(b"/v1/chat/completions", body)
        status, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model"], "test-model")
        status, _, body = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "test-model")

    def test_non_streaming_completion(self):
        status, content_type, body = self.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Salut"}],
        })
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        result = json.loads(body)
        self.assertEqual(result["choices"][0]["message"]["content"], "Bonjour")
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.assertEqual(result["local_llm"]["kv_cache_bytes"], 128)

    def test_streaming_completion(self):
        status, content_type, body = self.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Salut"}],
            "stream": True,
        })
        text = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", content_type)
        self.assertIn('"content": "Bon"', text)
        self.assertIn('"content": "jour"', text)
        self.assertIn('"decode_tokens_per_second"', text)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_invalid_request_and_unknown_route(self):
        status, _, body = self.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "tool", "content": "x"}],
        })
        self.assertEqual(status, 400)
        self.assertIn("unsupported chat role", json.loads(body)["error"]["message"])
        status, _, _ = self.request("GET", "/missing")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
