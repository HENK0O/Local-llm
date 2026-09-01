from __future__ import annotations

import codecs
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from .chat import ChatMessage, format_chat, require_chat_template
from .generation import GenerationStats, generate_tokens
from .loading import load_runtime


MAX_REQUEST_BYTES = 2 * 1024 * 1024
WEB_INDEX = Path(__file__).with_name("web") / "index.html"


@dataclass(frozen=True)
class ChatRequest:
    messages: List[ChatMessage]
    max_tokens: int
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    seed: Optional[int]
    stream: bool
    template_name: Optional[str]


@dataclass(frozen=True)
class StreamPiece:
    text: str
    token_id: Optional[int]
    stats: Optional[GenerationStats]


@dataclass(frozen=True)
class CompletionResult:
    text: str
    token_ids: List[int]
    prompt_tokens: int
    stats: GenerationStats
    finish_reason: str


def parse_chat_request(payload: object, default_max_tokens: int = 128) -> ChatRequest:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")
    messages: List[ChatMessage] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{index}] must be an object")
        try:
            messages.append(ChatMessage(item.get("role"), item.get("content")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"messages[{index}]: {exc}") from exc

    def integer(name: str, default: Optional[int]) -> Optional[int]:
        value = payload.get(name, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    def number(name: str, default: Optional[float]) -> Optional[float]:
        value = payload.get(name, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)

    max_tokens = integer("max_tokens", default_max_tokens)
    top_k = integer("top_k", None)
    seed = integer("seed", None)
    temperature = number("temperature", 0.0)
    top_p = number("top_p", None)
    stream = payload.get("stream", False)
    template_name = payload.get("chat_template")
    if max_tokens is None or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if temperature is None or temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_p is not None and not math.isfinite(top_p):
        raise ValueError("top_p must be finite")
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    if template_name is not None and not isinstance(template_name, str):
        raise ValueError("chat_template must be a string")
    return ChatRequest(messages, max_tokens, temperature, top_k, top_p, seed,
                       stream, template_name)


class ChatService:
    def __init__(self, model_path: Path, default_max_tokens: int = 128) -> None:
        self.model_path = Path(model_path)
        self.model, self.tokenizer = load_runtime(self.model_path)
        require_chat_template(self.tokenizer)
        self.model_name = self.model_path.stem
        self.default_max_tokens = default_max_tokens
        self._generation_lock = threading.Lock()

    def parse(self, payload: object) -> ChatRequest:
        request = parse_chat_request(payload, self.default_max_tokens)
        require_chat_template(self.tokenizer, request.template_name)
        return request

    def _prompt_tokens(self, request: ChatRequest) -> List[int]:
        prompt = format_chat(request.messages, self.tokenizer,
                             template_name=request.template_name)
        return self.tokenizer.encode(prompt)

    def iter_completion(self, request: ChatRequest) -> Iterator[StreamPiece]:
        prompt_tokens = self._prompt_tokens(request)
        self._generation_lock.acquire()
        try:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            for token, stats in generate_tokens(
                self.model, prompt_tokens, request.max_tokens, request.temperature,
                request.top_k, request.top_p, request.seed,
            ):
                data = self.tokenizer.token_bytes(token)
                yield StreamPiece(decoder.decode(data, final=False) if data else "", token, stats)
            tail = decoder.decode(b"", final=True)
            if tail:
                yield StreamPiece(tail, None, None)
        finally:
            self._generation_lock.release()

    def complete(self, request: ChatRequest) -> CompletionResult:
        text_parts: List[str] = []
        token_ids: List[int] = []
        stats: Optional[GenerationStats] = None
        for piece in self.iter_completion(request):
            text_parts.append(piece.text)
            if piece.token_id is not None:
                token_ids.append(piece.token_id)
            stats = piece.stats or stats
        if stats is None:
            raise RuntimeError("generation completed without statistics")
        prompt_tokens = stats.prompt_tokens
        stopped = (self.model.config.eos_token_id is not None and token_ids
                   and token_ids[-1] == self.model.config.eos_token_id)
        return CompletionResult("".join(text_parts), token_ids, prompt_tokens, stats,
                                "stop" if stopped else "length")


class LocalLLMHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], service: ChatService) -> None:
        self.service = service
        super().__init__(address, LocalLLMRequestHandler)


class LocalLLMRequestHandler(BaseHTTPRequestHandler):
    server: LocalLLMHTTPServer
    protocol_version = "HTTP/1.0"
    server_version = "local-llm/0.9"

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                         "connect-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def _path(self) -> str:
        return urlsplit(self.path).path

    def do_GET(self) -> None:
        path = self._path()
        if path in {"/", "/index.html", "/chat"}:
            try:
                self._send_bytes(200, WEB_INDEX.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self._error(500, f"web interface unavailable: {exc}")
            return
        if path == "/favicon.ico":
            self._send_bytes(204, b"", "image/x-icon")
            return
        if path == "/health":
            self._send_json(200, {"status": "ok", "model": self.server.service.model_name})
            return
        if path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{
                "id": self.server.service.model_name,
                "object": "model",
                "owned_by": "local",
            }]})
            return
        self._error(404, "route not found")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _read_payload(self) -> object:
        value = self.headers.get("Content-Length")
        if value is None:
            raise ValueError("Content-Length header is required")
        try:
            length = int(value)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body must contain 1 to {MAX_REQUEST_BYTES} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc

    @staticmethod
    def _response_id() -> str:
        return "chatcmpl-local-" + uuid.uuid4().hex

    def do_POST(self) -> None:
        if self._path() != "/v1/chat/completions":
            self._error(404, "route not found")
            return
        try:
            request = self.server.service.parse(self._read_payload())
            if request.stream:
                self._stream_completion(request)
            else:
                self._complete(request)
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"generation failed: {exc}")

    def _complete(self, request: ChatRequest) -> None:
        result = self.server.service.complete(request)
        stats = result.stats
        self._send_json(200, {
            "id": self._response_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.service.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": len(result.token_ids),
                "total_tokens": result.prompt_tokens + len(result.token_ids),
            },
            "local_llm": {
                "prefill_tokens_per_second": stats.prefill_tokens_per_second,
                "decode_tokens_per_second": stats.decode_tokens_per_second,
                "kv_cache_bytes": stats.cache_bytes,
            },
        })

    def _write_event(self, payload: object) -> None:
        value = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {value}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_completion(self, request: ChatRequest) -> None:
        response_id = self._response_id()
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._write_event({
            "id": response_id, "object": "chat.completion.chunk", "created": created,
            "model": self.server.service.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
        token_ids: List[int] = []
        final_stats: Optional[GenerationStats] = None
        try:
            for piece in self.server.service.iter_completion(request):
                if piece.token_id is not None:
                    token_ids.append(piece.token_id)
                final_stats = piece.stats or final_stats
                if piece.text:
                    self._write_event({
                        "id": response_id, "object": "chat.completion.chunk", "created": created,
                        "model": self.server.service.model_name,
                        "choices": [{"index": 0, "delta": {"content": piece.text},
                                     "finish_reason": None}],
                    })
            stopped = (self.server.service.model.config.eos_token_id is not None and token_ids
                       and token_ids[-1] == self.server.service.model.config.eos_token_id)
            final = {
                "id": response_id, "object": "chat.completion.chunk", "created": created,
                "model": self.server.service.model_name,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop" if stopped else "length"}],
            }
            if final_stats is not None:
                final["usage"] = {
                    "prompt_tokens": final_stats.prompt_tokens,
                    "completion_tokens": len(token_ids),
                    "total_tokens": final_stats.prompt_tokens + len(token_ids),
                }
                final["local_llm"] = {
                    "prefill_tokens_per_second": final_stats.prefill_tokens_per_second,
                    "decode_tokens_per_second": final_stats.decode_tokens_per_second,
                    "kv_cache_bytes": final_stats.cache_bytes,
                }
            self._write_event(final)
            self._write_event("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            # HTTP headers have already been sent, so report generation errors as
            # an SSE event rather than attempting a second HTTP response.
            try:
                self._write_event({"error": {
                    "message": f"generation failed: {exc}",
                    "type": "server_error",
                }})
                self._write_event("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                pass


def create_server(model_path: Path, host: str = "127.0.0.1", port: int = 8080,
                  default_max_tokens: int = 128) -> LocalLLMHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return LocalLLMHTTPServer((host, port), ChatService(model_path, default_max_tokens))


def serve(model_path: Path, host: str = "127.0.0.1", port: int = 8080,
          default_max_tokens: int = 128) -> None:
    server = create_server(model_path, host, port, default_max_tokens)
    address, actual_port = server.server_address[:2]
    print(f"local-llm server listening on http://{address}:{actual_port}")
    print(f"model: {server.service.model_name} | POST /v1/chat/completions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()
