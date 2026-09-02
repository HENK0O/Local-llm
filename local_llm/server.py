from __future__ import annotations

import codecs
import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from .chat import ChatMessage, format_chat, require_chat_template
from .evaluation import capture_external_reference, capture_trace, compare_traces
from .generation import GenerationStats, generate_tokens
from .loading import load_runtime
from .version import __version__


MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_REQUEST_TOKENS = 512
DEFAULT_MAX_CONNECTIONS = 8
REQUEST_TIMEOUT_SECONDS = 30
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


@dataclass(frozen=True)
class BenchmarkRequest:
    prompt: str
    tokens: int


def parse_chat_request(payload: object, default_max_tokens: int = 128,
                       max_request_tokens: int = MAX_REQUEST_TOKENS) -> ChatRequest:
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
    if max_tokens is None or not 1 <= max_tokens <= max_request_tokens:
        raise ValueError(f"max_tokens must be between 1 and {max_request_tokens}")
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


def parse_benchmark_request(payload: object) -> BenchmarkRequest:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    prompt = payload.get("prompt")
    tokens = payload.get("tokens", 8)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 32:
        raise ValueError("tokens must be an integer between 1 and 32")
    return BenchmarkRequest(prompt.strip(), tokens)


class ChatService:
    def __init__(self, model_path: Path, default_max_tokens: int = 128,
                 reference: Optional[Path] = None,
                 reference_repo: Optional[Path] = None,
                 max_request_tokens: int = MAX_REQUEST_TOKENS) -> None:
        self.model_path = Path(model_path)
        self.model, self.tokenizer = load_runtime(self.model_path)
        require_chat_template(self.tokenizer)
        self.model_name = self.model_path.stem
        self.default_max_tokens = default_max_tokens
        self.max_request_tokens = max_request_tokens
        if not 1 <= self.default_max_tokens <= self.max_request_tokens:
            raise ValueError("default max tokens must not exceed the request token limit")
        self.reference = Path(reference) if reference is not None else None
        self.reference_repo = Path(reference_repo) if reference_repo is not None else None
        if self.reference is not None and not self.reference.is_file():
            raise FileNotFoundError(f"reference not found: {self.reference}")
        if self.reference_repo is not None and not self.reference_repo.is_dir():
            raise FileNotFoundError(f"reference repository not found: {self.reference_repo}")
        self._generation_lock = threading.Lock()

    @property
    def reference_name(self) -> str:
        return self.reference.name if self.reference is not None else "Recalcul complet"

    def parse(self, payload: object) -> ChatRequest:
        request = parse_chat_request(
            payload, self.default_max_tokens, self.max_request_tokens
        )
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

    @staticmethod
    def _trace_speed(trace, prompt_tokens: int) -> Dict[str, float]:
        decoded = max(0, len(trace.generated_token_ids) - 1)
        return {
            "prefill_tokens_per_second": (
                prompt_tokens / trace.prefill_seconds if trace.prefill_seconds else 0.0
            ),
            "decode_tokens_per_second": (
                decoded / trace.decode_seconds if trace.decode_seconds else 0.0
            ),
        }

    def benchmark(self, payload: object) -> Dict:
        request = parse_benchmark_request(payload)
        prompt = format_chat([
            ChatMessage("system", "Tu es un assistant utile. Réponds en français, sauf si "
                        "l’utilisateur demande une autre langue."),
            ChatMessage("user", request.prompt),
        ], self.tokenizer)
        prompt_ids = self.tokenizer.encode(prompt)
        with self._generation_lock:
            optimized = capture_trace(self.model, prompt_ids, request.tokens, use_cache=True)
            if self.reference is None:
                reference = capture_trace(self.model, prompt_ids, request.tokens, use_cache=False)
                reference_kind = "full_recompute"
                reference_label = "Même moteur · recalcul complet"
            else:
                reference, reference_label = capture_external_reference(
                    self.model_path, self.reference, self.reference_repo,
                    prompt_ids, request.tokens,
                )
                reference_kind = "external"

        comparison = compare_traces(optimized, reference, 2e-4, 2e-4)
        optimized_ids = optimized.generated_token_ids.tolist()
        reference_ids = reference.generated_token_ids.tolist()
        optimized_speed = self._trace_speed(optimized, len(prompt_ids))
        reference_speed = self._trace_speed(reference, len(prompt_ids))
        return {
            "model": self.model_name,
            "reference": {"kind": reference_kind, "name": reference_label},
            "prompt_tokens": len(prompt_ids),
            "requested_tokens": request.tokens,
            "optimized": {
                "name": "local-llm optimisé",
                "text": self.tokenizer.decode(optimized_ids),
                "token_ids": optimized_ids,
                "kv_cache_bytes": optimized.kv_cache_bytes,
                **optimized_speed,
            },
            "baseline": {
                "name": reference_label,
                "text": self.tokenizer.decode(reference_ids),
                "token_ids": reference_ids,
                "kv_cache_bytes": reference.kv_cache_bytes,
                **reference_speed,
            },
            "comparison": {
                "tokens_identical": comparison.greedy_tokens_identical,
                "within_tolerance": comparison.within_tolerance,
                "max_absolute_error": comparison.max_absolute_error,
                "mean_absolute_error": comparison.mean_absolute_error,
            },
            "passed": comparison.greedy_tokens_identical and comparison.within_tolerance,
        }


class LocalLLMHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], service: ChatService,
                 max_connections: int = DEFAULT_MAX_CONNECTIONS) -> None:
        if not 1 <= max_connections <= 128:
            raise ValueError("max connections must be between 1 and 128")
        self.service = service
        self.max_connections = max_connections
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, LocalLLMRequestHandler)

    def process_request(self, request: socket.socket, client_address) -> None:
        if not self._connection_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connection_slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class LocalLLMRequestHandler(BaseHTTPRequestHandler):
    server: LocalLLMHTTPServer
    protocol_version = "HTTP/1.0"
    server_version = f"local-llm/{__version__}"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

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
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
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
            self._send_json(200, {
                "status": "ok",
                "model": self.server.service.model_name,
                "benchmark_reference": self.server.service.reference_name,
                "external_reference": self.server.service.reference is not None,
            })
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
        # Cross-origin browser access is intentionally disabled. The bundled UI
        # is same-origin and therefore does not require CORS preflights.
        self.send_response(405)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # CLI clients such as curl do not send Origin.
        parsed = urlsplit(origin)
        host = self.headers.get("Host", "")
        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc == host
            and not parsed.path.strip("/")
            and not parsed.query
            and not parsed.fragment
        )

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
        path = self._path()
        if path not in {"/v1/chat/completions", "/v1/benchmark"}:
            self._error(404, "route not found")
            return
        if not self._same_origin():
            self._error(403, "cross-origin requests are disabled")
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._error(415, "Content-Type must be application/json")
            return
        try:
            payload = self._read_payload()
            if path == "/v1/benchmark":
                self._send_json(200, self.server.service.benchmark(payload))
                return
            request = self.server.service.parse(payload)
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
        self.send_header("X-Content-Type-Options", "nosniff")
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
                  default_max_tokens: int = 128, reference: Optional[Path] = None,
                  reference_repo: Optional[Path] = None,
                  max_request_tokens: int = MAX_REQUEST_TOKENS,
                  max_connections: int = DEFAULT_MAX_CONNECTIONS,
                  allow_remote: bool = False) -> LocalLLMHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise ValueError("remote binding requires --allow-remote")
    service = ChatService(model_path, default_max_tokens, reference, reference_repo,
                          max_request_tokens)
    return LocalLLMHTTPServer((host, port), service, max_connections)


def serve(model_path: Path, host: str = "127.0.0.1", port: int = 8080,
          default_max_tokens: int = 128, reference: Optional[Path] = None,
          reference_repo: Optional[Path] = None,
          max_request_tokens: int = MAX_REQUEST_TOKENS,
          max_connections: int = DEFAULT_MAX_CONNECTIONS,
          allow_remote: bool = False) -> None:
    server = create_server(model_path, host, port, default_max_tokens,
                           reference, reference_repo, max_request_tokens,
                           max_connections, allow_remote)
    address, actual_port = server.server_address[:2]
    print(f"local-llm server listening on http://{address}:{actual_port}")
    print(f"model: {server.service.model_name} | POST /v1/chat/completions")
    print(f"benchmark: {server.service.reference_name} | POST /v1/benchmark")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()
