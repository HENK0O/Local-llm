from __future__ import annotations

import codecs
import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass, replace
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from .chat import ChatMessage, format_chat, require_chat_template
from .evaluation import capture_external_reference, capture_trace, compare_traces
from .generation import GenerationStats, generate_tokens
from .comparison import compare_cached
from .discovery import default_model_roots, discover_models, inspect_model
from .lmstudio import LMStudioClient
from .recommendations import recommend_models
from .telemetry import SystemTelemetry
from .cache import PrefixCache
from .gguf import Q4Matrix, Q8Matrix, q4_backend_name, q8_backend_name
from .loading import load_runtime
from .reference_runtime import ReferenceRuntime
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
    backend: str
    model_id: Optional[str] = None


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
    backend = payload.get("backend", "local")
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
    if not isinstance(backend, str) or backend not in {"local", "reference", "lmstudio"}:
        raise ValueError("backend must be local, reference or lmstudio")
    return ChatRequest(messages, max_tokens, temperature, top_k, top_p, seed,
                       stream, template_name, backend)


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
    def __init__(self, model_path: Optional[Path] = None, default_max_tokens: int = 128,
                 reference: Optional[Path] = None,
                 reference_repo: Optional[Path] = None,
                 max_request_tokens: int = MAX_REQUEST_TOKENS,
                 model_dirs=None, lm_studio_url: str = "http://127.0.0.1:1234") -> None:
        self.telemetry = SystemTelemetry()
        self._generation_lock = threading.RLock()
        self._records = OrderedDict()
        self.prefix_cache = PrefixCache()
        self.extra_model_roots = list(model_dirs or [])
        self.model_roots = default_model_roots() + self.extra_model_roots
        self.catalog = {m.id: m for m in discover_models(self.model_roots)}
        if model_path is not None:
            explicit = inspect_model(Path(model_path))
            self.catalog[explicit.id] = explicit
        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.model_name = "Aucun modèle chargé"
        self.current_id = None
        self.lmstudio = LMStudioClient(lm_studio_url)
        self.default_max_tokens = default_max_tokens
        self.max_request_tokens = max_request_tokens
        if not 1 <= self.default_max_tokens <= self.max_request_tokens:
            raise ValueError("default max tokens must not exceed the request token limit")
        self.reference = Path(reference) if reference is not None else None
        self.reference_repo = Path(reference_repo) if reference_repo is not None else None
        if self.reference is not None and not self.reference.is_file():
            if not self.reference.is_dir():
                raise FileNotFoundError(f"reference not found: {self.reference}")
        if self.reference_repo is not None and not self.reference_repo.is_dir():
            raise FileNotFoundError(f"reference repository not found: {self.reference_repo}")
        self.reference_runtime = None
        if model_path is not None:
            self.load_model({"id": explicit.id})
        else:
            compatible = next((m for m in self.catalog.values() if m.compatible), None)
            if compatible is not None:
                self.load_model({"id": compatible.id})

    def available_models(self, refresh: bool = False) -> Dict:
        with self._generation_lock:
            if refresh:
                self.model_roots = default_model_roots() + self.extra_model_roots
                items = {m.id: m for m in discover_models(self.model_roots)}
                if self.current_id and self.current_id in self.catalog:
                    items[self.current_id] = self.catalog[self.current_id]
                self.catalog = items
            return {"models": [m.to_dict() for m in self.catalog.values()],
                    "current_id": self.current_id,
                    "roots": [str(Path(p).expanduser()) for p in self.model_roots]}

    def load_model(self, payload: object) -> Dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
            raise ValueError("model id is required")
        with self._generation_lock:
            item = self.catalog.get(payload["id"])
            if item is None:
                raise ValueError("Modèle absent des bibliothèques configurées")
            if not item.compatible:
                raise ValueError(item.reason)
            if item.id != self.current_id:
                model, tokenizer = load_runtime(Path(item.path))
                require_chat_template(tokenizer)
                if tokenizer.vocab_size != model.config.vocab_size:
                    raise ValueError("Tokenizer vocabulary does not match model")
                self.model, self.tokenizer = model, tokenizer
                self.model_path = Path(item.path)
                self.model_name = item.name
                self.current_id = item.id
                self._records.clear()
                self.prefix_cache.clear()
                self.reference_runtime = (
                    ReferenceRuntime(self.reference, self.reference_repo, model.config)
                    if self.reference is not None and (self.reference.is_dir() or
                       self.reference.suffix.lower() in {".pt", ".pth"}) else None
                )
            return self.info()

    def info(self) -> Dict:
        backend = "numpy-blas"
        if self.model is not None:
            if any(isinstance(w, Q8Matrix) for w in self.model.weights.values()):
                backend = "q8-" + q8_backend_name()
            elif any(isinstance(w, Q4Matrix) for w in self.model.weights.values()):
                backend = "q4-" + q4_backend_name()
        return {"status": "ok", "model": self.model_name, "current_id": self.current_id,
                "loaded": self.model is not None, "engine": backend,
                "context_length": self.model.config.max_position_embeddings if self.model else None,
                "benchmark_reference": self.reference_name,
                "external_reference": self.reference is not None,
                "reference_chat_available": self.reference_runtime is not None,
                "max_request_tokens": self.max_request_tokens,
                "features": ["system_telemetry", "prefix_cache", "model_unload"],
                "retained_cache_bytes": self.prefix_cache.nbytes,
                "retained_cache_limit_bytes": self.prefix_cache.max_bytes}

    def unload_model(self) -> Dict:
        with self._generation_lock:
            self.prefix_cache.clear()
            self._records.clear()
            self.model = self.tokenizer = self.reference_runtime = None
            self.model_path = self.current_id = None
            self.model_name = "Aucun modèle chargé"
            return self.info()

    def compare_completion(self, payload: object) -> Dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("completion_id"), str):
            raise ValueError("completion_id is required")
        with self._generation_lock:
            record = self._records.get(payload["completion_id"])
            if record is None or time.monotonic() - record["created"] > 900:
                raise ValueError("Réponse expirée ou modèle changé ; génère une nouvelle réponse")
            kind = payload.get("baseline", "numpy")
            if kind == "numpy":
                return compare_cached(self.model, record["prompt_ids"], record["token_ids"])
            if kind == "lmstudio":
                model_id = payload.get("lmstudio_model")
                if not isinstance(model_id, str):
                    raise ValueError("lmstudio_model must be a string")
                available = self.lmstudio.models()
                if not available["available"]:
                    raise ValueError("LM Studio indisponible : " + available["error"])
                if model_id not in {m["id"] for m in available["models"]}:
                    raise ValueError("Choisis un modèle disponible dans LM Studio")
                reference = self.lmstudio.complete_raw(model_id, record["prompt"], len(record["token_ids"]))
                local = record["stats"].decode_tokens_per_second
                baseline = reference["stats"]["tokens_per_second"]
                choices = reference.get("choices", [])
                text = choices[0].get("text", "") if choices else ""
                local_text = self.tokenizer.decode(record["token_ids"])
                return {"kind": "lmstudio", "reference_name": "LM Studio · " + model_id,
                        "local_tokens_per_second": local, "baseline_tokens_per_second": baseline,
                        "delta_tokens_per_second": local - baseline if local > 0 else None,
                        "speedup": local / baseline if local > 0 else None,
                        "comparable": False, "same_weights": None,
                        "scope": "cross_engine_observation", "validated_engine_gain": False,
                        "reference_is_standard_engine": True,
                        "text_identical": text == local_text,
                        "baseline_text": text, "runtime": reference.get("runtime"),
                        "baseline_usage": reference.get("usage"),
                        "method": "Même prompt rendu et température 0. Poids, tokenizer, offload CPU/GPU et définition du débit non vérifiés : écart indicatif, pas un gain validé."}
            raise ValueError("baseline must be 'numpy' or 'lmstudio'")

    @property
    def reference_name(self) -> str:
        if self.reference_runtime is not None:
            return self.reference_runtime.name
        return self.reference.name if self.reference is not None else "Recalcul complet"

    def parse(self, payload: object) -> ChatRequest:
        request = parse_chat_request(payload, self.default_max_tokens, self.max_request_tokens)
        if request.backend == "lmstudio":
            model_id = payload.get("model")
            listing = self.lmstudio.models()
            if not listing["available"]:
                raise ValueError("Activez le serveur local dans LM Studio, puis actualisez la bibliothèque")
            if not isinstance(model_id, str) or model_id not in {m["id"] for m in listing["models"]}:
                raise ValueError("Modèle absent du serveur LM Studio")
            if not request.stream:
                raise ValueError("LM Studio nécessite stream=true")
            return replace(request, model_id=model_id)
        if self.model is None:
            raise ValueError("Charge un modèle avant de démarrer une conversation")
        model_id = payload.get("model") if isinstance(payload, dict) else None
        if model_id is not None and not isinstance(model_id, str):
            raise ValueError("model must be a string")
        if model_id not in {None, self.current_id, self.model_name}:
            raise ValueError("Le modèle demandé n’est pas chargé")
        request = replace(request, model_id=self.current_id)
        require_chat_template(self.tokenizer, request.template_name)
        if request.backend == "reference" and self.reference_runtime is None:
            raise ValueError(
                "reference backend is unavailable; start serve with a Baguette .pt "
                "checkpoint or a Hugging Face model directory"
            )
        return request

    def iter_completion(self, request: ChatRequest, completion_id: Optional[str] = None) -> Iterator[StreamPiece]:
        self._generation_lock.acquire()
        generator = None
        try:
            if request.model_id is not None and request.model_id != self.current_id:
                raise ValueError("Le modèle a changé ; renvoie la requête")
            prompt = format_chat(request.messages, self.tokenizer, template_name=request.template_name)
            prompt_tokens = self.tokenizer.encode(prompt)
            emitted = []
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            generator = (
                self.reference_runtime.generate_tokens(
                    prompt_tokens, request.max_tokens, request.temperature,
                    request.top_k, request.top_p, request.seed,
                )
                if request.backend == "reference" and self.reference_runtime is not None
                else generate_tokens(
                    self.model, prompt_tokens, request.max_tokens, request.temperature,
                    request.top_k, request.top_p, request.seed,
                    prefix_cache=self.prefix_cache if request.backend == "local" else None,
                )
            )
            for token, stats in generator:
                emitted.append(token)
                if stats is not None and completion_id is not None and request.backend == "local":
                    self._records[completion_id] = {"prompt": prompt, "prompt_ids": prompt_tokens,
                        "token_ids": list(emitted), "stats": stats, "created": time.monotonic()}
                    while len(self._records) > 8:
                        self._records.popitem(last=False)
                data = self.tokenizer.token_bytes(token)
                yield StreamPiece(decoder.decode(data, final=False) if data else "", token, stats)
            tail = decoder.decode(b"", final=True)
            if tail:
                yield StreamPiece(tail, None, None)
        finally:
            if generator is not None:
                generator.close()
            self._generation_lock.release()

    def complete(self, request: ChatRequest, completion_id: Optional[str] = None) -> CompletionResult:
        text_parts: List[str] = []
        token_ids: List[int] = []
        stats: Optional[GenerationStats] = None
        for piece in self.iter_completion(request, completion_id):
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
            if self.reference is None or self.reference.is_dir():
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

    def server_close(self) -> None:
        super().server_close()
        telemetry = getattr(self.service, "telemetry", None)
        if telemetry is not None:
            telemetry.close()

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
        if path == "/v1/local-models":
            self._send_json(200, self.server.service.available_models())
            return
        if path == "/v1/system":
            self._send_json(200, self.server.service.telemetry.snapshot())
            return
        if path == "/v1/recommendations":
            self._send_json(200, recommend_models(installed=self.server.service.catalog.values()))
            return
        if path == "/v1/lmstudio/models":
            self._send_json(200, self.server.service.lmstudio.models())
            return
        if path == "/health":
            if hasattr(self.server.service, "info"):
                self._send_json(200, self.server.service.info())
                return
            self._send_json(200, {
                "status": "ok",
                "model": self.server.service.model_name,
                "benchmark_reference": self.server.service.reference_name,
                "external_reference": self.server.service.reference is not None,
                "reference_chat_available": self.server.service.reference_runtime is not None,
            })
            return
        if path == "/v1/models":
            if getattr(self.server.service, "model", True) is None:
                self._send_json(200, {"object": "list", "data": []})
                return
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
        if path not in {"/v1/chat/completions", "/v1/benchmark", "/v1/local-models/load", "/v1/local-models/unload", "/v1/local-models/refresh", "/v1/compare"}:
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
            if path == "/v1/local-models/unload":
                self._send_json(200, self.server.service.unload_model())
                return
            if path == "/v1/local-models/load":
                self._send_json(200, self.server.service.load_model(payload))
                return
            if path == "/v1/local-models/refresh":
                self._send_json(200, self.server.service.available_models(refresh=True))
                return
            if path == "/v1/compare":
                self._send_json(200, self.server.service.compare_completion(payload))
                return
            if path == "/v1/benchmark":
                self._send_json(200, self.server.service.benchmark(payload))
                return
            request = self.server.service.parse(payload)
            if request.backend == "lmstudio":
                self._stream_lmstudio(request)
            elif request.stream:
                self._stream_completion(request)
            else:
                self._complete(request)
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"generation failed: {exc}")

    def _complete(self, request: ChatRequest) -> None:
        response_id = self._response_id()
        result = self.server.service.complete(request, response_id)
        stats = result.stats
        self._send_json(200, {
            "id": response_id,
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
                "backend": request.backend,
                "completion_id": response_id,
                "prefill_seconds": stats.prefill_seconds,
                "decode_seconds": stats.decode_seconds,
                "prefill_tokens_per_second": stats.prefill_tokens_per_second,
                "decode_tokens_per_second": stats.decode_tokens_per_second,
                "kv_cache_bytes": stats.cache_bytes,
                "reused_prompt_tokens": stats.reused_prompt_tokens,
            },
        })

    def _write_event(self, payload: object) -> None:
        value = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {value}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_lmstudio(self, request: ChatRequest) -> None:
        response_id = self._response_id()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        payload = {"model": request.model_id,
                   "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                   "max_tokens": request.max_tokens, "temperature": request.temperature}
        for key in ("top_p", "seed"):
            if getattr(request, key) is not None:
                payload[key] = getattr(request, key)
        started = time.perf_counter()
        first = None
        usage = {}
        finish = "stop"
        upstream = self.server.service.lmstudio.iter_chat(payload)
        try:
            for chunk in upstream:
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    if choice.get("delta", {}).get("content"):
                        if first is None:
                            first = time.perf_counter()
                chunk.update(id=response_id, model=request.model_id, backend="lmstudio")
                self._write_event(chunk)
            seconds = time.perf_counter() - started
            count = usage.get("completion_tokens")
            count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
            self._write_event({"id": response_id, "model": request.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage, "local_llm": {"backend": "lmstudio", "completion_id": response_id,
                "decode_tokens_per_second": count / seconds if count is not None and seconds > 0 else None,
                "prefill_seconds": first - started if first is not None else None,
                "decode_seconds": seconds, "kv_cache_bytes": None,
                "reused_prompt_tokens": None,
                "timing_kind": "observed_total", "optimized": False}})
            self._write_event("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self._write_event({"error": {"message": str(exc)}})
                self._write_event("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            upstream.close()

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
            "backend": request.backend,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
        token_ids: List[int] = []
        final_stats: Optional[GenerationStats] = None
        try:
            for piece in self.server.service.iter_completion(request, response_id):
                if piece.token_id is not None:
                    token_ids.append(piece.token_id)
                final_stats = piece.stats or final_stats
                if piece.text:
                    self._write_event({
                        "id": response_id, "object": "chat.completion.chunk", "created": created,
                        "model": self.server.service.model_name,
                        "backend": request.backend,
                        "choices": [{"index": 0, "delta": {"content": piece.text},
                                     "finish_reason": None}],
                    })
            stopped = (self.server.service.model.config.eos_token_id is not None and token_ids
                       and token_ids[-1] == self.server.service.model.config.eos_token_id)
            final = {
                "id": response_id, "object": "chat.completion.chunk", "created": created,
                "model": self.server.service.model_name,
                "backend": request.backend,
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
                    "backend": request.backend,
                    "completion_id": response_id,
                    "prefill_seconds": final_stats.prefill_seconds,
                    "decode_seconds": final_stats.decode_seconds,
                    "prefill_tokens_per_second": final_stats.prefill_tokens_per_second,
                    "decode_tokens_per_second": final_stats.decode_tokens_per_second,
                    "kv_cache_bytes": final_stats.cache_bytes,
                    "reused_prompt_tokens": final_stats.reused_prompt_tokens,
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


def create_server(model_path: Optional[Path] = None, host: str = "127.0.0.1", port: int = 8080,
                  default_max_tokens: int = 128, reference: Optional[Path] = None,
                  reference_repo: Optional[Path] = None,
                  max_request_tokens: int = MAX_REQUEST_TOKENS,
                  max_connections: int = DEFAULT_MAX_CONNECTIONS,
                  allow_remote: bool = False, model_dirs=None,
                  lm_studio_url: str = "http://127.0.0.1:1234") -> LocalLLMHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise ValueError("remote binding requires --allow-remote")
    service = ChatService(model_path, default_max_tokens, reference, reference_repo,
                          max_request_tokens, model_dirs, lm_studio_url)
    return LocalLLMHTTPServer((host, port), service, max_connections)


def serve(model_path: Optional[Path] = None, host: str = "127.0.0.1", port: int = 8080,
          default_max_tokens: int = 128, reference: Optional[Path] = None,
          reference_repo: Optional[Path] = None,
          max_request_tokens: int = MAX_REQUEST_TOKENS,
          max_connections: int = DEFAULT_MAX_CONNECTIONS,
          allow_remote: bool = False, model_dirs=None,
          lm_studio_url: str = "http://127.0.0.1:1234") -> None:
    server = create_server(model_path, host, port, default_max_tokens,
                           reference, reference_repo, max_request_tokens,
                           max_connections, allow_remote, model_dirs, lm_studio_url)
    address, actual_port = server.server_address[:2]
    print(f"local-llm server listening on http://{address}:{actual_port}")
    print(f"model: {server.service.model_name} | POST /v1/chat/completions")
    print(f"benchmark: {server.service.reference_name} | POST /v1/benchmark")
    if server.service.reference_runtime is not None:
        print(f"reference chat backend: {server.service.reference_runtime.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()
