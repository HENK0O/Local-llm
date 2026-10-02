"""Small local-only LM Studio client; no SDK or network model downloads."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("LM Studio redirects are disabled")


class LMStudioClient:
    def __init__(self, url: str = "http://127.0.0.1:1234", token: Optional[str] = None):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username or parts.password or parts.path.strip("/") or parts.query or parts.fragment):
            raise ValueError("LM Studio URL must be a loopback HTTP address, e.g. http://127.0.0.1:1234")
        self.url = url.rstrip("/")
        self.token = token if token is not None else os.environ.get("LM_STUDIO_API_TOKEN")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def _request(self, route: str, payload=None, timeout=3) -> Dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + route, data=data, headers=headers)
        with self.opener.open(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("LM Studio response is too large")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Invalid LM Studio response")
        return result

    def models(self) -> Dict:
        try:
            try:
                result = self._request("/api/v1/models")
                entries = result.get("models", [])
                models = [{"id": m["key"], "name": m.get("display_name", m["key"]),
                           "architecture": m.get("architecture"),
                           "quantization": (m.get("quantization") or {}).get("name"),
                           "loaded": bool(m.get("loaded_instances"))}
                          for m in entries if m.get("type") == "llm"]
            except HTTPError as exc:
                if exc.code not in {404, 405}:
                    raise
                entries = self._request("/api/v0/models").get("data", [])
                models = [{"id": m["id"], "name": m["id"], "architecture": m.get("arch"),
                           "quantization": m.get("quantization"), "loaded": m.get("state") == "loaded"}
                          for m in entries if m.get("type") == "llm"]
            return {"available": True, "url": self.url, "models": models, "error": None}
        except (OSError, ValueError, KeyError, TypeError, URLError) as exc:
            return {"available": False, "url": self.url, "models": [], "error": str(exc)}

    def iter_chat(self, payload):
        """Relay OpenAI SSE, including usage; close upstream on disconnect."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        body = dict(payload, stream=True, stream_options={"include_usage": True})
        request = Request(self.url + "/v1/chat/completions",
                          data=json.dumps(body).encode(), headers=headers)
        with self.opener.open(request, timeout=180) as response:
            frame = []
            size = 0
            done = False
            while True:
                line = response.readline(1024 * 1024 + 1)
                if not line:
                    break
                size += len(line)
                if size > 8 * 1024 * 1024 or len(line) > 1024 * 1024:
                    raise ValueError("LM Studio stream is too large")
                text = line.decode("utf-8").strip()
                if not text:
                    if frame:
                        raw = "\n".join(frame)
                        frame = []
                        if raw == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(raw)
                        if not isinstance(chunk, dict) or chunk.get("error"):
                            raise ValueError("LM Studio generation failed")
                        yield chunk
                elif text.startswith("data:"):
                    frame.append(text[5:].lstrip())
            if not done:
                raise ValueError("LM Studio stream ended before completion")

    def complete_raw(self, model_id: str, prompt: str, tokens: int) -> Dict:
        # v0 retains raw completions and engine timings; v1 chat would apply
        # another chat template, making a prompt comparison ambiguous.
        result = self._request("/api/v0/completions", {
            "model": model_id, "prompt": prompt, "max_tokens": tokens,
            "temperature": 0, "stream": False,
        }, timeout=180)
        speed = result.get("stats", {}).get("tokens_per_second")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("LM Studio did not return a valid engine token rate")
        return result
