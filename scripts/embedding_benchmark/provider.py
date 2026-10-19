"""Explicit local embedding-only HTTP adapter, independent of the app's .env."""
import json
import math
import time
from urllib.parse import urlsplit
import urllib.request


class EmbeddingProvider:
    def __init__(self, base_url, protocol="ollama", timeout=120, request=None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query or fragment")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("This runner accepts local endpoints only; use an SSH tunnel for a remote server")
        if protocol not in {"ollama", "openai"}:
            raise ValueError("Protocol must be ollama or openai")
        self.url, self.protocol, self.timeout = base_url.rstrip("/"), protocol, timeout
        if protocol == "ollama" and self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self.request = request or self._request

    def _request(self, path, payload=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        # Never send a dataset to an HTTP redirect target or use environment proxies.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def embed(self, model, texts, dimensions=None):
        if not texts:
            return [], {}
        payload = {"model": model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if self.protocol == "ollama":
            payload.update(truncate=False, keep_alive="10m")
        begun = time.perf_counter()
        result = self.request("/api/embed" if self.protocol == "ollama" else "/embeddings", payload)
        elapsed = (time.perf_counter() - begun) * 1000
        if result.get("error"):
            raise ValueError("Embedding server returned an error; no fallback or truncation is allowed")
        if result.get("model") and result["model"] not in {model, model + ":latest"}:
            raise ValueError("Server returned a different model than requested")
        if self.protocol == "ollama":
            vectors = result.get("embeddings")
        else:
            data = result.get("data", [])
            if sorted(item.get("index", -1) for item in data) != list(range(len(texts))):
                raise ValueError("Embedding indices are missing, duplicated or out of range")
            vectors = [item.get("embedding") for item in sorted(data, key=lambda item: item["index"])]
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Embedding count does not match requested inputs")
        for vector in vectors:
            if (not isinstance(vector, list) or not vector or (dimensions is not None and len(vector) != dimensions)
                    or not all(type(v) in {int, float} and math.isfinite(v) for v in vector) or not any(vector)):
                raise ValueError("Invalid, zero, non-finite or dimension-mismatched embedding")
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Inconsistent embedding dimensions in one batch")
        return vectors, {"latency_ms": elapsed, "inputs": len(texts),
                         "prompt_tokens": result.get("prompt_eval_count", (result.get("usage") or {}).get("prompt_tokens")),
                         **{key: result[key] for key in ("total_duration", "load_duration") if key in result}}

    def metadata(self, model):
        if self.protocol != "ollama":
            return {"status": "server_metadata_not_available_for_openai_protocol"}
        try:
            show = self.request("/api/show", {"model": model})
            tags = self.request("/api/tags").get("models", [])
            match = next((row for row in tags if row.get("name") in {model, model + ":latest"}), {})
            return {"details": show.get("details"), "model_info": show.get("model_info"),
                    "digest": match.get("digest"), "model_bytes": match.get("size")}
        except Exception as exc:
            return {"status": "metadata_unavailable", "error_type": type(exc).__name__}

    def resources(self):
        if self.protocol != "ollama":
            return None
        try:
            models = self.request("/api/ps").get("models", [])
            return [{key: row.get(key) for key in ("name", "size", "size_vram", "digest", "context_length")} for row in models]
        except Exception:
            return None

    def release(self, model):
        if self.protocol != "ollama":
            return {"status": "not_supported"}
        try:
            self.request("/api/embed", {"model": model, "input": [], "keep_alive": 0})
            return {"status": "unload_requested"}
        except Exception as exc:
            return {"status": "unload_failed", "error_type": type(exc).__name__}
