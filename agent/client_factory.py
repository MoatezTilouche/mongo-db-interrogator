"""Environment-driven clients. The unconfigured path retains legacy clients/options."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class ClientConfigError(ValueError):
    """An invalid environment option (values/secrets are never echoed)."""


def custom_endpoint_active():
    return any(os.getenv(name, "").strip() for name in (
        "LLM_BASE_URL", "OPENAI_BASE_URL", "EMBEDDING_BASE_URL"))


def _number(name, default, *, integer=False, minimum=0):
    raw = os.getenv(name, "").strip()
    try:
        value = (int if integer else float)(raw) if raw else default
        if not math.isfinite(value) or value < minimum:
            raise ValueError
        return value
    except (ValueError, TypeError, OverflowError):
        raise ClientConfigError(f"{name} must be {'an integer' if integer else 'a number'} >= {minimum}") from None


def _boolean(name, default):
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value not in {"true", "false"}:
        raise ClientConfigError(f"{name} must be true or false")
    return value == "true"


def _url(name, value):
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError
        parsed.port
        if parsed.query or parsed.fragment:
            raise ValueError
    except ValueError:
        raise ClientConfigError(f"{name} must be an absolute HTTP(S) URL without credentials, query or fragment") from None
    return value.rstrip("/")


@dataclass(frozen=True)
class ClientConfig:
    chat_url: str | None
    embedding_url: str | None
    chat_key: str = field(repr=False)
    embedding_key: str = field(repr=False)
    chat_model: str | None
    embedding_model: str | None
    dimensions: int | None
    timeout: float
    retries: int
    extra_body: dict = field(repr=False)
    drop_params: frozenset[str]
    send_dimensions: bool
    batch_size: int
    structured_output: bool | None
    embedding_options_active: bool

    def host(self, *, embedding=False):
        url = self.embedding_url if embedding else self.chat_url
        return urlsplit(url).hostname if url else "sdk_default"


def get_config(*, api_key=None, timeout=None, retries=None, dimensions=None):
    # Lazy import: config loads .env and provides the existing defaults.
    from agent import config as legacy
    chat_url = _url("LLM_BASE_URL / OPENAI_BASE_URL", os.getenv("LLM_BASE_URL", "").strip()
                    or os.getenv("OPENAI_BASE_URL", "").strip())
    embedding_url = _url("EMBEDDING_BASE_URL", os.getenv("EMBEDDING_BASE_URL", "").strip()) or chat_url
    # An embedding-only endpoint must never silently leave chat on a public default.
    if embedding_url and not chat_url:
        raise ClientConfigError("LLM_BASE_URL (or OPENAI_BASE_URL) is required with EMBEDDING_BASE_URL")
    if custom_endpoint_active():
        # Includes LangChain's automatic tracing, not just application decorators.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    old_key = legacy.OPENAI_API_KEY if api_key is None else api_key
    chat_key = os.getenv("LLM_API_KEY", "").strip() or ("unused" if chat_url else old_key)
    embedding_key = os.getenv("EMBEDDING_API_KEY", "").strip() or (
        chat_key if embedding_url == chat_url else "unused")
    raw_body = os.getenv("LLM_EXTRA_BODY", "").strip()
    try:
        def reject_constant(_):
            raise ValueError
        extra_body = json.loads(raw_body, parse_constant=reject_constant) if raw_body else {}
        if not isinstance(extra_body, dict):
            raise ValueError
    except (ValueError, TypeError):
        raise ClientConfigError("LLM_EXTRA_BODY must be a valid JSON object") from None
    dimensions = legacy.SCHEMA_EMBEDDING_DIMENSIONS if dimensions is None else dimensions
    if os.getenv("EMBEDDING_DIMENSIONS", "").strip():
        dimensions = _number("EMBEDDING_DIMENSIONS", None, integer=True, minimum=1)
    elif dimensions is not None and (not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0):
        raise ClientConfigError("SCHEMA_EMBEDDING_DIMENSIONS must be a positive integer")
    return ClientConfig(
        chat_url, embedding_url, chat_key, embedding_key,
        os.getenv("LLM_MODEL", "").strip() or None,
        os.getenv("EMBEDDING_MODEL", "").strip() or None, dimensions,
        _number("LLM_TIMEOUT_SECONDS", legacy.LLM_TIMEOUT_SECONDS if timeout is None else timeout, minimum=0.001),
        _number("LLM_MAX_RETRIES", legacy.LLM_MAX_RETRIES if retries is None else retries, integer=True),
        extra_body, frozenset(item.strip() for item in os.getenv("LLM_DROP_PARAMS", "").split(",") if item.strip()),
        _boolean("EMBEDDING_SEND_DIMENSIONS", True),
        _number("EMBEDDING_BATCH_SIZE", 32, integer=True, minimum=1),
        _boolean("LLM_STRUCTURED_OUTPUT", None),
        bool(embedding_url or any(os.getenv(name, "").strip() for name in (
            "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_SEND_DIMENSIONS", "EMBEDDING_BATCH_SIZE",
            "EMBEDDING_API_KEY"))),
    )


def cache_identity(config):
    """Keep client caches isolated when operator configuration changes."""
    return (config.chat_url, config.embedding_url, config.chat_key, config.embedding_key,
            config.chat_model, config.embedding_model, config.dimensions, config.timeout,
            config.retries, json.dumps(config.extra_body, sort_keys=True), config.drop_params,
            config.send_dimensions, config.batch_size, config.structured_output)


def request_body(payload, config):
    """Apply options at the final payload boundary, including library defaults."""
    payload = dict(payload)
    body = {**(payload.pop("extra_body", None) or {}), **config.extra_body}
    for name in config.drop_params:
        payload.pop(name, None)
        body.pop(name, None)
    if body:
        payload["extra_body"] = body
    return payload


def get_chat_client(model=None, *, api_key=None, timeout=None, retries=None, callbacks=None):
    from agent import config as legacy
    config = get_config(api_key=api_key, timeout=timeout, retries=retries)
    if not config.chat_key:
        raise ClientConfigError("LLM_API_KEY or OPENAI_API_KEY is required without LLM_BASE_URL")
    from langchain_openai import ChatOpenAI
    options = dict(model=config.chat_model or model or legacy.OPENAI_MODEL, api_key=config.chat_key,
                   timeout=config.timeout, max_retries=0, callbacks=callbacks)
    if config.chat_url:
        options.update(base_url=config.chat_url, use_responses_api=False)
    if config.extra_body or config.drop_params:
        class ConfiguredChat(ChatOpenAI):
            def _get_request_payload(self, *args, **kwargs):
                return request_body(super()._get_request_payload(*args, **kwargs), config)
        return ConfiguredChat(**options)
    return ChatOpenAI(**options)


def recoverable(exc):
    status = getattr(exc, "status_code", None)
    return (isinstance(exc, (TimeoutError, ConnectionError)) or status == 429
            or isinstance(status, int) and status >= 500
            or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"})


def backoff(attempt):
    time.sleep(min(0.25 * (2 ** attempt), 2.0))


class EmbeddingClient:
    def __init__(self, client, model, dimensions, config, *, direct=False, purpose="schema_embedding"):
        self.client, self.model, self.dimensions = client, model, dimensions
        self.config, self.direct, self.purpose = config, direct, purpose

    def embed_documents(self, texts, *, purpose=None):
        vectors = []
        size = self.config.batch_size if self.config.embedding_options_active else max(1, len(texts))
        for start in range(0, len(texts), size):
            chunk = texts[start:start + size]
            for attempt in range((self.config.retries if self.direct else 0) + 1):
                begun = time.perf_counter()
                tokens = None
                try:
                    if self.direct:
                        options = {"dimensions": self.dimensions} if self.config.send_dimensions and self.dimensions else {}
                        response = self.client.embeddings.create(model=self.model, input=chunk, **options)
                        data = sorted(response.data, key=lambda item: item.index)
                        batch = [item.embedding for item in data]
                        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
                    else:
                        batch = self.client.embed_documents(chunk)
                    if len(batch) != len(chunk):
                        raise ValueError("Embedding provider returned a mismatched vector count")
                    # Checking each result also protects later batches/model switches in one process.
                    for vector in batch:
                        if self.dimensions is not None and len(vector) != self.dimensions:
                            raise ValueError(f"Embedding model {self.model}: expected {self.dimensions} dimensions, actual {len(vector)}")
                    vectors.extend(batch)
                    break
                except Exception as exc:
                    if self.direct and attempt < self.config.retries and recoverable(exc):
                        backoff(attempt)
                        continue
                    raise
                finally:
                    logger.info("embedding purpose=%s model=%s host=%s latency_ms=%.1f prompt_tokens=%s "
                                "completion_tokens=0 retry_count=%d structured_output_fallback=False",
                                purpose or self.purpose, self.model, self.config.host(embedding=True),
                                (time.perf_counter() - begun) * 1000, tokens, attempt)
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embedding_client(model=None, dimensions=None, *, api_key=None, purpose="schema_embedding",
                         legacy_options=None):
    from agent import config as legacy
    config = get_config(api_key=api_key, dimensions=dimensions)
    model = config.embedding_model or model or legacy.SCHEMA_EMBEDDING_MODEL
    if not config.embedding_key:
        raise ClientConfigError("EMBEDDING_API_KEY or OPENAI_API_KEY is required without EMBEDDING_BASE_URL")
    if config.embedding_url:
        # The SDK is already a dependency of langchain-openai. No tokenizer/network downloads.
        from openai import OpenAI
        client = OpenAI(base_url=config.embedding_url, api_key=config.embedding_key,
                        timeout=config.timeout, max_retries=0)
        return EmbeddingClient(client, model, config.dimensions, config, direct=True, purpose=purpose)
    from langchain_openai import OpenAIEmbeddings
    options = dict(legacy_options or {})
    if not config.send_dimensions:
        options.pop("dimensions", None)
    if config.send_dimensions and config.dimensions is not None:
        options["dimensions"] = config.dimensions
    client = OpenAIEmbeddings(model=model, api_key=config.embedding_key, **options)
    return EmbeddingClient(client, model, config.dimensions, config, purpose=purpose)


def schema_response_format(schema):
    if hasattr(schema, "feedback_response_format"):
        return schema.feedback_response_format()
    if hasattr(schema, "model_json_schema"):
        document = schema.model_json_schema()
    elif isinstance(schema, dict):
        document = schema
    else:
        return None
    return {"type": "json_schema", "json_schema": {
        "name": getattr(schema, "__name__", "output"), "schema": document, "strict": True}}
