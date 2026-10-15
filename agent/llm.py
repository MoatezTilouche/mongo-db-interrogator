"""Centralized provider/model routing for all LLM-backed agents."""
from __future__ import annotations

import logging
import sys
import time
from typing import Any
from pydantic import ValidationError

from agent.config import (
    LLM_MAX_RETRIES,
    LLM_ROLE_CONFIG,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    SCHEMA_EMBEDDING_MODEL,
    SCHEMA_EMBEDDING_DIMENSIONS,
)
from agent.llm_structured import parse_json_from_text
from agent.token_usage import get_token_usage_callback
from agent.telemetry import record_attempt, current_request_id
from agent.client_factory import (get_chat_client, get_embedding_client, get_config,
                                  schema_response_format, backoff, custom_endpoint_active, cache_identity, EmbeddingClient)

logger = logging.getLogger(__name__)


class LLMRouterError(RuntimeError):
    """Raised when the configured OpenAI model fails."""

    def __init__(self, message: str, *, role: str | None = None,
                 code: str = "llm_unavailable", detail: str | None = None):
        super().__init__(message)
        self.role = role
        self.code = code
        self.detail = detail


class StructuredOutputError(ValueError):
    """Raised when a provider returns invalid output for the requested schema."""


class LLMRouter:
    def __init__(self) -> None:
        self._clients: dict[tuple, Any] = {}
        self._embedding_clients: dict[tuple, Any] = {}

    def _client(self, model: str) -> Any:
        identity = (model, cache_identity(get_config(api_key=OPENAI_API_KEY,
                    timeout=LLM_TIMEOUT_SECONDS, retries=LLM_MAX_RETRIES)))
        if identity in self._clients:
            return self._clients[identity]

        callback = get_token_usage_callback()
        callbacks = [callback] if callback else None
        client = get_chat_client(model, api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS,
                                 retries=LLM_MAX_RETRIES, callbacks=callbacks)
        self._clients[identity] = client
        return client

    def _embedding_client(self, model: str | None = None, dimensions: int | None = None) -> Any:
        model = model or SCHEMA_EMBEDDING_MODEL
        dimensions = SCHEMA_EMBEDDING_DIMENSIONS if dimensions is None else dimensions
        identity = (model, dimensions, cache_identity(get_config(api_key=OPENAI_API_KEY, dimensions=dimensions)))
        if identity in self._embedding_clients:
            return self._embedding_clients[identity]
        client = get_embedding_client(model, dimensions, api_key=OPENAI_API_KEY)
        self._embedding_clients[identity] = client
        return client

    def embed_documents(self, texts: list[str], model: str | None = None,
                        dimensions: int | None = None) -> list[list[float]]:
        model = model or SCHEMA_EMBEDDING_MODEL
        config = get_config(api_key=OPENAI_API_KEY, dimensions=dimensions)
        model = config.embedding_model or model
        started = time.perf_counter()
        client = self._embedding_client(model, dimensions)
        caller = sys._getframe(1)
        if caller.f_code.co_name == "embed_query" and caller.f_globals.get("__name__") == __name__:
            caller = caller.f_back
        purpose = caller.f_globals.get("__name__", "schema_embedding")
        if isinstance(client, EmbeddingClient):
            vectors = client.embed_documents(texts, purpose=purpose)
        else:
            vectors = client.embed_documents(texts)
        try:
            if custom_endpoint_active():
                # Token estimates must not trigger tokenizer downloads in local mode.
                raise LookupError
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = sum(len(encoding.encode(text)) for text in texts)
        except Exception:
            input_tokens = sum(max(1, len(text) // 4) for text in texts)
        from agent.telemetry import record_embedding
        record_embedding(model, input_tokens, (time.perf_counter() - started) * 1000)
        return vectors

    def embed_query(self, text: str, model: str | None = None) -> list[float]:
        return self.embed_documents([text], model)[0]

    @staticmethod
    def _is_recoverable(exc: Exception) -> bool:
        if isinstance(exc, (StructuredOutputError, TimeoutError, ConnectionError)):
            return True
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        status = status or getattr(response, "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            return True
        reason = str(exc).lower()
        return any(term in reason for term in (
            "429", "rate limit", "quota", "resource exhausted", "timeout",
            "timed out", "connection", "temporarily unavailable", "service unavailable",
            "model unavailable", "model not found", "not found", "5xx",
            "malformed", "not valid json", "structured response",
        ))

    @staticmethod
    def _validate_structured(content: str, schema: Any) -> None:
        try:
            parsed = parse_json_from_text(content)
            if schema is dict and not isinstance(parsed, dict):
                raise StructuredOutputError("Structured response must be a JSON object.")
            if schema is list and not isinstance(parsed, list):
                raise StructuredOutputError("Structured response must be a JSON array.")
            if hasattr(schema, "model_validate"):
                schema.model_validate(parsed)
            elif callable(schema) and schema not in (dict, list):
                schema(parsed)
        except StructuredOutputError:
            raise
        except ValidationError as exc:
            # Never expose inputs, model-generated keys, URLs or validator contexts.
            from agent.output_schemas import Query, QueryContext
            known = set().union(*(model.model_fields for model in (Query, QueryContext)))
            known.update(getattr(schema, "model_fields", {}))
            details = []
            for error in exc.errors(include_input=False, include_context=False, include_url=False)[:5]:
                location = error.get("loc", ())
                field = location[0] if location and location[0] in known else "output"
                details.append(f"{field} ({error['type']})")
            from agent.output_schemas import GeneratedQuery
            if schema is GeneratedQuery:
                # Binding validators use a small fixed set of messages. Expose
                # those actionable reasons, never arbitrary validator contexts.
                allowed = {
                    "Unexpected query parameters", "Invalid parameter placeholder",
                    "Only limits and filter literal values may be parameterized",
                    "Limit parameter must be a positive integer named limit",
                    "Filter parameters must be literal data, not operators or placeholders",
                    "Membership parameters must be arrays", "Regex parameters must be strings",
                    "String placeholders are not supported",
                }
                for error in exc.errors(include_input=False, include_context=True, include_url=False)[:5]:
                    reason = str((error.get("ctx") or {}).get("error", ""))
                    if reason in allowed:
                        details.append(reason)
                    elif reason.startswith("Missing parameter:"):
                        details.append("A template parameter is missing its corresponding parameters value")
            raise StructuredOutputError("Output schema validation failed: " + ", ".join(details)) from None
        except ValueError:
            raise StructuredOutputError("Response is not valid JSON or does not match the output contract.") from None
        except Exception:
            raise StructuredOutputError("Structured output validation failed.") from None

    @staticmethod
    def _finish_reason(response: Any) -> str:
        metadata = getattr(response, "response_metadata", None) or {}
        reason = metadata.get("finish_reason")
        return reason if reason in ("stop", "length", "content_filter", "tool_calls", "function_call", "error") else "unknown"

    def invoke(self, role: str, messages: list[Any], structured_output_schema: Any = None) -> Any:
        try:
            model = LLM_ROLE_CONFIG[role]
        except KeyError as exc:
            raise ValueError(f"Unknown LLM role: {role}") from exc

        config = get_config(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS, retries=LLM_MAX_RETRIES)
        model = config.chat_model or model

        from agent.output_schemas import GeneratedQuery, QueryContext
        query_contract = role in {"query_generator", "repair"} and structured_output_schema is GeneratedQuery
        context_contract = role == "query_context" and structured_output_schema is QueryContext
        request_messages = list(messages)
        format_retries = 0
        retry_count = 0
        structured_fallback = False
        for attempt in range(config.retries + 1):
            retry_count = attempt
            started = time.perf_counter()
            response = None
            using_schema = False
            try:
                client = self._client(model)
                response_format = schema_response_format(structured_output_schema) if config.structured_output is True else None
                if structured_fallback or config.structured_output is False:
                    pass  # Existing local parser/validators handle the plain response.
                elif response_format:
                    client = client.bind(response_format=response_format)
                    using_schema = "response_format" not in config.drop_params
                elif role == "feedback_analysis" and hasattr(structured_output_schema, "feedback_response_format"):
                    # Bind per invocation, leaving other roles' settings intact.
                    client = client.bind(response_format=structured_output_schema.feedback_response_format())
                elif query_contract or context_contract:
                    # MongoDB filter/parameter maps have dynamic keys, so enforce
                    # a JSON object here and keep the full local schema validator.
                    client = client.bind(response_format={"type": "json_object"})
                response = client.invoke(request_messages)
                if structured_output_schema is not None:
                    self._validate_structured(getattr(response, "content", "") or "", structured_output_schema)
                return response
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                if using_schema and not structured_fallback and (
                        isinstance(exc, StructuredOutputError) or status in {400, 415, 422}):
                    structured_fallback = True
                    logger.warning("structured_output_fallback role=%s model=%s retry_count=%d request_id=%s",
                                   role, model, attempt, current_request_id())
                    # Reparse the same response first, with the existing repair parser.
                    # Rejection/invalid output consumes a retry, never an extra call.
                    if response is not None:
                        try:
                            self._validate_structured(getattr(response, "content", "") or "", structured_output_schema)
                            return response
                        except StructuredOutputError:
                            pass
                    if attempt < config.retries:
                        continue
                    if not isinstance(exc, StructuredOutputError):
                        raise LLMRouterError(f"LLM structured output rejected for {role}", role=role) from None
                recoverable = self._is_recoverable(exc)
                if not recoverable:
                    raise
                if isinstance(exc, StructuredOutputError) and (query_contract or context_contract) and format_retries == 0 \
                        and attempt < config.retries and self._finish_reason(response) != "content_filter" \
                        and not (getattr(response, "additional_kwargs", None) or {}).get("refusal"):
                    from langchain_core.messages import HumanMessage
                    format_retries += 1
                    # Only sanitized schema diagnostics; do not echo raw output,
                    # result rows, credentials, or model-produced field names.
                    guidance = (
                        f"The previous response failed JSON validation: {exc}. "
                        "Regenerate the same read-only query using only the selected schema and original constraints. "
                        "Return one JSON object with semantic_intent, template (a query object), and parameters (an object). "
                        "These are sibling keys. Return a query specification, never result rows or text-content blocks. "
                        "Use concrete filter literals; the backend parameterizes them. Never use $param strings. "
                        "parameters is {} unless template has an explicit limit placeholder {\"$param\":\"limit\"}; "
                        "then parameters contains only its integer limit value. "
                        "Preserve requested filters, ordering and limits; do not invent schema fields."
                    ) if query_contract else (
                        f"The previous response failed JSON validation: {exc}. "
                        "Return one JSON object with complexity, operation, collections and fields. "
                        "complexity must be exactly simple or complex, never aggregate. "
                        "operation must be find, count, distinct or aggregate; aggregate requires complex. "
                        "collections is an array of catalog collection names; fields maps those names to arrays of real field names. "
                        "Preserve the question's meaning and applicable supplied business rules."
                    )
                    request_messages = [*messages, HumanMessage(content=guidance)]
                    logger.warning("llm structured_retry role=%s model=%s retry_count=%d detail=%s request_id=%s",
                                   role, model, attempt + 1, str(exc), current_request_id())
                    continue
                if attempt < config.retries and not isinstance(exc, StructuredOutputError):
                    backoff(attempt)
                    continue
                message = f"OpenAI request failed for {role} ({type(exc).__name__})"
                if isinstance(exc, StructuredOutputError):
                    finish_reason = self._finish_reason(response)
                    detail = str(exc)
                    if finish_reason == "length":
                        detail += "; response reached the output token limit and may be incomplete"
                    logger.warning("llm structured_failure role=%s request_id=%s finish_reason=%s detail=%s",
                                   role, current_request_id(), finish_reason, detail)
                    message += f": {detail}; finish_reason={finish_reason}"
                code = ("invalid_query_output" if query_contract else "invalid_llm_output") \
                    if isinstance(exc, StructuredOutputError) else "llm_unavailable"
                raise LLMRouterError(message, role=role, code=code,
                                     detail=detail if isinstance(exc, StructuredOutputError) else None) from None
            finally:
                record = record_attempt(role, model, response, (time.perf_counter() - started) * 1000, False)
                record["finish_reason"] = self._finish_reason(response)
                record["retry_count"] = attempt
                record["structured_output_retries"] = format_retries
                record["structured_output_fallback"] = structured_fallback
                logger.info("llm role=%s purpose=%s model=%s host=%s latency_ms=%.1f prompt_tokens=%s "
                            "completion_tokens=%s retry_count=%d fallback_used=False structured_output_fallback=%s request_id=%s",
                            role, role, model, config.host(), record["latency_ms"], record["input_tokens"],
                            record["output_tokens"], attempt, structured_fallback, current_request_id())
        raise LLMRouterError(f"OpenAI request failed for {role}")


llm_router = LLMRouter()
