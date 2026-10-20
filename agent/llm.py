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

    def _client(self, model: str, *, chat_role: str | None = None) -> Any:
        route = {"chat_role": chat_role} if chat_role else {}
        identity = (model, cache_identity(get_config(api_key=OPENAI_API_KEY,
                    timeout=LLM_TIMEOUT_SECONDS, retries=LLM_MAX_RETRIES, **route)))
        if identity in self._clients:
            return self._clients[identity]

        callback = get_token_usage_callback()
        callbacks = [callback] if callback else None
        client = get_chat_client(model, api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS,
                                 retries=LLM_MAX_RETRIES, callbacks=callbacks, **route)
        self._clients[identity] = client
        return client

    def _embedding_client(self, model: str | None = None, dimensions: int | None = None,
                          *, embedding_role: str | None = None) -> Any:
        route = {"embedding_role": embedding_role} if embedding_role else {}
        if embedding_role is None and dimensions is None:
            dimensions = SCHEMA_EMBEDDING_DIMENSIONS
        config = get_config(api_key=OPENAI_API_KEY, dimensions=dimensions, **route)
        model = config.embedding_model or model or SCHEMA_EMBEDDING_MODEL
        dimensions = config.dimensions
        identity = (model, dimensions, cache_identity(config))
        if identity in self._embedding_clients:
            return self._embedding_clients[identity]
        client = get_embedding_client(model, dimensions, api_key=OPENAI_API_KEY, **route)
        self._embedding_clients[identity] = client
        return client

    def embed_documents(self, texts: list[str], model: str | None = None,
                        dimensions: int | None = None, *, embedding_role: str | None = None) -> list[list[float]]:
        route = {"embedding_role": embedding_role} if embedding_role else {}
        model = model or SCHEMA_EMBEDDING_MODEL
        config = get_config(api_key=OPENAI_API_KEY, dimensions=dimensions, **route)
        model = config.embedding_model or model
        started = time.perf_counter()
        client = self._embedding_client(model, dimensions, **route)
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

    def embed_query(self, text: str, model: str | None = None, *,
                    embedding_role: str | None = None) -> list[float]:
        route = {"embedding_role": embedding_role} if embedding_role else {}
        config = get_config(api_key=OPENAI_API_KEY, **route)
        if config.query_instruction:
            text = f"Instruct: {config.query_instruction}\nQuery: {text}"
        return self.embed_documents([text], model, **route)[0]

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

        route = {"chat_role": role} if role == "template_parameters" else {}
        config = get_config(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS, retries=LLM_MAX_RETRIES, **route)
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
            response_format_type = "none"
            try:
                client = self._client(model, **route)
                response_format = schema_response_format(structured_output_schema) if config.structured_output is True else None
                if structured_fallback or config.structured_output is False:
                    pass  # Existing local parser/validators handle the plain response.
                elif response_format:
                    client = client.bind(response_format=response_format)
                    response_format_type = response_format.get("type", "none")
                elif getattr(structured_output_schema, "plan_output", False):
                    client = client.bind(response_format=schema_response_format(structured_output_schema))
                    response_format_type = "json_schema"
                elif role == "feedback_analysis" and hasattr(structured_output_schema, "feedback_response_format"):
                    # Bind per invocation, leaving other roles' settings intact.
                    client = client.bind(response_format=structured_output_schema.feedback_response_format())
                    response_format_type = "json_schema"
                elif query_contract or context_contract:
                    # MongoDB filter/parameter maps have dynamic keys, so enforce
                    # a JSON object here and keep the full local schema validator.
                    client = client.bind(response_format={"type": "json_object"})
                    response_format_type = "json_object"
                # Report the final request policy, including environment overrides.
                override = (getattr(config, "extra_body", {}) or {}).get("response_format")
                if override is not None:
                    response_format_type = override.get("type") if isinstance(override, dict) else "none"
                if "response_format" in getattr(config, "drop_params", ()):
                    response_format_type = "none"
                if not isinstance(response_format_type, str) or response_format_type not in {"json_schema", "json_object"}:
                    response_format_type = "none"
                using_schema = response_format_type == "json_schema"
                response = client.invoke(request_messages)
                if structured_output_schema is not None and not getattr(structured_output_schema, "plan_output", False):
                    self._validate_structured(getattr(response, "content", "") or "", structured_output_schema)
                return response
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                http_status = status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
                # Provider exception messages can contain credentials, request
                # bodies and URLs. Keep startup diagnostics useful without them.
                logger.warning("llm_request_failed role=%s model=%s error_type=%s status_code=%s "
                               "retry_count=%d timeout_seconds=%s response_format=%s request_id=%s",
                               role, model, type(exc).__name__, http_status, attempt,
                               config.timeout, response_format_type, current_request_id())
                if using_schema and not structured_fallback and (
                        status in {400, 415, 422} or isinstance(exc, StructuredOutputError)
                        and not (query_contract or context_contract)):
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
                # Invalid query/context output does not mean the provider rejected
                # schema support. Keep schema constraints for the one bounded
                # format-regeneration attempt below, rather than retrying plain text.
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
                        "Write a requested limit as an integer in template.limit or the final top-level $limit. "
                        "parameters must be {}; the backend creates bindings. Do not emit any $param placeholders. "
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
                record["response_format"] = response_format_type
                logger.info("llm role=%s purpose=%s model=%s host=%s latency_ms=%.1f prompt_tokens=%s "
                            "completion_tokens=%s retry_count=%d fallback_used=False structured_output_fallback=%s request_id=%s",
                            role, role, model, config.host(), record["latency_ms"], record["input_tokens"],
                            record["output_tokens"], attempt, structured_fallback, current_request_id())
        raise LLMRouterError(f"OpenAI request failed for {role}")


llm_router = LLMRouter()
