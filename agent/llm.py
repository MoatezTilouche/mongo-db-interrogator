"""Centralized provider/model routing for all LLM-backed agents."""
from __future__ import annotations

import logging
import time
from typing import Any
from pydantic import ValidationError

from agent.config import (
    LLM_MAX_RETRIES,
    LLM_ROLE_CONFIG,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
)
from agent.llm_structured import parse_json_from_text
from agent.token_usage import get_token_usage_callback
from agent.telemetry import record_attempt, current_request_id

logger = logging.getLogger(__name__)


class LLMRouterError(RuntimeError):
    """Raised when the configured OpenAI model fails."""


class StructuredOutputError(ValueError):
    """Raised when a provider returns invalid output for the requested schema."""


class LLMRouter:
    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def _client(self, model: str) -> Any:
        if model in self._clients:
            return self._clients[model]

        callback = get_token_usage_callback()
        callbacks = [callback] if callback else None
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
            callbacks=callbacks,
        )
        self._clients[model] = client
        return client

    @staticmethod
    def _is_recoverable(exc: Exception) -> bool:
        if isinstance(exc, StructuredOutputError):
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

        retry_count = 0
        for attempt in range(LLM_MAX_RETRIES + 1):
            retry_count = attempt
            started = time.perf_counter()
            response = None
            try:
                response = self._client(model).invoke(messages)
                if structured_output_schema is not None:
                    self._validate_structured(str(getattr(response, "content", "") or ""), structured_output_schema)
                logger.info(
                    "llm role=%s model=%s fallback_used=%s retry_count=%d latency_ms=%.1f request_id=%s",
                    role, model, False, retry_count,
                    (time.perf_counter() - started) * 1000,
                    current_request_id(),
                )
                return response
            except Exception as exc:
                recoverable = self._is_recoverable(exc)
                if not recoverable:
                    raise
                if attempt < LLM_MAX_RETRIES and not isinstance(exc, StructuredOutputError):
                    continue
                logger.warning(
                    "llm failure role=%s model=%s fallback_used=%s retry_count=%d latency_ms=%.1f reason=%s request_id=%s",
                    role, model, False, retry_count,
                    (time.perf_counter() - started) * 1000, type(exc).__name__,
                    current_request_id(),
                )
                message = f"OpenAI request failed for {role} ({type(exc).__name__})"
                if isinstance(exc, StructuredOutputError):
                    finish_reason = self._finish_reason(response)
                    detail = str(exc)
                    if finish_reason == "length":
                        detail += "; response reached the output token limit and may be incomplete"
                    logger.warning("llm structured_failure role=%s request_id=%s finish_reason=%s detail=%s",
                                   role, current_request_id(), finish_reason, detail)
                    message += f": {detail}; finish_reason={finish_reason}"
                raise LLMRouterError(message) from None
            finally:
                record = record_attempt(role, model, response, (time.perf_counter() - started) * 1000, False)
                record["finish_reason"] = self._finish_reason(response)
                logger.debug("llm telemetry=%s", record)
        raise LLMRouterError(f"OpenAI request failed for {role}")


llm_router = LLMRouter()
