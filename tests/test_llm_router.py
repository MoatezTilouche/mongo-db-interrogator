import logging
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import agent.llm as module
from agent.llm import LLMRouter, LLMRouterError, StructuredOutputError
from agent.telemetry import capture_usage, totals

@pytest.mark.parametrize("role", list(module.LLM_ROLE_CONFIG))
def test_roles_use_their_configured_model(monkeypatch, role):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    for variable in module.__dict__.get("LLM_ROLE_MODEL_ENV", {}).values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("SCHEMA_ENRICHMENT_MODEL", raising=False)
    monkeypatch.setitem(module.LLM_ROLE_CONFIG, role, "test-model")
    router = LLMRouter()
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content='{"ok":true}')
    factory = Mock(return_value=client)
    monkeypatch.setattr(router, "_client", factory)
    router.invoke(role, [], dict)
    route = {"chat_role": role}
    factory.assert_called_once_with("test-model", **route)

@pytest.mark.parametrize("error", [TimeoutError("timed out"), type("RateLimitError", (Exception,), {"status_code":429})("rate limit")])
def test_transient_error_retries_same_model_without_fallback(monkeypatch, error):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    client.invoke.side_effect = error
    factory = Mock(return_value=client)
    monkeypatch.setattr(router, "_client", factory)
    with capture_usage() as records, pytest.raises(LLMRouterError):
        router.invoke("query_context", [])
    assert client.invoke.call_count == 2
    assert {call.args[0] for call in factory.call_args_list} == {module.LLM_ROLE_CONFIG["query_context"]}
    assert all(not record["fallback_used"] for record in records)


def test_structured_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content="not json")
    monkeypatch.setattr(router, "_client", lambda model, **kwargs: client)
    with pytest.raises(LLMRouterError, match="StructuredOutputError"):
        router.invoke("query_generator", [], dict)
    client.invoke.assert_called_once()


def test_retry_can_succeed_and_telemetry_counts_tokens(monkeypatch):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    response = SimpleNamespace(content="answer", usage_metadata={"input_tokens":12,"output_tokens":3})
    client.invoke.side_effect = [TimeoutError("timed out"), response]
    monkeypatch.setattr(router, "_client", lambda model, **kwargs: client)
    with capture_usage() as records:
        assert router.invoke("summary", []) == response
    usage = totals(records)
    assert usage["total_llm_calls"] == 2
    assert usage["total_input_tokens"] == 12
    assert usage["total_output_tokens"] == 3
    assert all(not r["fallback_used"] for r in records)


def test_failures_do_not_expose_provider_secrets(monkeypatch, caplog):
    router = LLMRouter()
    client = Mock()
    client.invoke.side_effect = TimeoutError("private-credential timed out")
    monkeypatch.setattr(router, "_client", lambda model, **kwargs: client)
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 0)
    with caplog.at_level(logging.INFO), pytest.raises(LLMRouterError) as failure:
        router.invoke("summary", [])
    assert "private-credential" not in str(failure.value) + caplog.text
    assert "fallback_used=False" in caplog.text


@pytest.mark.parametrize("error_class,status,retries,recoverable", [
    (TimeoutError, None, 1, True),
    (ConnectionError, None, 0, True),
    (type("RateLimitError", (Exception,), {}), 429, 1, True),
    (type("ServerError", (Exception,), {}), 503, 0, True),
    (type("AuthenticationError", (Exception,), {}), 401, 0, False),
    (type("BadRequestError", (Exception,), {}), 400, 0, False),
])
def test_schema_enrichment_failure_logs_safe_cause(monkeypatch, caplog, error_class, status, retries, recoverable):
    secret = "private-credential-and-provider-body"
    error = error_class(secret)
    error.status_code = status
    config = SimpleNamespace(chat_model="offline-model", retries=retries, structured_output=False,
                             timeout=30, host=lambda: "offline.test")
    monkeypatch.setattr(module, "get_config", lambda **kwargs: config)
    backoff = Mock()
    monkeypatch.setattr(module, "backoff", backoff)
    router, client = LLMRouter(), Mock()
    client.invoke.side_effect = error
    monkeypatch.setattr(router, "_client", lambda model, **kwargs: client)
    with caplog.at_level(logging.WARNING), pytest.raises(LLMRouterError if recoverable else error_class):
        router.invoke("schema_enrichment", [])
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("llm_request_failed ")]
    assert len(lines) == retries + 1
    assert all("role=schema_enrichment" in line and f"error_type={error_class.__name__}" in line
               and f"status_code={status}" in line and "timeout_seconds=30" in line for line in lines)
    assert secret not in caplog.text
    assert client.invoke.call_count == retries + 1
    assert backoff.call_count == retries


def test_schema_invalid_json_diagnostic_hides_response(monkeypatch, caplog):
    secret = "private-model-output-and-credential"
    config = SimpleNamespace(chat_model="offline-model", retries=0, structured_output=False,
                             timeout=30, host=lambda: "offline.test")
    monkeypatch.setattr(module, "get_config", lambda **kwargs: config)
    router, client = LLMRouter(), Mock()
    client.invoke.return_value = SimpleNamespace(content=secret)
    monkeypatch.setattr(router, "_client", lambda model, **kwargs: client)
    with caplog.at_level(logging.WARNING), pytest.raises(LLMRouterError) as failure:
        router.invoke("schema_enrichment", [], dict)
    assert "error_type=StructuredOutputError" in caplog.text
    assert failure.value.code == "invalid_llm_output"
    assert secret not in caplog.text + str(failure.value)
    client.invoke.assert_called_once()


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM role"):
        LLMRouter().invoke("missing", [])


def test_client_factory_accepts_only_a_model_argument():
    with pytest.raises(TypeError):
        LLMRouter()._client("another-provider", "model")


def test_openai_client_does_not_set_an_output_token_cap(monkeypatch):
    import langchain_openai
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", constructor)
    monkeypatch.setattr(module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(module, "get_token_usage_callback", lambda: None)
    LLMRouter()._client("gpt-6.1-sol")
    options = constructor.call_args.kwargs
    assert "temperature" not in options
    assert "max_tokens" not in options
    assert "max_completion_tokens" not in options
    assert options["model"] == "gpt-6.1-sol"
