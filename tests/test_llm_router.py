import logging
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import agent.llm as module
from agent.llm import LLMRouter, LLMRouterError, StructuredOutputError
from agent.telemetry import capture_usage, totals

@pytest.mark.parametrize("role", list(module.LLM_ROLE_CONFIG))
def test_all_roles_use_single_openrouter_model(monkeypatch, role):
    router = LLMRouter()
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content='{"ok":true}')
    factory = Mock(return_value=client)
    monkeypatch.setattr(router, "_client", factory)
    router.invoke(role, [], dict)
    factory.assert_called_once_with(module.LLM_ROLE_CONFIG[role])
    assert len(set(module.LLM_ROLE_CONFIG.values())) == 1

@pytest.mark.parametrize("error", [TimeoutError("timed out"), type("RateLimitError", (Exception,), {"status_code":429})("rate limit")])
def test_transient_error_retries_same_model_without_fallback(monkeypatch, error):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    client.invoke.side_effect = error
    factory = Mock(return_value=client)
    monkeypatch.setattr(router, "_client", factory)
    with capture_usage() as records, pytest.raises(LLMRouterError):
        router.invoke("planner", [])
    assert client.invoke.call_count == 2
    assert {call.args[0] for call in factory.call_args_list} == {module.LLM_ROLE_CONFIG["planner"]}
    assert all(not record["fallback_used"] for record in records)


def test_structured_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content="not json")
    monkeypatch.setattr(router, "_client", lambda model: client)
    with pytest.raises(LLMRouterError, match="StructuredOutputError"):
        router.invoke("query_generator", [], dict)
    client.invoke.assert_called_once()


def test_retry_can_succeed_and_telemetry_counts_tokens(monkeypatch):
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    response = SimpleNamespace(content="answer", usage_metadata={"input_tokens":12,"output_tokens":3})
    client.invoke.side_effect = [TimeoutError("timed out"), response]
    monkeypatch.setattr(router, "_client", lambda model: client)
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
    monkeypatch.setattr(router, "_client", lambda model: client)
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 0)
    with caplog.at_level(logging.INFO), pytest.raises(LLMRouterError) as failure:
        router.invoke("summary", [])
    assert "private-credential" not in str(failure.value) + caplog.text
    assert "fallback_used=False" in caplog.text


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM role"):
        LLMRouter().invoke("missing", [])


def test_other_provider_signature_is_not_supported():
    with pytest.raises(TypeError):
        LLMRouter()._client("another-provider", "model")


def test_openrouter_client_does_not_set_an_output_token_cap(monkeypatch):
    import langchain_openai
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", constructor)
    monkeypatch.setattr(module, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(module, "get_token_usage_callback", lambda: None)
    LLMRouter()._client("openrouter/free")
    options = constructor.call_args.kwargs
    assert "max_tokens" not in options
    assert "max_completion_tokens" not in options
    assert options["model"] == "openrouter/free"
