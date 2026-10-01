import logging

import pytest

import agent.llm as llm_module
from agent.llm import LLMRouter, LLMRouterError
from agents.validator_agent import QueryValidationError, ValidatorAgent


class Response:
    def __init__(self, content):
        self.content = content


class ProviderError(Exception):
    def __init__(self, status_code, message="provider failure"):
        super().__init__(message)
        self.status_code = status_code


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def install_clients(monkeypatch, clients):
    router = LLMRouter()
    calls = []

    def client(model):
        calls.append(model)
        return clients[model]

    monkeypatch.setattr(router, "_client", client)
    return router, calls


def test_role_mapping_uses_configured_primary_and_fallback(monkeypatch):
    primary = "qwen/qwen3.8-27b:free"
    fallback = "openrouter/free"
    router, calls = install_clients(monkeypatch, {
        primary: FakeClient(response=Response('{"ok": true}')),
        fallback: FakeClient(response=Response('{"ok": true}')),
    })

    router.invoke("query_generator", [], dict)

    assert calls == [primary]


@pytest.mark.parametrize("role, primary, fallback", [
    ("schema_selection", "qwen/qwen3.8-27b:free", "openrouter/free"),
    ("planner", "qwen/qwen3.8-27b:free", "openrouter/free"),
    ("query_generator", "qwen/qwen3.8-27b:free", "openrouter/free"),
    ("repair", "openrouter/free", "qwen/qwen3.8-27b:free"),
    ("summary", "openrouter/free", "qwen/qwen3.8-27b:free"),
    ("python_artifact", "qwen/qwen3.8-27b:free", "openrouter/free"),
])
def test_each_role_uses_its_configured_provider_and_model(monkeypatch, role, primary, fallback):
    router, calls = install_clients(monkeypatch, {
        primary: FakeClient(response=Response('{"ok": true}')),
        fallback: FakeClient(response=Response('{"ok": true}')),
    })

    router.invoke(role, [], dict)

    assert calls == [primary]


@pytest.mark.parametrize("status", [429, 504])
def test_openrouter_falls_back_on_rate_limit_and_5xx(monkeypatch, status):
    primary = "qwen/qwen3.8-27b:free"
    fallback = "openrouter/free"
    router, calls = install_clients(monkeypatch, {
        primary: FakeClient(error=ProviderError(status)),
        fallback: FakeClient(response=Response('{"ok": true}')),
    })
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 0)

    router.invoke("query_generator", [], dict)

    assert calls == [primary, fallback]


def test_openrouter_falls_back_to_second_model_on_timeout(monkeypatch):
    primary = "openrouter/free"
    fallback = "qwen/qwen3.8-27b:free"
    router, calls = install_clients(monkeypatch, {
        primary: FakeClient(error=TimeoutError("timed out")),
        fallback: FakeClient(response=Response("answer")),
    })
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 0)

    router.invoke("summary", [])

    assert calls == [primary, fallback]


def test_malformed_structured_output_falls_back_without_retry(monkeypatch):
    primary = "qwen/qwen3.8-27b:free"
    fallback = "openrouter/free"
    primary_client = FakeClient(response=Response("not json"))
    router, calls = install_clients(monkeypatch, {
        primary: primary_client,
        fallback: FakeClient(response=Response('{"operation": "find"}')),
    })
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 1)

    response = router.invoke("query_generator", [], dict)

    assert response.content == '{"operation": "find"}'
    assert primary_client.calls == 1
    assert calls == [primary, fallback]


def test_no_fallback_for_mongo_validation_errors():
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({"operation": "find", "collection": "missing"}, {
            "collections": {"orders": {"fields": {}}},
        })


def test_both_models_failing_raises_router_error(monkeypatch):
    primary = "qwen/qwen3.8-27b:free"
    fallback = "openrouter/free"
    router, _ = install_clients(monkeypatch, {
        primary: FakeClient(error=ProviderError(503)),
        fallback: FakeClient(error=ProviderError(429)),
    })
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 0)

    with pytest.raises(LLMRouterError):
        router.invoke("query_generator", [])


def test_telemetry_has_model_and_never_api_key(monkeypatch, caplog):
    primary = "openrouter/free"
    secret = "super-secret-api-key"
    router, _ = install_clients(monkeypatch, {
        primary: FakeClient(response=Response("answer")),
    })
    with caplog.at_level(logging.INFO, logger="agent.llm"):
        router.invoke("summary", [])

    text = caplog.text
    assert "model=openrouter/free" in text
    assert "fallback_used=False" in text
    assert secret not in text


def test_fallback_telemetry_records_actual_provider_and_model(monkeypatch, caplog):
    primary = "qwen/qwen3.8-27b:free"
    fallback = "openrouter/free"
    router, _ = install_clients(monkeypatch, {
        primary: FakeClient(error=ProviderError(429)),
        fallback: FakeClient(response=Response("answer")),
    })
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 0)

    with caplog.at_level(logging.INFO, logger="agent.llm"):
        router.invoke("query_generator", [])

    assert "model=openrouter/free" in caplog.text
    assert "fallback_used=True" in caplog.text


def test_non_openrouter_provider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LLMRouter()._client("openrouter", "qwen/qwen3.8-27b:free")