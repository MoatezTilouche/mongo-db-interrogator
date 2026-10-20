"""Offline characterization and local endpoint safety; all providers are fakes."""
from dataclasses import replace
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from agent import client_factory as factory
from agent import llm
from agent.llm import LLMRouter, LLMRouterError

NEW_ENV = ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBEDDING_BASE_URL",
           "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "LLM_EXTRA_BODY",
           "LLM_DROP_PARAMS", "EMBEDDING_SEND_DIMENSIONS", "EMBEDDING_BATCH_SIZE", "LLM_STRUCTURED_OUTPUT",
           "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES", "SCHEMA_EMBEDDING_DIMENSIONS",
           "LANGSMITH_ALLOW_LOCAL", "TEMPLATE_PARAMETER_MODEL")


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    from agent import config
    for name in NEW_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "SCHEMA_EMBEDDING_DIMENSIONS", None)
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(factory.time, "sleep", Mock())
    yield
    from langsmith.utils import get_env_var
    get_env_var.cache_clear()


def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://local-model.test/private/v1")
    monkeypatch.setenv("LLM_MODEL", "test-chat")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")


def fake_sdk(monkeypatch, side_effect=None):
    import openai
    calls = []
    def create(**options):
        calls.append(options)
        data = [SimpleNamespace(index=i, embedding=[float(text), 1.0])
                for i, text in enumerate(options["input"])]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(prompt_tokens=len(data)))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock(side_effect=side_effect or create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    return client, constructor, calls


def test_legacy_constructors_preserve_exact_options(monkeypatch):
    import langchain_openai
    chat, embeddings = Mock(), Mock()
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", embeddings)
    callback = Mock()
    factory.get_chat_client("old-model", api_key="fake-key", callbacks=[callback])
    chat.assert_called_once_with(model="old-model", api_key="fake-key", timeout=30, max_retries=0, callbacks=[callback])
    factory.get_embedding_client("old-embedding", 2, api_key="fake-key")
    embeddings.assert_called_once_with(model="old-embedding", api_key="fake-key", dimensions=2)


def test_custom_endpoint_uses_placeholder_not_old_secret(monkeypatch):
    import langchain_openai
    local(monkeypatch)
    chat = Mock()
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    factory.get_chat_client(api_key="public-secret-must-not-be-sent")
    assert chat.call_args.kwargs["base_url"] == "http://local-model.test/private/v1"
    assert chat.call_args.kwargs["api_key"] == "unused"
    assert chat.call_args.kwargs["model"] == "test-chat"
    assert chat.call_args.kwargs["use_responses_api"] is False
    _, constructor, _ = fake_sdk(monkeypatch)
    factory.get_embedding_client(api_key="public-secret-must-not-be-sent")
    assert constructor.call_args.kwargs["base_url"] == chat.call_args.kwargs["base_url"]
    assert constructor.call_args.kwargs["api_key"] == "unused"


def test_standard_base_url_and_independent_embedding_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.internal.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://vectors.internal.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "fake-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "fake-vector")
    config = factory.get_config()
    assert config.chat_key == "fake-chat" and config.embedding_key == "fake-vector"
    assert config.host(embedding=True) == "vectors.internal.test"
    assert "fake-chat" not in repr(config) and "fake-vector" not in repr(config)


def test_embedding_only_endpoint_cannot_use_public_chat_default(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://vector.internal.test/v1")
    with pytest.raises(factory.ClientConfigError, match="LLM_BASE_URL"):
        factory.get_chat_client(api_key="public-secret")


@pytest.mark.parametrize("name,value", [("LLM_EXTRA_BODY", "[]"), ("LLM_EXTRA_BODY", "secret-not-json"),
    ("LLM_EXTRA_BODY", '{"n":NaN}'), ("LLM_BASE_URL", "secret-not-url"),
    ("LLM_BASE_URL", "https://user:secret@internal.test/v1"), ("EMBEDDING_BATCH_SIZE", "0"),
    ("EMBEDDING_DIMENSIONS", "invalid-secret"), ("LLM_TIMEOUT_SECONDS", "nan"),
    ("LLM_MAX_RETRIES", "-1"), ("EMBEDDING_SEND_DIMENSIONS", "sometimes"),
    ("LLM_STRUCTURED_OUTPUT", "sometimes")])
def test_bad_config_is_clear_and_does_not_echo_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(factory.ClientConfigError, match=name) as error:
        factory.get_config()
    assert value not in str(error.value)


def test_extra_body_merge_and_drop_at_actual_payload_boundary(monkeypatch):
    from langchain_core.messages import HumanMessage
    local(monkeypatch)
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"think":false,"nested":{"enabled":false},"temperature":9}')
    monkeypatch.setenv("LLM_DROP_PARAMS", "temperature, response_format, service_tier")
    # Construction is offline; no .invoke occurs. Exercise LangChain's final payload.
    client = factory.get_chat_client()
    payload = client._get_request_payload([HumanMessage(content="private prompt")],
        response_format={"type": "json_object"}, service_tier="auto", temperature=0.4,
        extra_body={"old_option": 1, "think": True})
    assert payload["extra_body"] == {"old_option": 1, "think": False, "nested": {"enabled": False}}
    assert not {"temperature", "response_format", "service_tier"} & payload.keys()


def test_dimensions_disabled_batching_order_and_logging(monkeypatch, caplog):
    local(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "2")
    monkeypatch.setenv("EMBEDDING_SEND_DIMENSIONS", "false")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("EMBEDDING_API_KEY", "private-embedding-key")
    _, constructor, calls = fake_sdk(monkeypatch)
    with caplog.at_level(logging.INFO):
        vectors = factory.get_embedding_client().embed_documents(["1", "2", "3", "4", "5"])
    assert vectors == [[float(i), 1.0] for i in range(1, 6)]
    assert [len(call["input"]) for call in calls] == [2, 2, 1]
    assert all("dimensions" not in call for call in calls)
    assert constructor.call_args.kwargs["timeout"] == 30
    assert caplog.text.count("embedding purpose=") == 3
    assert "private-embedding-key" not in caplog.text and "/private/v1" not in caplog.text


def test_wrong_vector_length_names_model_expected_and_actual(monkeypatch):
    local(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3")
    fake_sdk(monkeypatch)
    with pytest.raises(ValueError, match="test-embed: expected 3 dimensions, actual 2"):
        factory.get_embedding_client().embed_documents(["1"])


def test_dimensions_disabled_on_legacy_client(monkeypatch):
    import langchain_openai
    monkeypatch.setenv("EMBEDDING_SEND_DIMENSIONS", "false")
    constructor = Mock()
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", constructor)
    factory.get_embedding_client("model", 2, api_key="fake")
    assert "dimensions" not in constructor.call_args.kwargs


def test_embedding_timeout_retries_backoff(monkeypatch, caplog):
    local(monkeypatch)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1., 0.])], usage=None)
    client, constructor, _ = fake_sdk(monkeypatch, [TimeoutError("private-secret"), ConnectionError(), response])
    with caplog.at_level(logging.INFO):
        assert factory.get_embedding_client().embed_documents(["1"]) == [[1., 0.]]
    assert constructor.call_args.kwargs["timeout"] == 120
    assert client.embeddings.create.call_count == 3
    assert [call.args[0] for call in factory.time.sleep.call_args_list] == [0.25, 0.5]
    assert "retry_count=2" in caplog.text and "private-secret" not in caplog.text


class Output(BaseModel):
    ok: bool


@pytest.mark.parametrize("first", [type("Rejected", (Exception,), {"status_code": 400})("private-key"),
                                   SimpleNamespace(content="not JSON")])
def test_structured_fallback_consumes_existing_retry_once(monkeypatch, caplog, first):
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "true")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    bound = Mock()
    client = Mock()
    client.bind.return_value = bound
    if isinstance(first, Exception):
        bound.invoke.side_effect = first
    else:
        bound.invoke.return_value = first
    response = SimpleNamespace(content='{"ok":true}', usage_metadata={"input_tokens": 3, "output_tokens": 2})
    client.invoke.return_value = response
    router = LLMRouter()
    monkeypatch.setattr(router, "_client", lambda model: client)
    with caplog.at_level(logging.INFO):
        assert router.invoke("summary", [], Output) is response
    assert client.bind.call_count == 1 and bound.invoke.call_count == 1 and client.invoke.call_count == 1
    assert client.bind.call_args.kwargs["response_format"]["type"] == "json_schema"
    assert "structured_output_fallback" in caplog.text and "private-key" not in caplog.text


def test_structured_fallback_does_not_exceed_zero_retry_budget(monkeypatch, caplog):
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "true")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    client = Mock()
    client.bind.return_value.invoke.side_effect = type("Rejected", (Exception,), {"status_code": 422})()
    router = LLMRouter()
    monkeypatch.setattr(router, "_client", lambda model: client)
    with pytest.raises(LLMRouterError), caplog.at_level(logging.INFO):
        router.invoke("summary", [], Output)
    assert client.bind.return_value.invoke.call_count == 1
    assert client.invoke.call_count == 0


def test_chat_timeout_retry_backoff_and_secret_free_logs(monkeypatch, caplog):
    local(monkeypatch)
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    client = Mock()
    response = SimpleNamespace(content="answer", usage_metadata={"input_tokens": 7, "output_tokens": 2})
    client.invoke.side_effect = [TimeoutError("private-secret private prompt"), TimeoutError(), response]
    router = LLMRouter()
    monkeypatch.setattr(router, "_client", lambda model: client)
    with caplog.at_level(logging.INFO):
        router.invoke("summary", [])
    assert client.invoke.call_count == 3
    assert [call.args[0] for call in factory.time.sleep.call_args_list] == [0.25, 0.5]
    assert caplog.text.count("llm role=") == 3
    assert "host=local-model.test" in caplog.text and "prompt_tokens=7" in caplog.text
    assert "private-secret" not in caplog.text and "private prompt" not in caplog.text and "/private/v1" not in caplog.text


def test_client_cache_cannot_keep_public_client_after_local_switch(monkeypatch):
    import langchain_openai
    constructor = Mock(side_effect=[Mock(), Mock()])
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", constructor)
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "fake-public")
    router = LLMRouter()
    old = router._client("same-model")
    local(monkeypatch)
    assert router._client("same-model") is not old
    assert constructor.call_args.kwargs["base_url"] == "http://local-model.test/private/v1"


def test_local_configuration_disables_automatic_and_decorator_tracing(monkeypatch):
    from agent import tracing
    local(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    decorator = Mock(return_value=Mock())
    monkeypatch.setattr(tracing, "_traceable", decorator)
    @tracing.traceable(name="local-test")
    def action():
        return 5
    assert action() == 5
    decorator.return_value.return_value.assert_not_called()
    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


@pytest.mark.parametrize("endpoint", ["LLM_BASE_URL", "OPENAI_BASE_URL", "EMBEDDING_BASE_URL"])
@pytest.mark.parametrize("master,allow,expected", [
    ("true", "true", True),
    ("true", "false", False),
    ("false", "true", False),
    ("false", "false", False),
])
def test_tracing_policy_for_custom_endpoints(monkeypatch, endpoint, master, allow, expected):
    import os
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv(endpoint, "http://local-model.test/v1")
    monkeypatch.setenv("LANGSMITH_TRACING", master)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_ALLOW_LOCAL", allow)
    assert factory.configure_tracing() is expected
    assert os.environ["LANGCHAIN_TRACING_V2"] == str(expected).lower()
    # This is the same predicate LangChain uses to enable automatic LLM tracing.
    assert tracing_is_enabled() is expected


def test_local_tracing_opt_in_calls_instrumented_agents_and_master_switch_disables(monkeypatch):
    from agent import tracing

    local(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ALLOW_LOCAL", "true")
    traced_calls = Mock()

    def instrument(**options):
        def decorate(func):
            def run(*args, **kwargs):
                traced_calls(options["name"])
                return func(*args, **kwargs)
            return run
        return decorate

    monkeypatch.setattr(tracing, "_traceable", instrument)

    @tracing.traceable(name="local-test")
    def action():
        return 5

    factory.get_config()
    assert action() == 5
    traced_calls.assert_called_once_with("local-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert action() == 5
    traced_calls.assert_called_once_with("local-test")


def test_tracing_legacy_flag_and_master_precedence_without_custom_endpoint(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert factory.configure_tracing() is True
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert factory.configure_tracing() is False


def test_master_switch_disables_child_tracing_inside_enabled_context(monkeypatch):
    from agent import tracing
    from langsmith import tracing_context
    from langsmith.utils import tracing_is_enabled

    local(monkeypatch)
    monkeypatch.setenv("LANGSMITH_ALLOW_LOCAL", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    decorator = Mock(return_value=Mock())
    monkeypatch.setattr(tracing, "_traceable", decorator)

    @tracing.traceable(name="disabled-child")
    def action():
        return tracing_is_enabled()

    with tracing_context(enabled=True):
        assert action() is False
    decorator.return_value.return_value.assert_not_called()
