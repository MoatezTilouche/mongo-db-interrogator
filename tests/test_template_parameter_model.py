"""Parameter extraction uses its selected chat model, without external calls."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.llm import LLMRouter, LLMRouterError
from agent.telemetry import capture_usage
from services import template_parameter_extractor as extractor


@pytest.fixture
def local_models(monkeypatch):
    import langchain_openai
    for name, value in {
        "LLM_BASE_URL": "http://offline.test/v1", "LLM_MODEL": "large-local-model",
        "TEMPLATE_PARAMETER_MODEL": "small-local-model", "LLM_API_KEY": "unused",
        "LLM_STRUCTURED_OUTPUT": "true", "LLM_MAX_RETRIES": "0", "LLM_EXTRA_BODY": "",
        "LLM_DROP_PARAMS": "", "LANGSMITH_TRACING": "false", "LANGSMITH_ALLOW_LOCAL": "false",
    }.items():
        monkeypatch.setenv(name, value)
    clients = []

    def construct(**options):
        client = Mock()
        client.bind.return_value = client
        client.invoke.return_value = SimpleNamespace(content=json.dumps({
            "can_extract": True, "parameters": {"filter_code": "NEW-2", "limit": 1}}))
        clients.append((options, client))
        return client

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", factory)
    return factory, clients


def document():
    return {
        "source_question": "Find records with code OLD-1",
        "template": {"operation": "find", "collection": "records", "filter": {"code": {"$param": "filter_code"}},
                     "limit": {"$param": "limit"}},
        "parameter_schema": {
            "filter_code": {"type": "string", "source_path": "filter.code", "example": "OLD-1",
                            "user_variable": True, "source_visible": True},
            "limit": {"type": "integer", "source_path": "limit", "example": 1,
                      "user_variable": False, "source_visible": False},
        },
    }


def test_fallback_uses_small_model_at_constructor_and_reports_actual_model(local_models, monkeypatch):
    factory, clients = local_models
    router = LLMRouter()
    monkeypatch.setattr(extractor, "llm_router", router)
    saved = document()
    with capture_usage() as usage:
        values, method = extractor.extract_parameters("Records linked to NEW-2 please", saved)
    assert method == "llm" and values == {"filter_code": "NEW-2", "limit": 1}
    assert saved == document()
    assert factory.call_args.kwargs["model"] == "small-local-model"
    assert factory.call_args.kwargs["base_url"] == "http://offline.test/v1"
    assert usage[0]["agent"] == "template_parameters" and usage[0]["model"] == "small-local-model"
    clients[0][1].invoke.assert_called_once()
    # Existing structured response validation remains enabled for the small model.
    assert clients[0][1].bind.call_args.kwargs["response_format"]["type"] == "json_schema"


def test_deterministic_extraction_still_makes_no_model_call(local_models, monkeypatch):
    factory, _ = local_models
    monkeypatch.setattr(extractor, "llm_router", LLMRouter())
    values, method = extractor.extract_parameters("Find records with code NEW-2", document())
    assert method == "deterministic" and values == {"filter_code": "NEW-2", "limit": 1}
    factory.assert_not_called()


def test_main_roles_keep_large_model_and_clients_are_cached_separately(local_models):
    factory, clients = local_models
    router = LLMRouter()
    for role in ("template_parameters", "query_generator", "template_parameters", "query_context", "repair"):
        router.invoke(role, [], dict)
    assert factory.call_count == 2
    assert [options["model"] for options, _ in clients] == ["small-local-model", "large-local-model"]
    assert clients[0][1].invoke.call_count == 2
    assert clients[1][1].invoke.call_count == 3


def test_changing_extraction_model_replaces_cached_client(local_models, monkeypatch):
    factory, _ = local_models
    router = LLMRouter()
    router.invoke("template_parameters", [], dict)
    monkeypatch.setenv("TEMPLATE_PARAMETER_MODEL", "another-small-model")
    router.invoke("template_parameters", [], dict)
    assert [call.kwargs["model"] for call in factory.call_args_list] == ["small-local-model", "another-small-model"]


@pytest.mark.parametrize("role,variable", [
    ("query_context", "QUERY_CONTEXT_MODEL"),
    ("query_generator", "QUERY_GENERATOR_MODEL"),
    ("repair", "QUERY_REPAIR_MODEL"),
    ("feedback_analysis", "FEEDBACK_ANALYSIS_MODEL"),
    ("summary", "SUMMARY_MODEL"),
    ("schema_enrichment", "SCHEMA_ENRICHMENT_MODEL"),
])
def test_role_override_routes_constructor_telemetry_and_cached_client(local_models, monkeypatch, role, variable):
    factory, _ = local_models
    router = LLMRouter()
    monkeypatch.setenv(variable, "role-model")
    with capture_usage() as usage:
        router.invoke(role, [], dict)
    assert factory.call_args.kwargs["model"] == "role-model"
    assert usage[0]["model"] == "role-model"
    router.invoke(role, [], dict)
    assert factory.call_count == 1
    monkeypatch.setenv(variable, "replacement-model")
    router.invoke(role, [], dict)
    assert factory.call_args.kwargs["model"] == "replacement-model"
    monkeypatch.setenv(variable, "   ")
    router.invoke(role, [], dict)
    assert factory.call_args.kwargs["model"] == "large-local-model"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_override_uses_existing_global_model(local_models, monkeypatch, blank):
    factory, _ = local_models
    monkeypatch.setenv("TEMPLATE_PARAMETER_MODEL", blank)
    LLMRouter().invoke("template_parameters", [], dict)
    assert factory.call_args.kwargs["model"] == "large-local-model"


def test_small_model_failure_does_not_silently_call_large_model(local_models, monkeypatch):
    factory, clients = local_models
    router = LLMRouter()
    client = router._client("small-local-model", chat_role="template_parameters")
    client.invoke.side_effect = TimeoutError("synthetic timeout")
    with pytest.raises(LLMRouterError):
        router.invoke("template_parameters", [], dict)
    assert factory.call_count == 1 and clients[0][0]["model"] == "small-local-model"


@pytest.mark.parametrize("parameters", [
    {"filter_code": "NEW-2", "limit": "1"},
    {"filter_code": "NEW-2", "limit": 99},
    {"filter_code": "NEW-2", "limit": 1, "extra": "invented"},
])
def test_small_model_cannot_bypass_types_constants_or_allowed_keys(local_models, monkeypatch, parameters):
    _, clients = local_models
    router = LLMRouter()
    client = router._client("small-local-model", chat_role="template_parameters")
    client.invoke.return_value = SimpleNamespace(content=json.dumps({"can_extract": True, "parameters": parameters}))
    monkeypatch.setattr(extractor, "llm_router", router)
    with pytest.raises((ValueError, LLMRouterError)):
        extractor.extract_parameters("Records linked to NEW-2 please", document())
    assert len(clients) == 1
