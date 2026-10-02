import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.llm import LLMRouter, LLMRouterError
from agent.output_schemas import Query
from agent.telemetry import capture_usage
from agents.planner_agent import PlannerAgent


SCHEMA = {"collections": {"customers": {"fields": {"name": ["string"]}},
                          "orders": {"fields": {"total": ["int"]}}}}


def install_response(monkeypatch, content, finish_reason="stop"):
    import agent.llm as module
    router = LLMRouter()
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content=content, response_metadata={"finish_reason": finish_reason})
    monkeypatch.setattr(router, "_client", lambda model: client)
    monkeypatch.setattr(module, "LLM_MAX_RETRIES", 1)
    return router, client


def test_invalid_plan_collection_fails_before_query_generation(monkeypatch):
    import agents.planner_agent as module
    router, client = install_response(monkeypatch, json.dumps({
        "operation": "aggregate", "collection": "name", "steps": ["group orders"]}))
    monkeypatch.setattr(module, "llm_router", router)
    with pytest.raises(LLMRouterError, match="selected collections"):
        PlannerAgent().plan("Top customers by spending", SCHEMA)
    client.invoke.assert_called_once()
    assert '"collection":"name"' not in client.invoke.call_args.args[0][0].content


def test_valid_plan_collection_is_accepted(monkeypatch):
    import agents.planner_agent as module
    router, _ = install_response(monkeypatch, json.dumps({
        "operation": "aggregate", "collection": "orders", "steps": ["sum total by customer"]}))
    monkeypatch.setattr(module, "llm_router", router)
    assert PlannerAgent().plan("Top customers", SCHEMA)["collection"] == "orders"


def test_truncation_diagnostic_and_finish_reason_are_preserved(monkeypatch, caplog):
    router, client = install_response(monkeypatch, '{"operation":"find",', "length")
    with capture_usage() as records, pytest.raises(LLMRouterError) as error:
        router.invoke("query_generator", [], Query)
    assert "output token limit" in str(error.value)
    assert "finish_reason=length" in str(error.value)
    assert "finish_reason=length" in caplog.text
    assert records[0]["finish_reason"] == "length"
    client.invoke.assert_called_once()


def test_schema_diagnostics_identify_fields_without_leaking_values(monkeypatch, caplog):
    secret = "private-customer-data"
    router, _ = install_response(monkeypatch, json.dumps({
        "operation": "find", "collection": "customers", "limit": secret, secret: "sensitive"}))
    with pytest.raises(LLMRouterError) as error:
        router.invoke("query_generator", [], Query)
    detail = str(error.value) + caplog.text
    assert "limit (int_type)" in detail
    assert "extra_forbidden" in detail
    assert secret not in detail
    assert "sensitive" not in detail
    assert "finish_reason=stop" in detail


def test_unknown_finish_reason_is_not_echoed(monkeypatch, caplog):
    router, _ = install_response(monkeypatch, "invalid", "private-provider-secret")
    with pytest.raises(LLMRouterError) as error:
        router.invoke("query_generator", [], Query)
    assert "finish_reason=unknown" in str(error.value)
    assert "private-provider-secret" not in str(error.value) + caplog.text
