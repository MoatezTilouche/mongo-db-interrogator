"""Provider block parsing, bounded regeneration, and public failure diagnostics."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agent.llm as llm_module
from agent.llm import LLMRouter, LLMRouterError
from agent.llm_structured import extract_text_content, parse_json_from_text
from agent.output_schemas import GeneratedQuery, QueryContext
from agent.telemetry import capture_usage, totals
from agents.query_generator_agent import QueryGeneratorAgent
from agents.query_context_agent import QueryContextAgent
from services.mongo_capabilities import MongoCapabilities
from services.query_parameter_binder import bind_query_parameters
from pydantic import ValidationError


SCHEMA = {"collections": {"orders_proc_hist": {"fields": {
    "commandId": ["string"], "status": ["string"], "_id": ["ObjectId"],
}}}}
VALID = {"semantic_intent": "Retrieve recent failed records", "template": {
    "operation": "find", "collection": "orders_proc_hist",
    "filter": {"status": {"$param": "status"}}, "sort": [["_id", -1]],
    "limit": {"$param": "limit"}}, "parameters": {"limit": 5, "status": "Failed"}}
# Exact malformed structure reported by the user, without credential data.
BROKEN = ('{"semantic_intent":"Retrieve the five most recent failed microservices from orders_proc_hist",'
          '"template":{"operation":"find","collection":"orders_proc_hist","filter":{"status":"Failed"},'
          '"projection":null,"sort":[["_id",-1]],"limit":{"$param":"limit"}},{"parameters":{"limit":5}}}')


def message(content, finish_reason="stop", **metadata):
    return SimpleNamespace(content=content, response_metadata={"finish_reason": finish_reason},
                           usage_metadata={"input_tokens": 20, "output_tokens": 10}, **metadata)


def install(monkeypatch, responses, retries=1):
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", retries)
    router, client, bound = LLMRouter(), Mock(), Mock()
    client.bind.return_value = bound
    bound.invoke.side_effect = responses
    monkeypatch.setattr(router, "_client", lambda model: client)
    monkeypatch.setattr("agents.query_generator_agent.llm_router", router)
    return QueryGeneratorAgent(MongoCapabilities()), router, client, bound


@pytest.mark.parametrize("kind", ["text", "output_text"])
def test_real_provider_text_blocks_parse_without_metadata(kind):
    blocks = [{"type": "reasoning", "text": "not JSON"},
              {"type": kind, "text": json.dumps(VALID)}]
    original = deepcopy(blocks)
    assert parse_json_from_text(blocks) == VALID
    assert blocks == original


def test_split_text_and_fenced_json_keep_existing_parser_behavior():
    parts = [{"type": "text", "text": '{"a":'}, {"type": "text", "text": "1}"}]
    assert parse_json_from_text(parts) == {"a": 1}
    assert parse_json_from_text("```json\n{}\n```") == {}
    assert parse_json_from_text("null") is None


def test_result_rows_and_nontext_blocks_are_not_reinterpreted_as_queries():
    with pytest.raises(ValueError):
        extract_text_content([{"commandId": "a-record"}])
    with pytest.raises(ValueError):
        extract_text_content([{"type": "reasoning", "text": "untrusted reasoning"}])


def test_user_response_regenerates_once_and_preserves_query_intent(monkeypatch):
    agent, _, client, bound = install(monkeypatch, [
        message([{"type": "text", "text": BROKEN}]),
        message([{"type": "text", "text": json.dumps(VALID)}]),
    ])
    with capture_usage() as records:
        generated = agent.generate("Retrieve five most recent failed records", SCHEMA, operation_hint="find")
    assert bind_query_parameters(generated["template"], generated["parameters"]) == {
        "operation": "find", "collection": "orders_proc_hist", "filter": {"status": "Failed"},
        "sort": [["_id", -1]], "limit": 5}
    assert bound.invoke.call_count == 2
    assert all(call.kwargs == {"response_format": {"type": "json_object"}} for call in client.bind.call_args_list)
    first, second = [call.args[0] for call in bound.invoke.call_args_list]
    assert second[0] is first[0]  # Preserve the original question, selected schema, and constraints.
    assert "sibling keys" in second[-1].content and BROKEN not in second[-1].content
    assert totals(records)["total_llm_calls"] == 2
    assert totals(records)["total_input_tokens"] == 40
    assert records[-1]["structured_output_retries"] == 1


@pytest.mark.parametrize("invalid", ["[]", "null", '"a string"', "{}", '{"template": []}',
                                     '{"semantic_intent":"x","template":{"operation":"find","collection":"orders_proc_hist"},"parameters":{"unused":1}}'])
def test_invalid_shape_or_parameters_can_recover(monkeypatch, invalid):
    agent, _, _, bound = install(monkeypatch, [message(invalid), message(json.dumps(VALID))])
    generated = agent.generate("Recent failed records", SCHEMA, operation_hint="find")
    assert generated["parameters"]["limit"] == 5
    assert bound.invoke.call_count == 2


@pytest.mark.parametrize("role", ["query_generator", "repair"])
def test_correct_responses_take_one_call(monkeypatch, role):
    _, router, client, bound = install(monkeypatch, [message(json.dumps(VALID))])
    router.invoke(role, [], GeneratedQuery)
    client.bind.assert_called_once_with(response_format={"type": "json_object"})
    bound.invoke.assert_called_once_with([])


def test_invalid_output_retries_at_most_once_and_keeps_errors_sanitized(monkeypatch, caplog):
    secret = "private-model-output"
    agent, _, _, bound = install(monkeypatch, [message(json.dumps([secret])), message(json.dumps([secret]))], retries=8)
    with pytest.raises(LLMRouterError) as caught:
        agent.generate("Recent records", SCHEMA, operation_hint="find")
    assert caught.value.code == "invalid_query_output"
    assert caught.value.role == "query_generator" and "model_type" in caught.value.detail
    assert bound.invoke.call_count == 2
    assert secret not in str(caught.value) + caplog.text


def test_binding_retry_identifies_unused_parameters_without_exposing_values(monkeypatch):
    secret = "private-unused-value"
    invalid = {"semantic_intent": "Count records", "template": {
        "operation": "count_documents", "collection": "orders_proc_hist", "filter": {}},
        "parameters": {"private_unused_name": secret}}
    _, router, _, bound = install(monkeypatch, [message(json.dumps(invalid)), message(json.dumps(VALID))])
    router.invoke("query_generator", [], GeneratedQuery)
    guidance = bound.invoke.call_args_list[1].args[0][-1].content
    assert "Unexpected query parameters" in guidance
    assert secret not in guidance and "private_unused_name" not in guidance


@pytest.mark.parametrize("placeholder,parameters", [
    ("$param.status", {"status": "Failed"}),
    ("$param", {"status": "Failed"}),
    ("$param", {"param": "Failed"}),
])
def test_explicit_parameter_syntax_normalizes_without_changing_filters(placeholder, parameters):
    raw = {"semantic_intent": "Find failed records", "template": {
        "operation": "find", "collection": "orders_proc_hist", "filter": {"status": placeholder}},
        "parameters": parameters}
    original = deepcopy(raw)
    parsed = GeneratedQuery.model_validate(raw).model_dump(by_alias=True, exclude_unset=True)
    assert raw == original
    assert bind_query_parameters(parsed["template"], parsed["parameters"])["filter"] == {"status": "Failed"}


def test_parameter_syntax_does_not_guess_missing_values_or_unrelated_names():
    raw = {"template": {"operation": "find", "collection": "orders_proc_hist", "filter": {"status": "$param"}},
           "parameters": {"unrelated": "Failed"}}
    with pytest.raises(ValidationError):
        GeneratedQuery.model_validate(raw)


def test_normalized_parameter_values_still_cannot_inject_operators():
    raw = {"template": {"operation": "find", "collection": "orders_proc_hist", "filter": {"status": "$param.status"}},
           "parameters": {"status": {"$ne": "Failed"}}}
    with pytest.raises(ValidationError):
        GeneratedQuery.model_validate(raw)


def test_context_count_and_aggregate_aliases_have_canonical_public_output():
    raw = {"complexity": "simple", "operation": "count_documents", "collections": ["orders_proc_hist"],
           "fields": {"orders_proc_hist": ["status"]}}
    assert QueryContext.model_validate(raw).operation == "count"
    assert raw["operation"] == "count_documents"
    grouped = QueryContext.model_validate({**raw, "complexity": "aggregate", "operation": "aggregate"})
    assert grouped.complexity == "complex" and grouped.operation == "aggregate"


def test_identical_literal_and_parameter_limit_keep_same_value_without_retry():
    raw = {"template": {"operation": "find", "collection": "orders_proc_hist", "filter": {}, "limit": 5},
           "parameters": {"limit": 5}}
    original = deepcopy(raw)
    parsed = GeneratedQuery.model_validate(raw).model_dump(by_alias=True, exclude_unset=True)
    assert raw == original
    assert parsed["template"]["limit"] == {"$param": "limit"}
    assert bind_query_parameters(parsed["template"], parsed["parameters"])["limit"] == 5


def test_mismatched_limit_and_parameter_are_not_silently_changed():
    raw = {"template": {"operation": "find", "collection": "orders_proc_hist", "filter": {}, "limit": 5},
           "parameters": {"limit": 10}}
    with pytest.raises(ValidationError):
        GeneratedQuery.model_validate(raw)


def test_generated_count_alias_preserves_count_semantics():
    raw = {"template": {"operation": "count", "collection": "orders_proc_hist", "filter": {"status": "Failed"}},
           "parameters": {}}
    result = GeneratedQuery.model_validate(raw)
    assert result.template.operation == "count_documents"
    assert result.template.filter == {"status": "Failed"}


def test_invalid_context_enum_retries_once(monkeypatch):
    bad = {"complexity": "wrong", "operation": "find", "collections": ["orders_proc_hist"],
           "fields": {"orders_proc_hist": ["commandId"]}}
    _, router, _, bound = install(monkeypatch, [message(json.dumps(bad)), message(json.dumps({**bad, "complexity": "simple"}))])
    router.invoke("query_context", [], QueryContext)
    assert bound.invoke.call_count == 2
    assert "complexity must be exactly simple or complex" in bound.invoke.call_args_list[1].args[0][-1].content


def test_zero_retry_setting_is_respected(monkeypatch):
    agent, _, _, bound = install(monkeypatch, [message("[]")], retries=0)
    with pytest.raises(LLMRouterError):
        agent.generate("Recent records", SCHEMA, operation_hint="find")
    assert bound.invoke.call_count == 1


@pytest.mark.parametrize("finish,refusal", [("content_filter", None), ("stop", "refused")])
def test_refusals_do_not_trigger_format_regeneration(monkeypatch, finish, refusal):
    agent, _, _, bound = install(monkeypatch, [message("", finish, additional_kwargs={"refusal": refusal})])
    with pytest.raises(LLMRouterError):
        agent.generate("Recent records", SCHEMA, operation_hint="find")
    assert bound.invoke.call_count == 1


def pipeline(monkeypatch, generator):
    import orchestrator as module
    monkeypatch.setattr(module.schema_bootstrap_service, "get_catalog", lambda: deepcopy(SCHEMA))
    monkeypatch.setattr(module.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module.schema_bootstrap_service, "retrieve", lambda *args: {
        "catalog": deepcopy(SCHEMA), "telemetry": {}, "_query_embedding": []})
    app = module.OrchestratorAgent()
    app.preference_service = Mock(user_id="default")
    app.preference_service.retrieve_for_query.return_value = []
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [], "telemetry": {}, "_active_rules": []}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [], "user_preferences": []}
    app.query_context_agent = Mock()
    app.query_context_agent.build.return_value = {
        "query_context": {"collections": ["orders_proc_hist"], "complexity": "simple",
                          "operation": "find", "fields": {"orders_proc_hist": ["commandId", "status", "_id"]}},
        "schema_spec": deepcopy(SCHEMA)}
    app.query_generator = generator
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": []}
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "test"}))
    monkeypatch.setattr(module.LearnedOptimizationService, "apply", lambda self, query, *args: (query, None))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    return app


def test_recovered_query_executes_only_after_binding_and_validation(monkeypatch):
    agent, _, _, bound = install(monkeypatch, [message([{"type": "text", "text": BROKEN}]), message(json.dumps(VALID))])
    app = pipeline(monkeypatch, agent)
    result = app.run("Retrieve five most recent failed records")
    assert result["execution_success"] is True, result
    assert result["telemetry"]["total_llm_calls"] == 2
    assert result["validated_query"]["filter"] == {"status": "Failed"}
    assert result["validated_query"]["limit"] == 5
    app.executor.execute.assert_called_once()
    assert bound.invoke.call_count == 2


def test_failed_generation_exposes_specific_error_and_never_executes(monkeypatch):
    agent, _, _, _ = install(monkeypatch, [message("[]"), message("[]")])
    app = pipeline(monkeypatch, agent)
    result = app.run("Retrieve recent records")
    assert result["execution_error_code"] == "invalid_query_output"
    assert "model_type" in result["execution_error"]
    assert result["validation_errors"][0]["code"] == "invalid_query_output"
    assert result["generated_query"] is None and result["validated_query"] is None
    app.executor.execute.assert_not_called()


def test_provider_errors_remain_sanitized_in_api_result(monkeypatch):
    agent, _, _, _ = install(monkeypatch, [TimeoutError("secret-provider-details"), TimeoutError("secret-provider-details")])
    app = pipeline(monkeypatch, agent)
    result = app.run("Retrieve records")
    assert result["execution_error_code"] == "llm_unavailable"
    assert "secret-provider-details" not in json.dumps(result)
    app.executor.execute.assert_not_called()


def test_recent_failed_microservices_keep_required_schema_fields(monkeypatch):
    catalog = {"collections": {"orders_proc_hist": {"fields": {
        "commandId": ["string"], "status": ["string"], "microservice": ["string"],
        "executionDate": ["string"], "unrelatedPayload": ["string"],
    }}}}
    context = {"complexity": "simple", "operation": "find", "collections": ["orders_proc_hist"],
               "fields": {"orders_proc_hist": ["commandId"]}}
    invoke = Mock(return_value=message(json.dumps(context)))
    monkeypatch.setattr("agents.query_context_agent.llm_router.invoke", invoke)
    result = QueryContextAgent().build("Give me the five most recent failed microservices", catalog)
    selected = result["schema_spec"]["collections"]["orders_proc_hist"]["fields"]
    assert set(selected) == {"commandId", "status", "microservice", "executionDate"}
    invoke.assert_called_once()


def test_unrelated_context_does_not_add_status_or_recency_fields(monkeypatch):
    catalog = {"collections": {"orders_proc_hist": {"fields": {
        "commandId": ["string"], "status": ["string"], "executionDate": ["string"],
    }}}}
    context = {"complexity": "simple", "operation": "find", "collections": ["orders_proc_hist"],
               "fields": {"orders_proc_hist": ["commandId"]}}
    monkeypatch.setattr("agents.query_context_agent.llm_router.invoke", Mock(return_value=message(json.dumps(context))))
    result = QueryContextAgent().build("Show commandId", catalog)
    assert result["query_context"]["fields"]["orders_proc_hist"] == ["commandId"]


def test_explicit_count_by_microservice_uses_aggregation(monkeypatch):
    catalog = {"collections": {"orders_proc_hist": {"fields": {
        "status": ["string"], "microservice": ["string"],
    }}}}
    context = {"complexity": "simple", "operation": "count", "collections": ["orders_proc_hist"],
               "fields": {"orders_proc_hist": ["status"]}}
    monkeypatch.setattr("agents.query_context_agent.llm_router.invoke", Mock(return_value=message(json.dumps(context))))
    result = QueryContextAgent().build("Count failed records by microservice, highest count first.", catalog)
    assert result["query_context"]["operation"] == "aggregate"
    assert result["query_context"]["complexity"] == "complex"
    assert result["query_context"]["fields"]["orders_proc_hist"] == ["status", "microservice"]
