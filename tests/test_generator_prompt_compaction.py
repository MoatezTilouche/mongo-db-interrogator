from copy import deepcopy
from datetime import datetime as real_datetime
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.query_generator_agent import QueryGeneratorAgent
from agent.output_schemas import GeneratedQuery, Query
from services.mongo_capabilities import MongoCapabilities
from services.prompt_context import compact_knowledge, compact_preferences
from services.query_prompt_instructions import operation_instructions
from services.query_parameter_binder import bind_query_parameters


SCHEMA = {"collections": {"orders_proc_hist": {"fields": {"actionCode": ["string"],
          "status": ["string"], "executionDate": ["string"]}}}}


def response(operation):
    template = {"operation": operation, "collection": "orders_proc_hist"}
    if operation == "aggregate":
        template["pipeline"] = [{"$match": {"status": {"$param": "status"}}}]
    elif operation == "distinct":
        template.update(field="actionCode", filter={"status": {"$param": "status"}})
    else:
        template["filter"] = {"status": {"$param": "status"}}
    return {"semantic_intent": "Retrieve failed orders", "template": template, "parameters": {"status": "failed"}}


def capture(monkeypatch, operation="find", supports_mixed=False):
    import agents.query_generator_agent as module
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(response(operation))))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    return QueryGeneratorAgent(MongoCapabilities(supports_mixed)), invoke


@pytest.mark.parametrize("hint,operation", [("find", "find"), ("count", "count_documents"),
                         ("count_documents", "count_documents"), ("distinct", "distinct"), ("aggregate", "aggregate")])
def test_generator_sends_only_chosen_shape_and_simple_operations_omit_aggregation(monkeypatch, hint, operation):
    agent, invoke = capture(monkeypatch, operation)
    result = agent.generate("Show failed orders", SCHEMA, operation_hint=hint)
    prompt = invoke.call_args.args[1][0].content
    examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"operation"')]
    assert len(examples) == 1
    assert examples[0]["operation"] == operation
    Query.model_validate(examples[0])
    assert invoke.call_args.args[2] is GeneratedQuery
    expected = response(operation)
    assert bind_query_parameters(result["template"], result["parameters"]) == bind_query_parameters(expected["template"], expected["parameters"])
    assert result["semantic_intent"] == expected["semantic_intent"]
    if operation != "count_documents":
        assert '"$param":"limit"' in prompt
        assert "Explicit current requests override stored defaults" in prompt
    else:
        assert '"$param":"limit"' not in prompt
    assert "read-only" in prompt
    assert "never bind $expr field references" in prompt
    if operation != "aggregate":
        assert "$lookup" not in prompt
        assert "$group" not in prompt
        assert "$match" not in prompt
        assert "Never combine pipeline" not in prompt
    else:
        assert "$group" in prompt
        assert "$match" in prompt
        assert "Never combine pipeline" in prompt


def test_generator_prompt_includes_today(monkeypatch):
    import agents.query_generator_agent as module

    monkeypatch.setattr(module, "datetime", Mock(now=Mock(return_value=real_datetime(2026, 10, 12))))
    agent, invoke = capture(monkeypatch)

    agent.generate("Show orders", SCHEMA)

    assert "Today: 2026-10-12" in invoke.call_args.args[1][0].content


@pytest.mark.parametrize("hint", ["unknown", None, "unexpected"])
def test_unknown_hints_keep_general_shapes_and_deployment_guidance(monkeypatch, hint):
    agent, invoke = capture(monkeypatch)
    agent.generate("Show orders", SCHEMA, operation_hint=hint)
    prompt = invoke.call_args.args[1][0].content
    examples = [json.loads(line)["operation"] for line in prompt.splitlines() if line.startswith('{"operation"')]
    assert examples == ["find", "count_documents", "distinct", "aggregate"]
    assert "Never combine pipeline" in prompt


@pytest.mark.parametrize("mixed", [False, True])
def test_aggregate_preserves_configured_lookup_capabilities(monkeypatch, mixed):
    agent, invoke = capture(monkeypatch, "aggregate", mixed)
    agent.generate("Group orders", SCHEMA, operation_hint="aggregate")
    assert MongoCapabilities(mixed).lookup_guidance() in invoke.call_args.args[1][0].content


def learned_context():
    rules = [{"knowledge_id": "INTERNAL_RULE_ID", "source_request_id": "INTERNAL_REQUEST_ID",
              "source": "negative_feedback", "confidence": 0.99, "ranking_score": 0.85,
              "semantic_similarity": 0.65, "created_from_feedback": True,
              "type": "business_rule", "text": "Use processing history for failed orders.",
              "collections": ["orders_proc_hist"], "fields": ["status"], "mandatory": True,
              "operation": "find", "enforcement": {"required_filters": {"status": "failed"}},
              "metadata": {"value": "failed", "meaning": "Failure", "excluded_collections": ["orders"],
                           "source_request_id": "NESTED_INTERNAL_REQUEST", "confidence": 0.99}},
             {"knowledge_id": "SECOND_INTERNAL_ID", "text": "Use the newest execution date first.",
              "collections": ["orders_proc_hist"], "fields": ["executionDate"]}]
    preferences = [{"_id": "INTERNAL_PREFERENCE_ID", "confidence": 0.98, "type": "result_limit_preference",
                    "collection": "orders_proc_hist", "direction": -1, "limit": 5, "latest": True,
                    "condition": "Failed-order requests", "apply_mode": "contextual",
                    "preference": "Show the five most recent failed orders."}]
    return rules, preferences


@pytest.mark.parametrize("method", ["generate", "repair", "optimize"])
def test_all_generator_paths_receive_full_business_rules_without_ids_or_scores(monkeypatch, method):
    agent, invoke = capture(monkeypatch)
    rules, preferences = learned_context()
    original = deepcopy((rules, preferences))
    query = {"operation": "find", "collection": "orders_proc_hist", "filter": {"status": "failed"}}
    if method == "generate":
        agent.generate("Last failed order", SCHEMA, operation_hint="find", user_preferences=preferences,
                       database_knowledge=rules)
    elif method == "repair":
        agent.repair("Last failed order", SCHEMA, query, "Invalid projection", preferences, rules)
    else:
        agent.optimize("Last failed order", {"operation": "find"}, SCHEMA, query, "Return actionCode", preferences, rules)
    prompt = invoke.call_args.args[1][0].content
    assert "INTERNAL_" not in prompt
    assert '"knowledge_id"' not in prompt and '"source_request_id"' not in prompt
    assert '"confidence"' not in prompt and '"ranking_score"' not in prompt and '"semantic_similarity"' not in prompt
    assert prompt.index(rules[0]["text"]) < prompt.index(rules[1]["text"])
    assert all(name in prompt for name in ("orders_proc_hist", "executionDate", "excluded_collections", "required_filters"))
    assert preferences[0]["preference"] in prompt
    assert "Failed-order requests" in prompt
    assert (rules, preferences) == original
    if method != "optimize":
        assert "$lookup" not in prompt and "$group" not in prompt
    assert "MongoDB $lookup rules" not in prompt


def test_compaction_preserves_business_ids_and_native_filter_field_names():
    rule = {"knowledge_id": "REMOVE_ME", "text": "Match this document identifier", "collections": ["orders"],
            "enforcement": {"required_filters": {"_id": {"$oid": "507f1f77bcf86cd799439011"}}},
            "metadata": {"required_filters": {"_id": "business-record-id"}, "_id": "metadata-record-id"}}
    result = compact_knowledge([rule])[0]
    assert result["enforcement"] == rule["enforcement"]
    assert result["metadata"] == {"required_filters": {"_id": "business-record-id"}}
    assert "knowledge_id" not in result
    assert compact_preferences(None) == [] and compact_knowledge(None) == []


@pytest.mark.parametrize("operation", ["find", "count_documents", "distinct", "aggregate"])
def test_repair_shapes_follow_failed_query_operation(monkeypatch, operation):
    agent, invoke = capture(monkeypatch, operation)
    previous = response(operation)["template"]
    agent.repair("Show failed orders", SCHEMA, previous, "Invalid structure")
    prompt = invoke.call_args.args[1][0].content
    examples = [json.loads(line)["operation"] for line in prompt.splitlines() if line.startswith('{"operation"')]
    assert examples == [operation]
    if operation == "aggregate":
        assert "never drop its foreign pipeline" in prompt
        assert "Never combine pipeline" in prompt
    else:
        assert "never drop its foreign pipeline" not in prompt
        assert "Never combine pipeline" not in prompt


def test_aggregate_fragment_dollar_symbols_survive_once():
    example = next(line for line in operation_instructions("aggregate").splitlines() if line.startswith("{"))
    stages = json.loads(example)["pipeline"]
    assert stages[0]["$group"] == {"_id": "$<group_field>", "value": {"$sum": "$<metric_field>"}}
