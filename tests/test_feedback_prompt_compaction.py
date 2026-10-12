"""Preserve classifier contracts while shrinking only its provider input."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agent.llm import LLMRouter
from agents.feedback_analysis_agent import FeedbackAnalysis, FeedbackAnalysisAgent
from agents.feedback_prompt import build_feedback_context, derive_scope_fields, feedback_output_contract
from services.query_history_service import QueryHistoryService


def output(scope="one_off", **changes):
    return {"feedback_type": "other", "cause": None, "corrected_meaning": None,
            "reusable_preference": scope == "user", "preference": None,
            "scope": scope, "confidence": 0.8, **changes}


@pytest.mark.parametrize("scope,knowledge,reusable,kind", [
    ("database", True, False, "business_rule"),
    ("user", False, True, None), ("one_off", False, False, None),
])
def test_scope_derived_fields_follow_original_rules(scope, knowledge, reusable, kind):
    raw = output(scope, knowledge_candidate=not knowledge, reusable_preference=True,
                 knowledge_type="business_rule")
    original = deepcopy(raw)
    normalized = derive_scope_fields(raw)
    assert raw == original
    assert normalized["knowledge_candidate"] is knowledge
    assert normalized["reusable_preference"] is reusable
    assert normalized["knowledge_type"] == kind
    parsed = feedback_output_contract().model_validate(raw)
    assert parsed.model_dump() == FeedbackAnalysis.model_validate(normalized).model_dump()


def test_user_reusability_still_requires_original_model_validation():
    with pytest.raises(ValidationError):
        feedback_output_contract().model_validate(output("user", reusable_preference=False))


@pytest.mark.parametrize("patch", [{"scope": "global"}, {"confidence": 1.5}, {"extra": "unknown"},
                                  {"preference": "x" * 501}, {"fields": ["a"] * 51}])
def test_api_adapter_does_not_relax_original_validation(patch):
    with pytest.raises(ValidationError):
        feedback_output_contract().model_validate({**output(), **patch})


def test_database_type_cannot_be_invented_by_code():
    with pytest.raises(ValidationError):
        feedback_output_contract().model_validate(output("database", knowledge_type=None))


def test_unknown_scope_and_wrong_json_shape_are_not_corrected():
    assert derive_scope_fields(["wrong"]) == ["wrong"]
    raw = output("unknown", knowledge_candidate=True)
    assert derive_scope_fields(raw) == raw


def test_api_schema_is_strict_without_changing_public_schema():
    original = deepcopy(FeedbackAnalysis.model_json_schema())
    response_format = feedback_output_contract().feedback_response_format()
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "knowledge_candidate" not in schema["properties"]
    assert {"type": "null"} in schema["properties"]["condition"]["anyOf"]
    assert FeedbackAnalysis.model_json_schema() == original


def test_derived_candidate_can_be_omitted_by_provider():
    parsed = feedback_output_contract().model_validate(output("database", knowledge_type="field_definition"))
    assert parsed.knowledge_candidate is True and parsed.reusable_preference is False


def test_compaction_removes_boilerplate_but_keeps_ordered_refinements():
    question = "Show recharge orders"
    effective = QueryHistoryService._corrected_question(
        QueryHistoryService._corrected_question(question, "Use orders_proc_hist."), "Only country = Tunisia.")
    history = {"question": question, "original_question": question, "effective_question": effective,
               "validated_query": {"collection": "orders_proc_hist", "filter": {"country": "Tunisia"}, "limit": 5},
               "generated_query": {"semantic_intent": "TEMPLATE_MARKER"}, "final_response": "SUMMARY_MARKER",
               "query_context": {"complexity": "complex", "operation": "aggregate"}}
    compact = build_feedback_context(history, "The actionCode must be Recharge.")
    assert compact["q"] == question
    assert compact["prior"] == ["Use orders_proc_hist.", "Only country = Tunisia."]
    assert compact["used"] == {"filter": {"country": "Tunisia"}, "limit": 5}
    rendered = json.dumps(compact)
    assert rendered.count(question) == 1
    for marker in ("Apply the latest correction", "Original question and any earlier corrections",
                   "TEMPLATE_MARKER", "SUMMARY_MARKER", "complexity", "operation"):
        assert marker not in rendered


def test_only_commented_fields_or_collections_send_field_names():
    history = {"question": "Show orders", "selected_schema": {
        "orders": {"fields": {"statusOrder": ["string"], "total": ["number"]}},
        "customers": {"fields": {"name": ["string"], "active": ["bool"]}}}}
    compact = build_feedback_context(history, "Use statusOrder.")
    assert compact["fields"] == {"orders": ["statusOrder"]}
    assert "fields" not in build_feedback_context(history, "Wrong answer.")
    compact = build_feedback_context(history, "Use customers instead.")
    assert compact["fields"] == {"customers": ["name", "active"]}


def test_aggregation_keeps_used_filters_sort_limits_and_join_collections():
    history = {"question": "Revenue", "validated_query": {"collection": "orders", "pipeline": [
        {"$match": {"statusOrder": "Done"}},
        {"$lookup": {"from": "customers", "pipeline": [{"$match": {"active": True}}], "as": "customer"}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}, {"$sort": {"total": -1}}, {"$limit": 5},
    ]}}
    compact = build_feedback_context(history, "Wrong revenue.")
    assert compact["cols"] == ["orders", "customers"]
    assert compact["used"] == {"match": [{"statusOrder": "Done"}, {"active": True}],
                               "sort_stages": [{"total": -1}], "limit_stages": [5]}
    assert "$group" not in json.dumps(compact)


def test_secret_redaction_is_retained(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-classifier-secret-value")
    compact = build_feedback_context({"question": "Show orders"}, "Wrong fake-classifier-secret-value")
    assert "fake-classifier-secret-value" not in json.dumps(compact)


def test_static_system_message_precedes_compact_user_context(monkeypatch):
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(output())))
    monkeypatch.setattr("agents.feedback_analysis_agent.llm_router.invoke", invoke)
    FeedbackAnalysisAgent().analyze({"question": "Show orders"}, "No")
    role, messages, _ = invoke.call_args.args
    assert role == "feedback_analysis"
    assert [message.type for message in messages] == ["system", "human"]
    assert json.loads(messages[1].content)["comment"] == "No"
    assert "$output_schema" not in messages[0].content and "$context_json" not in messages[0].content


def test_strict_format_is_bound_only_for_feedback_role(monkeypatch):
    client, bound = Mock(), Mock()
    client.bind.return_value = bound
    bound.invoke.return_value = SimpleNamespace(content=json.dumps(output()))
    client.invoke.return_value = SimpleNamespace(content='{"ok":true}')
    router = LLMRouter()
    monkeypatch.setattr(router, "_client", lambda model: client)
    router.invoke("feedback_analysis", [], feedback_output_contract())
    client.bind.assert_called_once_with(response_format=feedback_output_contract().feedback_response_format())
    bound.invoke.assert_called_once_with([])
    router.invoke("query_generator", [], dict)
    assert client.bind.call_count == 1
    client.invoke.assert_called_once_with([])


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_comment_shortcut_is_optional_and_off_by_default(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("FEEDBACK_SKIP_EMPTY_CLASSIFICATION", "true")
    else:
        monkeypatch.delenv("FEEDBACK_SKIP_EMPTY_CLASSIFICATION", raising=False)
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(output(confidence=0.0))))
    monkeypatch.setattr("agents.feedback_analysis_agent.llm_router.invoke", invoke)
    result = FeedbackAnalysisAgent().analyze({"question": "Show orders"}, "  \n ")
    assert result.scope == "one_off" and not result.knowledge_candidate
    assert result.feedback_type == "other" and result.confidence == 0.0
    assert invoke.call_count == (0 if enabled else 1)
