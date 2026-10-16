"""Grounding regressions for the reported local-model outputs, without services."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.config import DEFAULT_USER_ID, SCHEMA_EMBEDDING_MODEL
from agent.output_schemas import GeneratedQuery
from agent.telemetry import request_context
from agents.query_generator_agent import QueryGeneratorAgent
from agents.validator_agent import ValidatorAgent
from database.query_template_repository import template_signature, EMBEDDING_REPRESENTATION_VERSION
from services.aggregation_validation import QueryValidationError
from services.mongo_capabilities import MongoCapabilities
from services.query.generation_examples import select_examples, generation_examples_text, remember_candidates
from services.query.prompt_selection.pruning import remember_prompt_view
from services.query_parameter_binder import template_from_concrete, bind_query_parameters
from services.template_parameter_extractor import describe_parameters


SCHEMA = {"collections": {"orders": {"fields": {"status": ["string"], "commandId": ["string"],
                                               "executionDate": ["string"]}}}}
QUESTION = "give me 3 passed orders"


def document(status_value="Passed", **changes):
    question = f"give me 5 {status_value} orders"
    generated = template_from_concrete({"operation": "find", "collection": "orders",
        "filter": {"status": status_value}, "limit": 5}, parameterize_limit=True)
    return {"template_id": "tpl-test", "template": generated["template"],
        "template_signature": template_signature(generated["template"]),
        "parameter_schema": describe_parameters(generated["template"], generated["parameters"], question),
        "operation": "find", "collections": ["orders"], "source_question": question,
        "semantic_text": "retrieve orders with a given status", "user_id": DEFAULT_USER_ID,
        "status": "active", "trusted": True, "success_count": 1, "positive_feedback_count": 1,
        "schema_version": "old-global-version", "embedding_model": SCHEMA_EMBEDDING_MODEL,
        "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION, **changes}


def test_literal_find_limit_is_parameterized_by_backend_without_another_model_call(monkeypatch):
    raw = {"template": {"operation": "find", "collection": "orders", "filter": {"status": "Passed"},
                        "limit": 3}, "parameters": {}}
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(raw)))
    monkeypatch.setattr("agents.query_generator_agent.llm_router.invoke", invoke)
    generated = QueryGeneratorAgent(MongoCapabilities()).generate(QUESTION, deepcopy(SCHEMA), operation_hint="find")
    assert generated["template"]["limit"] == {"$param": "limit"}
    assert generated["parameters"]["limit"] == 3
    assert bind_query_parameters(generated["template"], generated["parameters"]) == raw["template"]
    invoke.assert_called_once()
    prompt = invoke.call_args.args[1][0].content
    assert "parameters must be {}" in prompt and "Never invent IDs" in prompt


def test_provider_schema_enforces_literal_limits_and_empty_parameters_without_breaking_legacy_parsing():
    from agent.client_factory import schema_response_format
    schema = schema_response_format(GeneratedQuery)["json_schema"]["schema"]
    assert schema["properties"]["parameters"]["additionalProperties"] is False
    assert schema["properties"]["parameters"]["properties"] == {}
    assert schema["$defs"]["QueryTemplate"]["properties"]["limit"]["anyOf"] == [
        {"type": "integer", "minimum": 1}, {"type": "null"}]
    assert "LimitParameter" not in schema["$defs"]
    legacy = {"template": {"operation": "find", "collection": "orders", "filter": {"status": {"$param": "status"}},
                           "limit": {"$param": "limit"}}, "parameters": {"status": "Passed", "limit": 3}}
    parsed = GeneratedQuery.model_validate(legacy).model_dump(by_alias=True, exclude_unset=True)
    assert bind_query_parameters(parsed["template"], parsed["parameters"])["limit"] == 3
    # Building the outbound schema must not alter Pydantic's legacy contract.
    assert "LimitParameter" in GeneratedQuery.model_json_schema()["$defs"]


def test_reported_unbound_limit_still_fails_instead_of_silently_dropping_it():
    raw = {"template": {"operation": "find", "collection": "orders", "filter": {}, "limit": None},
           "parameters": {"limit": 3}}
    with pytest.raises(ValueError, match="Unexpected query parameters"):
        GeneratedQuery.model_validate(raw)


def test_stale_template_can_be_an_example_only_after_current_schema_validation():
    doc = document()
    original = deepcopy(doc)
    schema = {**deepcopy(SCHEMA), "schema_version": "new-global-version"}
    examples = select_examples(QUESTION, schema, "find", ranked=[(.9, doc)])
    assert len(examples) == 1
    assert examples[0]["output"]["template"]["filter"] == {"status": "Passed"}
    assert examples[0]["output"]["parameters"] == {}
    assert doc == original
    from services.semantic_template_cache import SemanticTemplateCache
    assert SemanticTemplateCache.compatibility_reason(doc, QUESTION, {"orders"}, "new-global-version") == "schema_version_mismatch"
    schema["collections"]["orders"]["fields"].pop("status")
    assert select_examples(QUESTION, schema, "find", ranked=[(.9, doc)]) == []


@pytest.mark.parametrize("change", [
    {"positive_feedback_count": 0}, {"success_count": 0}, {"trusted": False}, {"status": "invalid"},
    {"user_id": "different-user"}, {"embedding_model": "different-embedding-model"},
    {"embedding_representation_version": -1}, {"template_signature": "tampered"},
    {"collections": ["manual_recycle"]},
])
def test_unapproved_wrong_scope_or_tampered_templates_are_never_examples(change):
    assert select_examples(QUESTION, SCHEMA, "find", ranked=[(.99, document(**change))]) == []


def test_examples_are_ranked_deduplicated_bounded_and_never_force_three():
    docs = [document(status) for status in ("Passed", "Failed", "Pending", "Canceled")]
    # Different predicates have the same parameterized signature: store distinct
    # projection shapes to represent distinct approved query structures.
    for index, doc in enumerate(docs):
        doc["template"]["skip"] = index
        doc["template_signature"] = template_signature(doc["template"])
    ranked = [(.8 + index * .02, doc) for index, doc in enumerate(docs)]
    examples = select_examples(QUESTION, SCHEMA, "find", ranked=ranked + [ranked[-1]], max_examples=20)
    assert len(examples) == 3
    assert examples[0]["question"] == docs[-1]["source_question"]
    assert select_examples(QUESTION, SCHEMA, ranked=ranked, max_chars=20) == []
    assert select_examples(QUESTION, SCHEMA, ranked=[(.2, docs[0])]) == []
    assert select_examples(QUESTION, SCHEMA, ranked=[(float("nan"), docs[0])]) == []


def test_examples_do_not_leak_across_requests_questions_or_disabled_feature(monkeypatch):
    with request_context("request-one"):
        remember_candidates(QUESTION, [(.9, document())])
        assert "Approved historical" in generation_examples_text(QUESTION, SCHEMA, "find")
        assert generation_examples_text("a different question", SCHEMA) == ""
        with request_context("request-two"):
            assert generation_examples_text(QUESTION, SCHEMA) == ""
        monkeypatch.setenv("GENERATOR_APPROVED_EXAMPLES_ENABLED", "false")
        assert generation_examples_text(QUESTION, SCHEMA) == ""
    assert generation_examples_text(QUESTION, SCHEMA) == ""


def test_prompt_uses_retrieved_approved_examples_without_database_or_embedding_calls(monkeypatch):
    raw = {"template": {"operation": "find", "collection": "orders", "filter": {"status": "Passed"}, "limit": 3},
           "parameters": {}}
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(raw)))
    monkeypatch.setattr("agents.query_generator_agent.llm_router.invoke", invoke)
    with request_context("example-integration"):
        remember_candidates(QUESTION, [(.9, document())])
        QueryGeneratorAgent(MongoCapabilities()).generate(QUESTION, deepcopy(SCHEMA), operation_hint="find")
    prompt = invoke.call_args.args[1][0].content
    assert "give me 5 Passed orders" in prompt
    assert "never copy an unrequested example ID" in prompt
    assert "embedding_model" not in prompt and "positive_feedback_count" not in prompt
    invoke.assert_called_once()


@pytest.mark.parametrize("predicate", [
    {"$or": [{"$eq": ["status", "Passed"]}]},
    {"fields": {"$expr": {"$eq": ["orders.commandId", "$commandId"]}}},
    {"$and": []}, {"status": {"$in": "Passed"}},
])
@pytest.mark.parametrize("operation", ["find", "count_documents", "aggregate"])
def test_invented_query_structure_is_rejected_before_execution(predicate, operation):
    query = {"operation": operation, "collection": "orders", "filter": predicate}
    if operation == "aggregate":
        query = {"operation": operation, "collection": "orders", "pipeline": [{"$match": predicate}]}
    with pytest.raises(QueryValidationError) as caught:
        ValidatorAgent().validate(query, deepcopy(SCHEMA), profile_lookup=lambda _: None)
    assert caught.value.to_dict()["code"] == "invalid_predicate_shape"


def test_legitimate_expression_and_literal_embedded_document_remain_valid():
    from services.query.predicate_validation import validate_predicate
    query = {"operation": "find", "collection": "orders", "filter": {
        "$expr": {"$eq": ["$status", "Passed"]}, "status": {"nested": {"data": "$literal-data"}}}}
    validate_predicate(query["filter"])
    query["filter"].pop("status")
    ValidatorAgent().validate(query, deepcopy(SCHEMA), profile_lookup=lambda _: None)


def test_generator_cannot_escape_the_visible_collection_scope():
    schema = {"collections": {**deepcopy(SCHEMA["collections"]), "manual_recycle": {"fields": {"status": ["string"]}}}}
    remember_prompt_view(schema, SCHEMA)
    with pytest.raises(QueryValidationError) as caught:
        ValidatorAgent().validate({"operation": "find", "collection": "manual_recycle", "filter": {}}, schema,
                                  profile_lookup=lambda _: None)
    assert caught.value.to_dict()["code"] == "query_collection_scope"


def test_context_uses_approved_source_hints_and_restores_passed_status_field(monkeypatch):
    from agents.query_context_agent import QueryContextAgent
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"complexity": "simple", "operation": "find",
        "collections": ["orders"], "fields": {"orders": ["commandId"]}})))
    monkeypatch.setattr("agents.query_context_agent.llm_router.invoke", invoke)
    with request_context("source-hints"):
        remember_candidates(QUESTION, [(.9, document())])
        result = QueryContextAgent().build(QUESTION, deepcopy(SCHEMA))
    assert "status" in result["schema_spec"]["collections"]["orders"]["fields"]
    prompt = invoke.call_args.args[1][0].content
    assert "Similar approved historical source/field mappings" in prompt
    assert "give me 5 Passed orders" in prompt
    assert '"sources":{"orders":["status"]}' in prompt
    invoke.assert_called_once()


def test_approved_aggregation_example_checks_current_visible_join_fields():
    schema = {"collections": {**deepcopy(SCHEMA["collections"]),
                               "customers": {"fields": {"commandId": ["string"]}}}}
    doc = document()
    doc.update(operation="aggregate", collections=["orders", "customers"], parameter_schema={},
               template={"operation": "aggregate", "collection": "orders", "pipeline": [
                   {"$lookup": {"from": "customers", "localField": "commandId", "foreignField": "commandId", "as": "customer"}},
                   {"$match": {"status": "Passed"}}, {"$limit": 5}]})
    doc["template_signature"] = template_signature(doc["template"])
    assert select_examples(QUESTION, schema, "find", ranked=[(.9, doc)])
    schema["collections"]["customers"]["fields"].clear()
    assert select_examples(QUESTION, schema, "find", ranked=[(.9, doc)]) == []


def test_unsafe_string_date_sort_is_not_promoted_as_an_example():
    doc = document()
    doc["template"]["sort"] = [["executionDate", -1]]
    doc["template_signature"] = template_signature(doc["template"])
    profile = {"collection": "orders", "fields": {"status": {"types": {"string": 1}},
        "executionDate": {"types": {"string": 1}, "kind": "date", "date_format": "%d/%m/%Y %H:%M:%S",
                          "sortable_as_string": False}}}
    assert select_examples(QUESTION, SCHEMA, "find", ranked=[(.9, doc)], profile_lookup=lambda _: profile) == []


def test_constant_expression_and_regex_not_remain_valid():
    import re
    from services.query.predicate_validation import validate_predicate
    validate_predicate({"$expr": True, "status": {"$not": re.compile("Pending")}})
    with pytest.raises(QueryValidationError):
        validate_predicate({"status": {"$not": {"status": "Passed"}}})


def test_nested_foreign_match_cannot_hide_invalid_predicate_operators():
    schema = {"collections": {**deepcopy(SCHEMA["collections"]), "customers": {"fields": {"status": ["string"]}}}}
    query = {"operation": "aggregate", "collection": "orders", "pipeline": [
        {"$lookup": {"from": "customers", "as": "customers", "pipeline": [{"$match": {"$eq": ["status", "Passed"]}}]}}]}
    with pytest.raises(QueryValidationError) as caught:
        ValidatorAgent().validate(query, schema, profile_lookup=lambda _: None)
    assert caught.value.to_dict()["code"] == "invalid_predicate_shape"
