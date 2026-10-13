import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from agents.query_context_agent import QueryContextAgent
from services.query_context_compactor import QueryContextCompactor, compact_json, count_tokens


def relationship(source, field, target, foreign_field):
    return {"from_collection": source, "from_field": field, "to_collection": target,
            "to_field": foreign_field, "source": "inferred", "confidence": 0.85}


def test_join_hints_are_deduplicated_validated_and_bounded_without_mutating_catalog():
    link = relationship("orders", "customerId", "customers", "_id")
    catalog = {"collections": {
        "orders": {"fields": {"customerId": ["ObjectId"], "total": ["number"]},
                   "relationships": [link, link, relationship("orders", "customerId", "missing", "_id"),
                                     relationship("orders", "hiddenField", "customers", "_id")]},
        "customers": {"fields": {"_id": ["ObjectId"], "name": ["string"]},
                      "relationships": [relationship("customers", "_id", "orders", "customerId")]},
    }}
    original = deepcopy(catalog)
    result = QueryContextCompactor().build("Orders and customer names", catalog, [], [])
    assert result["joins"] == [["orders", "customerId", "customers", "_id"]]
    assert all("relationships" not in entry for entry in result["schema"].values())
    assert catalog == original
    assert QueryContextCompactor(max_relationships=0).build("Orders", catalog, [], [])["joins"] == []


def test_knowledge_precedence_and_semantics_survive_but_bookkeeping_is_removed():
    catalog = {"collections": {"orders": {"fields": {"status": ["string"]}}}}
    rules = [{"knowledge_id": "private-id", "ranking_score": 0.9, "semantic_similarity": 0.8,
              "source_request_id": "private-request", "source": "negative_feedback",
              "text": "Only failed orders", "type": "business_rule", "collections": ["orders"],
              "fields": ["status"], "mandatory": True, "operation": "find",
              "enforcement": {"required_filters": {"status": "failed"}}},
             {"text": "Include successful orders", "collections": ["orders"], "fields": []}]
    preferences = [{"_id": "private-preference", "confidence": 0.95, "type": "sorting_preference",
                    "collection": "orders", "field": "status", "direction": -1,
                    "condition": "failed orders", "apply_mode": "contextual"}]
    original = deepcopy((rules, preferences))
    result = QueryContextCompactor().build("Orders", catalog, rules, preferences)
    assert result["knowledge"] == [rule["text"] for rule in rules]
    assert '"enforcement"' not in compact_json(result["knowledge"])
    assert result["preferences"][0]["direction"] == -1
    assert result["preferences"][0]["condition"] == "failed orders"
    assert "private-" not in compact_json(result)
    assert (rules, preferences) == original


def test_token_budget_retains_candidates_explicit_fields_and_required_rule_fields():
    fields = {f"irrelevant.deeply.nested.field{i}": ["string"] for i in range(100)}
    fields.update({"commandId": ["string"], "executionDate": ["date"], "status": ["string"]})
    catalog = {"collections": {"orders_proc_hist": {"fields": fields, "description": "History " * 150},
                               "orders": {"fields": deepcopy(fields), "aliases": ["orders " * 100]}}}
    rules = [{"text": "Use processing history", "collections": ["orders_proc_hist"], "fields": ["executionDate"]}]
    preferences = [{"collection": "orders_proc_hist", "field": "status", "direction": -1}]
    result = QueryContextCompactor(token_budget=240).build("Filter commandId", catalog, rules, preferences)
    assert result["telemetry"]["query_context_context_tokens_after"] <= 240
    assert not result["telemetry"]["query_context_budget_exceeded"]
    assert set(result["schema"]) == set(catalog["collections"])
    assert {"commandId", "executionDate", "status"} <= set(result["schema"]["orders_proc_hist"]["fields"])
    assert "commandId" in result["schema"]["orders"]["fields"]


def test_impossibly_small_budget_reports_overflow_without_cutting_rules_or_required_fields():
    catalog = {"collections": {"history": {"fields": {"commandId": ["string"], "status": ["string"]}}}}
    rules = [{"text": "A rule that must remain complete.", "collections": ["history"], "fields": ["status"]}]
    result = QueryContextCompactor(token_budget=1).build("Filter commandId", catalog, rules, [])
    assert result["telemetry"]["query_context_budget_exceeded"]
    assert result["knowledge"][0] == rules[0]["text"]
    assert set(result["schema"]["history"]["fields"]) == {"commandId", "status"}


def test_surviving_join_fields_are_preserved_under_budget_pressure():
    fields = {f"unused{i}": ["string"] for i in range(60)}
    catalog = {"collections": {
        "orders": {"fields": {**fields, "customerId": ["ObjectId"]},
                   "relationships": [relationship("orders", "customerId", "customers", "_id")]},
        "customers": {"fields": {**fields, "_id": ["ObjectId"]}},
    }}
    result = QueryContextCompactor(token_budget=180).build("Customer spending", catalog, [], [])
    assert result["joins"] == [["orders", "customerId", "customers", "_id"]]
    assert "customerId" in result["schema"]["orders"]["fields"]
    assert "_id" in result["schema"]["customers"]["fields"]
    assert result["telemetry"]["query_context_context_tokens_after"] <= 180


def test_enforced_filter_fields_are_preserved_even_when_not_explicitly_listed():
    catalog = {"collections": {"orders": {"fields": {"status": ["string"], "unused": ["string"]}}}}
    rules = [{"text": "Exclude cancelled", "collections": ["orders"], "fields": [],
              "enforcement": {"required_filters": {"status": {"$ne": "cancelled"}}}}]
    result = QueryContextCompactor(token_budget=1).build("Orders", catalog, rules, [])
    assert "status" in result["schema"]["orders"]["fields"]


def test_actual_llm_prompt_contains_compacted_context_and_metrics(monkeypatch):
    import agents.query_context_agent as module
    catalog = {"collections": {"orders": {"fields": {"commandId": ["string"]},
                               "sample_documents": [{"private": "NOT_FOR_PROMPTS"}],
                               "embedding": [0.1, 0.2]}}}
    rules = [{"knowledge_id": "VERBOSE_METADATA", "text": "Use orders", "collections": ["orders"], "fields": []}]
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"complexity": "simple", "operation": "find",
                 "collections": ["orders"], "fields": {"orders": ["commandId"]}})))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    result = QueryContextAgent().build("Show orders", catalog, database_knowledge=rules)
    prompt = invoke.call_args.args[1][0].content
    assert "VERBOSE_METADATA" not in prompt
    assert "NOT_FOR_PROMPTS" not in prompt
    assert "embedding" not in prompt
    assert "Use orders" in prompt
    assert result["_prompt_compaction"]["query_context_prompt_tokens_estimate"] == count_tokens(prompt)
    assert result["schema_spec"]["collections"]["orders"]["fields"] == {"commandId": ["string"]}


def test_user_text_matching_tokenizer_special_tokens_is_plain_data():
    assert count_tokens("<|endoftext|>") > 0


def test_only_rule_text_is_serialized_and_nested_field_leaf_names_are_protected():
    catalog = {"collections": {"orders": {"fields": {"customer.contact.email": ["string"],
                  "executionDate": ["date"], "unused": ["string"]}}}}
    rules = [{"text": "Use the supplied status meaning", "collections": ["orders"],
              "metadata": {"value": "X", "meaning": "cancelled", "excluded_collections": ["ordersHist"]}}]
    result = QueryContextCompactor(token_budget=1).build("Show email and execution date", catalog, rules, [])
    assert result["knowledge"] == [rules[0]["text"]]
    assert {"customer.contact.email", "executionDate"} <= set(result["schema"]["orders"]["fields"])
