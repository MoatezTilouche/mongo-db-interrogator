"""Prompt-boundary regressions for distracting sources, values and row counts."""
from copy import deepcopy
import pytest

from services.query.collection_scope import explicit_collection_scope
from services.query_context_compactor import QueryContextCompactor
from services.query.prompt_selection.builder import build_prompt_inputs
from services.query.prompt_selection.config import PromptSelectionConfig
from services.query.prompt_selection.relevance import compute_question_relevance


NAMES = ["orders", "orders_proc_hist", "manual_recycle", "customers"]


@pytest.mark.parametrize("question,expected", [
    ("give me 3 orders from orders_proc_hist", ["orders_proc_hist"]),
    ("show records from `orders_proc_hist`", ["orders_proc_hist"]),
    ("show records from orders", ["orders"]),
    ("show orders in progress", []),
    ("show orders from orders_proc_hist_archive", []),
    ("show orders not from orders_proc_hist", []),
    ("join orders from orders_proc_hist with customers", []),
    ("show orders and customer names from orders_proc_hist", []),
    ("show orders with customer names from orders_proc_hist", []),
])
def test_only_unambiguous_exact_sources_narrow_context(question, expected):
    assert explicit_collection_scope(question, NAMES) == expected


def catalog():
    return {"collections": {
        "orders_proc_hist": {"fields": {"commandId": ["string"], "executionDate": ["string"], "status": ["string"]}},
        "manual_recycle": {"fields": {f"nested.unrelated{i}": ["string"] for i in range(350)}},
    }}


def rules():
    return [{"text": "Latest record per commandId by executionDate.", "collections": ["orders_proc_hist"],
             "fields": ["commandId", "executionDate"]},
            {"text": "In-progress orders use manual_recycle.", "collections": ["manual_recycle"], "fields": []}]


def test_explicit_context_drops_distracting_sources_but_keeps_required_rule_fields():
    source, knowledge = catalog(), rules()
    original = deepcopy((source, knowledge))
    result = QueryContextCompactor().build("give me 3 orders from orders_proc_hist", source, knowledge, [])
    assert list(result["schema"]) == ["orders_proc_hist"]
    assert result["knowledge"] == [knowledge[0]["text"]]
    assert {"commandId", "executionDate"} <= set(result["schema"]["orders_proc_hist"]["fields"])
    assert (source, knowledge) == original


def test_large_restored_collection_is_capped_without_dropping_candidates_or_evidence():
    source = catalog()
    knowledge = [{"text": "Use this source", "collections": ["manual_recycle"], "fields": ["nested.unrelated349"]}]
    result = QueryContextCompactor(max_fields=8).build("show orders", source, knowledge, [])
    assert set(result["schema"]) == set(source["collections"])
    assert len(result["schema"]["manual_recycle"]["fields"]) == 8
    assert "nested.unrelated349" in result["schema"]["manual_recycle"]["fields"]


def test_suggestions_cannot_reintroduce_explicitly_unrequested_sources(monkeypatch):
    from services.knowledge import collection_suggestions as suggestions
    monkeypatch.setattr(suggestions, "KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED", True)
    context = {"collections": ["orders_proc_hist"], "fields": {"orders_proc_hist": ["commandId"]}}
    source = catalog()
    knowledge = [{**rule, "type": "business_rule"} for rule in rules()]
    updated, schema, added = suggestions.apply_rule_suggestions(
        context, {"collections": {"orders_proc_hist": source["collections"]["orders_proc_hist"]}},
        knowledge, source, question="give me 3 orders from orders_proc_hist")
    assert not added
    assert updated["collections"] == ["orders_proc_hist"]
    assert list(schema["collections"]) == ["orders_proc_hist"]


def test_generator_scope_is_applied_even_without_profiles():
    source = catalog()
    original = deepcopy(source)
    result = build_prompt_inputs(source, "give me 3 orders from orders_proc_hist", rules(),
                                 profile_lookup=lambda _: None, config=PromptSelectionConfig())
    assert list(result["schema"]) == ["orders_proc_hist"]
    assert result["rules"] == rules()[:1]
    assert source == original


def profile():
    return {"collection": "manual_recycle", "fields": {
        "status": {"kind": "categorical", "types": {"string": 10}, "presence": 1,
                   "values": [{"value": "Passed", "count": 5}, {"value": "Failed", "count": 5}]},
        "sequence": {"kind": "categorical", "types": {"string": 10}, "presence": 1,
                     "values": [{"value": "1", "count": 3}, {"value": "2", "count": 3}, {"value": "3", "count": 4}]},
    }}


@pytest.mark.parametrize("question,matched", [("give me 3 passed orders", False), ("orders with sequence 3", True),
                                            ("give me 3 orders with sequence 3", True)])
def test_requested_count_is_not_a_categorical_value_signal(question, matched):
    relevance = compute_question_relevance(question, ["manual_recycle"], lambda _: profile())
    sequence = next(row for row in relevance["manual_recycle"] if row["field"] == "sequence")
    assert ("value" in sequence["reasons"]) is matched


def test_facts_only_describe_relevant_fields_visible_in_schema(monkeypatch):
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "true")
    source = {"collections": {"manual_recycle": {"fields": {"status": ["string"], "sequence": ["string"]}}}}
    result = build_prompt_inputs(source, "give me 3 Passed orders", profile_lookup=lambda _: profile(),
                                 config=PromptSelectionConfig())
    assert "manual_recycle.status" in result["facts"]
    assert "manual_recycle.sequence" not in result["facts"]
    source["collections"]["manual_recycle"]["fields"].pop("status")
    result = build_prompt_inputs(source, "Passed orders", profile_lookup=lambda _: profile(), config=PromptSelectionConfig())
    assert "manual_recycle.status" not in result["facts"]


def test_give_me_count_overrides_stored_single_result_preference():
    from services.preferences.question_intent import explicit_result_limit
    from services.preferences.application import apply_preferences_to_query
    assert explicit_result_limit("give me 3 orders from orders_proc_hist") == {"limit": 3, "latest": False}
    query = {"operation": "find", "collection": "orders_proc_hist", "filter": {}, "limit": 1}
    apply_preferences_to_query(query, [{"type": "result_limit_preference", "collection": "orders_proc_hist", "limit": 1}],
                               question="give me 3 orders from orders_proc_hist")
    assert query["limit"] == 3
