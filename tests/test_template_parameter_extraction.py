"""Generic typed slot extraction with fake models and no database calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.template_parameter_extractor as extractor
from services.query_parameter_binder import parameterize_filter_values, bind_query_parameters

SHORTHAND_CASE = json.loads((Path(__file__).parent / "fixtures/template_parameter_shorthand_case.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    invoke = Mock(side_effect=AssertionError("Tests must explicitly fake the extraction fallback"))
    monkeypatch.setattr(extractor.llm_router, "invoke", invoke)
    return invoke


def saved(source, filters, limit=1):
    generated = parameterize_filter_values({"template": {"operation": "find", "collection": "records",
        "filter": filters, "sort": [["sequence", -1]], "limit": {"$param": "limit"}, "skip": 0},
        "parameters": {"limit": limit}})
    return {"source_question": source, "template": generated["template"],
            "parameter_schema": extractor.describe_parameters(generated["template"], generated["parameters"], source)}


def decline(monkeypatch):
    invoke = Mock(return_value=SimpleNamespace(content='{"can_extract":false,"parameters":null}'))
    monkeypatch.setattr(extractor.llm_router, "invoke", invoke)
    return invoke


def test_reported_template_changes_both_filter_values_and_preserves_legacy_implicit_limit(offline):
    source = "give me the order that have actionCode = Preactivated_SIM of this commandId CMD-000010284552"
    question = "give me the order that have actionCode = customer_check of this commandId CMD-000012825058"
    doc = saved(source, {"actionCode": "Preactivated_SIM", "commandId": "CMD-000010284552"})
    doc["parameter_schema"]["limit"].update(user_variable=True, source_visible=True)
    original = deepcopy(doc)
    parameters, method = extractor.extract_parameters(question, doc)
    assert method == "deterministic"
    assert bind_query_parameters(doc["template"], parameters)["filter"] == {
        "actionCode": "customer_check", "commandId": "CMD-000012825058"}
    assert parameters["limit"] == 1
    assert doc == original
    offline.assert_not_called()


@pytest.mark.parametrize("field,old,new", [
    ("processingMode", "Preactivated_SIM", "customer_check"),
    ("category", "old-value", "new-value"),
    ("accountKey", "PREFIX_123", "OTHER_789"),
    ("state", "Pending", "Failed"),
    ("contact", "old@example.org", "new@example.net"),
    ("region", "Québec", "Montréal"),
])
def test_arbitrary_string_fields_have_no_named_field_or_prefix_exception(field, old, new, offline):
    source = f"Find records where {field} = {old}"
    doc = saved(source, {field: old})
    values, method = extractor.extract_parameters(f"Find records where {field} = {new}", doc)
    assert method == "deterministic"
    assert bind_query_parameters(doc["template"], values)["filter"] == {field: new}
    offline.assert_not_called()


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_multiword_values_remain_bounded_by_original_quotes(quote, offline):
    doc = saved(f"Find records for {quote}North Region{quote}", {"regionLabel": "North Region"})
    values, method = extractor.extract_parameters(f"Find records for {quote}South West Region{quote}", doc)
    assert method == "deterministic"
    assert bind_query_parameters(doc["template"], values)["filter"] == {"regionLabel": "South West Region"}
    offline.assert_not_called()


def test_empty_quoted_string_has_one_bounded_slot(offline):
    doc = saved('Find records where label = ""', {"label": ""})
    values, method = extractor.extract_parameters('Find records where label = "New Label"', doc)
    assert method == "deterministic"
    assert bind_query_parameters(doc["template"], values)["filter"] == {"label": "New Label"}
    offline.assert_not_called()


@pytest.mark.parametrize("limit", [1, 7, 25])
def test_unmentioned_structural_values_are_marked_constant_without_value_specific_logic(limit):
    doc = saved("Find records for code ACTIVE", {"code": "ACTIVE"}, limit=limit)
    spec = doc["parameter_schema"]["limit"]
    assert spec["example"] == limit
    assert spec["source_visible"] is False
    assert spec["user_variable"] is False


def test_defaults_are_not_inferred_from_digits_inside_identifiers(offline):
    doc = saved("Find records for key ABC-1", {"key": "ABC-1"})
    assert doc["parameter_schema"]["limit"]["user_variable"] is False
    values, method = extractor.extract_parameters("Find records for key ABC-2", doc)
    assert method == "deterministic"
    assert values["limit"] == 1
    assert bind_query_parameters(doc["template"], values)["filter"]["key"] == "ABC-2"


@pytest.mark.parametrize("literal,limit", [("1/1/2026", 1), ("10:30", 10), ("1.5", 1)])
def test_defaults_are_not_inferred_from_components_of_other_literals(literal, limit):
    doc = saved(f"Find records for value {literal}", {"value": literal}, limit=limit)
    assert doc["parameter_schema"]["limit"]["source_visible"] is False


def test_explicit_numeric_quantity_remains_variable(offline):
    doc = saved("Show 7 records with code ACTIVE", {"code": "ACTIVE"}, limit=7)
    assert doc["parameter_schema"]["limit"]["user_variable"] is True
    values, method = extractor.extract_parameters("Show 12 records with code PAUSED", doc)
    assert method == "deterministic"
    assert values["limit"] == 12
    assert bind_query_parameters(doc["template"], values)["filter"]["code"] == "PAUSED"


def test_explicit_number_before_sentence_punctuation_remains_variable():
    doc = saved("Show records with limit 7.", {}, limit=7)
    values, method = extractor.extract_parameters("Show records with limit 12.", doc)
    assert method == "deterministic"
    assert values["limit"] == 12


@pytest.mark.parametrize("question", [
    "Find records where code = ACTIVE and price > 10",
    "Find records where code = ACTIVE sorted by price",
    "Find 10 records where code = ACTIVE",
    "Find records where otherField = ACTIVE",
    "Delete records where code = ACTIVE",
])
def test_changed_question_structure_is_not_accepted_as_a_string_value(monkeypatch, question):
    doc = saved("Find records where code = ACTIVE", {"code": "ACTIVE"})
    invoke = decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError) as failure:
        extractor.extract_parameters(question, doc)
    assert failure.value.reason == "unsafe_or_missing_parameter"
    invoke.assert_called_once()


def test_fallback_receives_corrected_legacy_flags_but_cannot_change_fixed_default(monkeypatch):
    doc = saved("Find records with code ACTIVE", {"code": "ACTIVE"}, limit=7)
    doc["parameter_schema"]["limit"].update(source_visible=True, user_variable=True)
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"can_extract": True, "parameters": {
        "filter_code": "PAUSED", "limit": 8}})))
    monkeypatch.setattr(extractor.llm_router, "invoke", invoke)
    with pytest.raises(extractor.TemplateParameterExtractionError) as failure:
        extractor.extract_parameters("Records with code PAUSED please", doc)
    assert failure.value.reason == "fixed_parameter_changed"
    payload = json.loads(invoke.call_args.args[1][0].content.split("\n", 1)[1])
    assert payload["parameter_schema"]["limit"]["user_variable"] is False
    assert doc["parameter_schema"]["limit"]["user_variable"] is True


def test_repeated_parameter_must_have_same_new_value():
    doc = saved("Find records for code ACTIVE, again ACTIVE", {"code": "ACTIVE"})
    with pytest.raises(extractor.TemplateParameterExtractionError) as failure:
        extractor.extract_parameters("Find records for code PAUSED, again CLOSED", doc)
    assert failure.value.reason == "ambiguous_value"


def test_shared_examples_for_different_parameters_require_fallback(monkeypatch):
    doc = saved("Find records where left = ACTIVE and right = ACTIVE", {"left": "ACTIVE", "right": "ACTIVE"})
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError) as failure:
        extractor.extract_parameters("Find records where left = PAUSED and right = CLOSED", doc)
    assert failure.value.reason == "ambiguous_parameter_positions"


def test_punctuation_between_adjacent_string_slots_does_not_choose_an_arbitrary_split(monkeypatch):
    doc = saved("Find records for left-right", {"left": "left", "right": "right"})
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError) as failure:
        extractor.extract_parameters("Find records for new-left-right", doc)
    assert failure.value.reason == "ambiguous_parameter_positions"


def test_boolean_values_are_typed_data(offline):
    doc = saved("Find records where enabled = true", {"enabled": True})
    values, method = extractor.extract_parameters("Find records where enabled = false", doc)
    assert method == "deterministic"
    assert bind_query_parameters(doc["template"], values)["filter"]["enabled"] is False


def test_unquoted_multiword_changes_keep_existing_typed_fallback(monkeypatch):
    doc = saved("Find records for North Region", {"region": "North Region"})
    invoke = decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters("Find records for South Region", doc)
    invoke.assert_called_once()


def test_fixed_filter_values_are_not_rebound(monkeypatch):
    doc = saved("Find records with code ACTIVE", {"code": "ACTIVE"})
    doc["parameter_schema"]["filter_code"]["user_variable"] = False
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters("Find records with code PAUSED", doc)


@pytest.mark.parametrize("case", SHORTHAND_CASE["accepted"])
def test_reported_reordered_shorthand_extracts_without_model_or_structure_changes(case, offline):
    doc = deepcopy(SHORTHAND_CASE["document"])
    before = deepcopy(doc)
    values, method = extractor.extract_parameters(case["question"], doc)
    assert method == "deterministic"
    assert values == {"limit": case["limit"], "filter_and_0_status": "Passed"}
    bound = bind_query_parameters(doc["template"], values)
    assert bound["filter"]["$and"][0] == {"status": "Passed"}
    assert bound["filter"]["$and"][1] == doc["template"]["filter"]["$and"][1]
    assert bound["sort"] == [] and bound["projection"] is None
    assert doc == before
    offline.assert_not_called()


@pytest.mark.parametrize("question", SHORTHAND_CASE["declined"])
def test_shorthand_does_not_guess_missing_values_or_accept_changed_constraints(question, monkeypatch):
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters(question, deepcopy(SHORTHAND_CASE["document"]))


def test_shorthand_is_generic_and_preserves_fixed_parameters(offline):
    doc = saved("Show 7 ready records today from records with state = Ready",
                {"state": "Ready", "category": "Fixed"}, limit=7)
    values, method = extractor.extract_parameters("Show 11 records ready today", doc)
    assert values == {"limit": 11, "filter_state": "Ready", "filter_category": "Fixed"} and method == "deterministic"
    offline.assert_not_called()


def test_shorthand_respects_refined_scope(monkeypatch):
    doc = deepcopy(SHORTHAND_CASE["document"])
    doc["scope_refined"] = True
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters(SHORTHAND_CASE["accepted"][0]["question"], doc)


def test_shorthand_does_not_ignore_case_changes_to_identifier_like_values(monkeypatch):
    doc = saved("Show 7 ABC records today from records with code = ABC", {"code": "ABC"}, limit=7)
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters("Show 11 records abc today", doc)


@pytest.mark.parametrize("annotation", ["code = abc", "Code = ABC"])
def test_shorthand_does_not_hide_changed_explicit_value_or_field_case(monkeypatch, annotation):
    doc = saved("Show 7 ABC records today from records with code = ABC", {"code": "ABC"}, limit=7)
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters("Show 11 records ABC today from records with " + annotation, doc)


def test_shorthand_does_not_rebind_shared_quantity_filter_slot(monkeypatch):
    doc = deepcopy(SHORTHAND_CASE["document"])
    doc["parameter_schema"]["limit"]["source_paths"].append("filter.sequence")
    decline(monkeypatch)
    with pytest.raises(extractor.TemplateParameterExtractionError):
        extractor.extract_parameters(SHORTHAND_CASE["accepted"][0]["question"], doc)


def test_fallback_prompt_distinguishes_redundant_annotations_from_removed_constraints(monkeypatch):
    doc = deepcopy(SHORTHAND_CASE["document"])
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"can_extract": True,
        "parameters": {"limit": 4, "filter_and_0_status": "Passed"}})))
    monkeypatch.setattr(extractor.llm_router, "invoke", invoke)
    values, method = extractor.extract_parameters("Please retrieve four orders that passed today", doc)
    assert values["limit"] == 4 and method == "llm"
    prompt, payload = invoke.call_args.args[1][0].content.split("\n", 1)
    assert "omitting its name alone does not change scope" in prompt
    assert "redundant field=value" in prompt and "Ground every user-variable value" in prompt
    assert json.loads(payload)["template_context"]["scope_refined"] is False
    assert invoke.call_args.args[0] == "template_parameters"
