"""Profile-backed spelling and prompt ranking, with synthetic offline facts."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from services.profiling.accessors import publish_profiles
from services.query.profile_view import profile_to_generator_view
from services.query.values.matching import normalize_for_match
import services.filter_value_resolution as resolver

C, D = "svq_collection_821", "svq_collection_823"
F, OTHER, IDENT, PII, EXCLUDED = ("svq_topic_827", "svq_other_829", "svq_key_839", "svq_hidden_853", "svq_skip_857")
STORED, CASE, HYPHEN = "assemble_widget", "Assemble_Widget", "assemble-widget"
SECRET = "svq_secret_859"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import services.profiling.accessors as accessors
    import services.value_catalog as catalog
    import agent.config as config
    from agent.llm import llm_router
    monkeypatch.setattr(accessors, "_profiles", {})
    monkeypatch.setattr(config, "MONGODB_DATABASE", "svq_database_863")
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    monkeypatch.setattr(resolver, "VALUE_SEPARATOR_MATCHING_ENABLED", True)
    monkeypatch.setattr(catalog, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_VARIANTS_POLICY", "merge")
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset())
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_SUGGESTION_CUTOFF", .8)
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "true")
    monkeypatch.setenv("GENERATOR_FACTS_RELEVANCE_FIRST", "true")
    monkeypatch.setenv("GENERATOR_FACTS_MAX_VALUES_PER_FIELD", "15")
    monkeypatch.setenv("GENERATOR_FACTS_MAX_CHARS", "2400")
    monkeypatch.setenv("DATE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setattr(llm_router, "invoke", Mock(side_effect=AssertionError("No model calls")))
    monkeypatch.setattr(catalog._catalog, "database_provider", Mock(side_effect=AssertionError("No live samples")))


def categorical(values):
    return {"kind": "categorical", "types": {"string": 1}, "values": [
        {"value": value, "count": index + 1} for index, value in enumerate(values)]}


def profiles(values=(STORED,)):
    return {C: {"collection": C, "config": {"exclude_fields": [EXCLUDED]}, "fields": {
        OTHER: categorical(["svq_unrelated_877"]), F: categorical(values),
        IDENT: {**categorical([SECRET]), "kind": "identifier"},
        PII: {**categorical([SECRET]), "pii_suspected": True}, EXCLUDED: categorical([SECRET])}},
        D: {"collection": D, "fields": {F: categorical(values)}}}


def publish(facts):
    from agent.config import MONGODB_DATABASE
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in facts.items()})


def schema(facts):
    return {"collections": {name: {"fields": {path: ["string"] for path in profile["fields"]}}
                            for name, profile in facts.items()}}


def resolve(value, rules=()):
    return resolver.resolve_filter_value(C, F, value, rules)


def apply(condition, facts, rules=()):
    original = {"operation": "find", "collection": C, "filter": {F: condition}}
    result = resolver.apply_value_resolution(original, schema(facts), rules)
    assert resolver.apply_value_resolution(result, schema(facts), rules) == result
    return result["filter"][F]


@pytest.mark.parametrize("text", ["assemble widget", "assemble-widget", "AssembleWidget", "ASSEMBLE_WIDGET",
                                 "assemble.widget", "assemble/widget", "  assemble__- /widget  ",
                                 "ＡｓｓｅｍｂｌｅＷｉｄｇｅｔ", "ássémble_wídget"])
def test_normalization_and_resolution(text, caplog):
    assert normalize_for_match(text) == ("assemble widget", ["assemble", "widget"])
    facts = profiles()
    publish(facts)
    assert apply(text, facts) == STORED
    if text not in {"ASSEMBLE_WIDGET", "ássémble_wídget"}:
        assert '"reason":"separator_variant"' in caplog.text


@pytest.mark.parametrize("text,expected", [("XMLParser42", "xml parser 42"), ("item42Code", "item 42 code"),
                                           ("", ""), ("\t_/.- ", ""), ("ÉclairWidget", "eclair widget")])
def test_generic_boundaries(text, expected):
    key, tokens = normalize_for_match(text)
    assert key == expected and tokens == expected.split()


def test_ambiguous_layout_only_in_fallback(caplog):
    publish(profiles([STORED, HYPHEN]))
    assert resolve("AssembleWidget") == (["AssembleWidget"], "ambiguous_separator_variant")
    assert '"reason":"ambiguous_separator_variant"' in caplog.text
    # The existing exact/case/accent path still wins without separator ambiguity.
    assert resolve(HYPHEN) == ([HYPHEN], "exact")
    assert resolve("ASSEMBLE_WIDGET") == ([STORED], "case_variant")


@pytest.mark.parametrize("policy", ["merge", "exact"])
def test_case_variant_policy_is_preserved(monkeypatch, policy):
    facts = profiles([CASE, STORED])
    publish(facts)
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_VARIANTS_POLICY", policy)
    assert resolve("AssembleWidget") == ([CASE, STORED], "separator_variant")
    assert resolve(CASE)[0] == ([CASE] if policy == "exact" else [CASE, STORED])
    assert apply("AssembleWidget", facts) == {"$in": [CASE, STORED]}


@pytest.mark.parametrize("text", ["assemble", "widget", "assemble widge", "assemble widget extra", "different"])
def test_partial_matches_only_suggest(text, caplog):
    publish(profiles([STORED, "disassemble_widget", "assemble_gadget"]))
    assert resolve(text) == ([text], "not_in_catalog_closest")
    assert '"reason":"not_in_catalog_closest"' in caplog.text
    if text == "assemble widge":
        assert '"closest_stored_value":"assemble_widget"' in caplog.text


def test_closest_respects_existing_cutoff(monkeypatch, caplog):
    publish(profiles())
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_SUGGESTION_CUTOFF", 1)
    assert resolve("assemble widge")[0] == ["assemble widge"]
    assert '"closest_stored_value"' not in caplog.text


@pytest.mark.parametrize("operator", [None, "$eq", "$ne", "$in", "$nin", "$all"])
def test_all_existing_operators(operator):
    facts = profiles()
    publish(facts)
    value = ["AssembleWidget"] if operator in {"$in", "$nin", "$all"} else "AssembleWidget"
    condition = {operator: value} if operator else value
    expected_value = [STORED] if isinstance(value, list) else STORED
    assert apply(condition, facts) == ({operator: expected_value} if operator else expected_value)


def test_nested_lookup_facet_and_alias():
    facts = profiles()
    publish(facts)
    query = {"operation": "aggregate", "collection": C, "pipeline": [{"$lookup": {
        "from": D, "as": "svq_join_881", "pipeline": [{"$match": {F: "AssembleWidget"}},
        {"$facet": {"svq_branch_883": [{"$match": {F: {"$all": ["assemble-widget"]}}}]}}]}},
        {"$match": {"svq_join_881." + F: {"$nin": ["assemble widget"]}}}]}
    before = deepcopy(query)
    result = resolver.apply_value_resolution(query, schema(facts), [])
    assert query == before
    assert result["pipeline"][0]["$lookup"]["pipeline"][0]["$match"][F] == STORED
    assert result["pipeline"][0]["$lookup"]["pipeline"][1]["$facet"]["svq_branch_883"][0]["$match"][F] == {"$all": [STORED]}
    assert result["pipeline"][1]["$match"]["svq_join_881." + F] == {"$nin": [STORED]}
    assert resolver.apply_value_resolution(result, schema(facts), []) == result


@pytest.mark.parametrize("field", [IDENT, PII, EXCLUDED])
def test_protected_fields_never_correct_or_log_values(field, caplog):
    facts = profiles()
    facts[C]["fields"][field]["values"] = [{"value": "svq_sensitive_spelling_887"}]
    publish(facts)
    value = "svq sensitive spelling 887"
    assert resolver.resolve_filter_value(C, field, value, []) == ([value], "not_categorical")
    assert "svq_sensitive_spelling_887" not in caplog.text and value not in caplog.text


def test_case_sensitive_and_rule_guard(monkeypatch, caplog):
    publish(profiles([CASE, STORED]))
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset({C + "." + F}))
    assert resolve("AssembleWidget") == (["AssembleWidget"], "case_sensitive")
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset())
    rule = {"active": True, "status": "active", "type": "coded_value", "fields": [F],
            "collections": [C], "text": CASE + " differs from " + STORED}
    assert resolve("AssembleWidget", [rule]) == (["AssembleWidget"], "rule_guard")
    assert '"reason":"rule_guard"' in caplog.text


def test_flag_off_and_no_profile_keep_legacy(monkeypatch, caplog):
    publish(profiles())
    monkeypatch.setattr(resolver, "VALUE_SEPARATOR_MATCHING_ENABLED", False)
    assert resolve("AssembleWidget") == (["AssembleWidget"], "not_in_catalog")
    assert resolve("ASSEMBLE_WIDGET") == ([STORED], "case_variant")
    monkeypatch.setattr(resolver, "VALUE_SEPARATOR_MATCHING_ENABLED", True)
    publish({})
    # Legacy catalog retrieval is faked so no profile never triggers a live read.
    monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: {STORED})
    assert resolve("AssembleWidget") == (["AssembleWidget"], "not_in_catalog")
    assert '"reason":"separator_variant"' not in caplog.text


def facts_view(facts, question):
    selected = schema(facts)
    selected["collections"].pop(D)
    return profile_to_generator_view(selected, question, facts.get)


def shown_values(view, field=F):
    row = next(line for line in view.splitlines() if line.startswith(C + "." + field + " "))
    return json.JSONDecoder().raw_decode(row.split("values=", 1)[1])[0]


def test_forty_values_relevant_tail_and_field_first(monkeypatch):
    values = [f"svq_option_{index:02}" for index in range(40)]
    facts = profiles(values)
    facts[C]["fields"][F]["values"][-1]["count"] = 0
    view = facts_view(facts, "Find SvqOption39")
    shown = shown_values(view)
    assert shown[0] == values[-1] and len(shown) == 15
    assert "(+25 more values not shown)" in view
    assert view.splitlines()[2].startswith(C + "." + F + " ")
    assert "count=" not in view and SECRET not in view and PII not in view and EXCLUDED not in view
    monkeypatch.setenv("GENERATOR_FACTS_RELEVANCE_FIRST", "false")
    old = facts_view(facts, "Find SvqOption39")
    assert shown_values(old) == values[:15]
    assert "more values not shown" not in old
    assert old.splitlines()[2].startswith(C + "." + OTHER + " ")


def test_ranking_ties_and_partial_scores(monkeypatch):
    facts = profiles()
    facts[C]["fields"][F]["values"] = [
        {"value": "widget_delta", "count": 9}, {"value": "unrelated_alpha", "count": 500},
        {"value": "widget_beta", "count": 9}, {"value": "widget_gamma", "count": 12},
        {"value": "widget_alpha", "count": 1}]
    monkeypatch.setenv("GENERATOR_FACTS_MAX_VALUES_PER_FIELD", "4")
    assert shown_values(facts_view(facts, "widget alpha")) == [
        "widget_alpha", "unrelated_alpha", "widget_gamma", "widget_beta"]
    # With equal scores: count descending then stored spelling alphabetically.
    assert shown_values(facts_view(facts, "widget")) == [
        "widget_gamma", "widget_beta", "widget_delta", "widget_alpha"]


def test_below_cap_order_caps_missing_profile_and_short_question_tokens(monkeypatch):
    facts = profiles(["widget_beta", "widget_alpha"])
    assert shown_values(facts_view(facts, "widget alpha")) == ["widget_beta", "widget_alpha"]
    facts[C]["fields"][OTHER] = categorical(["svq_other_889"])
    facts[C]["fields"][F] = categorical(["x_y"])
    assert facts_view(facts, "x y").splitlines()[2].startswith(C + "." + OTHER + " ")
    monkeypatch.setenv("GENERATOR_FACTS_MAX_CHARS", "150")
    assert len(facts_view(facts, "x y")) <= 150
    monkeypatch.setenv("GENERATOR_FACTS_MAX_VALUES_PER_FIELD", "0")
    assert len(facts_view(facts, "x y")) <= 150
    assert profile_to_generator_view(schema(facts), "widget", lambda _: None) == ""


def test_unselected_field_matching_value_in_selected_collection_is_prioritized():
    facts = profiles()
    selected = {"collections": {C: {"fields": {OTHER: ["string"]}}}}
    view = profile_to_generator_view(selected, "AssembleWidget", facts.get)
    assert view.splitlines()[2].startswith(C + "." + F + " ")
    assert not any(line.startswith(D + ".") for line in view.splitlines())


def test_no_fixture_literals_in_new_module():
    path = Path(__file__).resolve().parents[1] / "services/query/values/matching.py"
    literals = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    fixtures = {C, D, F, OTHER, IDENT, PII, EXCLUDED, STORED, CASE, HYPHEN, SECRET,
                "widget_alpha", "widget_beta", "widget_gamma", "widget_delta", "unrelated_alpha"}
    assert not fixtures & literals
