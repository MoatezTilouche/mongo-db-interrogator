"""Synthetic prompt budgets; every external boundary is forbidden."""
import ast
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.query.prompt_selection.config import PromptSelectionConfig
from services.query.prompt_selection.builder import build_prompt_inputs, prepare_relevance
from services.query.prompt_selection.relevance import compute_question_relevance, generic_tokens
from services.query.prompt_selection.pruning import prune_schema_for_prompt
from services.query.prompt_selection.gates import gate_rules, collection_relevance

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, caplog):
    from services.profiling import accessors
    from agent.llm import llm_router
    monkeypatch.setattr(accessors, "_profiles", {})
    monkeypatch.setattr(accessors, "_profile_labels", {})
    monkeypatch.setattr(llm_router, "invoke", Mock(side_effect=AssertionError("No model calls")))
    monkeypatch.setattr(llm_router, "embed_query", Mock(side_effect=AssertionError("No embedding calls")))
    caplog.set_level(logging.INFO)
    for name, value in {
        "SCHEMA_PRUNE_ENABLED":"true", "SCHEMA_PRUNE_MAX_FIELDS":"40", "SCHEMA_PRUNE_INCLUDE_PARENTS":"false",
        "SCHEMA_PRUNE_MIN_SIGNALS":"3", "PRUNE_ALLOW_PII_IF_NAMED":"true", "RULE_GATING_MODE":"on",
        "RULE_GENERIC_TOKEN_FRACTION":"0.30", "RULE_MIN_TOKEN_OVERLAP":"1", "RULE_GATE_MIN_SIMILARITY":"0.30",
        "GENERATOR_MAX_RULES":"4", "GENERATOR_FACTS_MAX_FIELDS":"6", "FACTS_IGNORE_VALUES":"",
        "GENERATOR_COLLECTION_GATE":"log", "QUESTION_RELEVANCE_WEIGHTS":"{}", "DATE_NORMALIZATION_ENABLED":"true",
        "DATE_TIMEZONE":"UTC", "DATE_EXPRESSIONS_FILE":"", "DATE_DAY_FIRST":"true",
        "GENERATOR_PROFILE_FACTS_ENABLED":"true", "GENERATOR_FACTS_RELEVANCE_FIRST":"true",
        "GENERATOR_FACTS_MAX_VALUES_PER_FIELD":"15", "GENERATOR_FACTS_MAX_CHARS":"2400",
        "PROFILE_VALIDATION_MODE":"strict"}.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def case():
    return json.loads((ROOT / "tests/fixtures/prompt_selection_case.json").read_text(encoding="utf-8"))


def symbols(case):
    name = next(iter(case["profiles"]))
    fields = case["profiles"][name]["fields"]
    select = lambda kind: next(path for path, spec in fields.items() if spec["kind"] == kind)
    return name, select("categorical"), select("identifier"), select("date"), next(path for path, spec in fields.items() if spec.get("pii_suspected"))


def inputs(case, question=None, config=None):
    return build_prompt_inputs(case["schema"], question or case["question"], case["rules"], case["preferences"],
                               case["profiles"].get, config, datetime.fromisoformat(case["now"]))


def publish(case):
    from services.profiling.accessors import publish_profiles
    from agent.config import MONGODB_DATABASE
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in case["profiles"].items()})


def test_prunes_three_hundred_fields_without_mutating(case):
    before = deepcopy(case)
    result = inputs(case)
    name, category, identity, moment, private = symbols(case)
    fields = result["schema"][name]["fields"]
    assert len(case["schema"]["collections"][name]["fields"]) == 300
    assert len(fields) == 40
    assert {category, identity, moment} <= set(fields) and private not in fields
    assert "(+260 more fields not shown; any field of the collection is valid)" == result["schema"][name]["omitted_fields"]
    assert fields[category] == case["schema"]["collections"][name]["fields"][category]
    assert case == before


def test_named_fields_and_presence_fill(case):
    name, _, identity, moment, _ = symbols(case)
    field = identity
    result = inputs(case, field)
    assert field in result["schema"][name]["fields"]
    assert identity in result["schema"][name]["fields"] and moment in result["schema"][name]["fields"]
    rows = result["relevance"][name]
    defaults = [row for row in rows if "default_scalar" in row["reasons"]]
    assert defaults == sorted(defaults, key=lambda row: (-row["presence"], row["field"]))
    assert len(result["schema"][name]["fields"]) == 40


@pytest.mark.parametrize("allow", [True, False])
def test_private_fields_only_when_named(case, allow):
    name, _, _, _, private = symbols(case)
    config = replace(PromptSelectionConfig(), allow_pii_if_named=allow)
    result = inputs(case, private, config)
    assert (private in result["schema"][name]["fields"]) is allow
    assert private not in result["facts"]
    result = inputs(case, config=config)
    assert private not in result["schema"][name]["fields"]


def test_excluded_container_paths(case):
    name, category, _, _, _ = symbols(case)
    case["profiles"][name]["config"] = {"exclude_fields": [category]}
    result = inputs(case)
    assert category not in result["schema"][name]["fields"]
    assert name + "." + category not in result["facts"]


def test_parent_paths_consume_budget(case):
    name, _, _, _, _ = symbols(case)
    nested = "qjxparent.qjxleaf"
    for path, kind in (("qjxparent", "object"), (nested, "text")):
        case["profiles"][name]["fields"][path] = {"kind": kind, "types": {"string": 1}}
        case["schema"]["collections"][name]["fields"][path] = ["string"]
    result = inputs(case, nested, replace(PromptSelectionConfig(), include_parents=True))
    assert {nested, "qjxparent"} <= set(result["schema"][name]["fields"])
    assert len(result["schema"][name]["fields"]) <= 40


def test_missing_profile_small_collection_and_flags(case):
    config = PromptSelectionConfig()
    name, _, _, _, _ = symbols(case)
    case["profiles"].pop(name)
    result = inputs(case)
    assert result["schema"][name]["fields"] == case["schema"]["collections"][name]["fields"]
    case["schema"]["collections"][name]["fields"] = {"qjxsmall": ["string"]}
    assert inputs(case)["schema"][name]["fields"] == {"qjxsmall": ["string"]}
    relevance = prepare_relevance(case["question"], case["schema"]["collections"], profile_lookup=case["profiles"].get)[3]
    assert prune_schema_for_prompt(case["schema"], relevance, replace(config, enabled=False)) == case["schema"]


def test_field_not_shown_still_validates_full_schema(case, caplog):
    from agents.validator_agent import ValidatorAgent, QueryValidationError
    publish(case)
    result = build_prompt_inputs(case["schema"], case["question"], case["rules"])
    name, _, _, _, private = symbols(case)
    omitted = next(field for field in case["schema"]["collections"][name]["fields"] if field not in result["schema"][name]["fields"] and field != private)
    query = {"operation": "find", "collection": name, "filter": {omitted: "qjxvalue"}}
    assert ValidatorAgent().validate(query, case["schema"])["filter"] == query["filter"]
    assert "field_outside_pruned_view" in caplog.text
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({**query, "filter": {"qjxmissing": "qjxvalue"}}, case["schema"])


def test_full_schema_identity_at_runtime_boundaries(case, monkeypatch):
    from agents.validator_agent import ValidatorAgent
    import services.filter_value_resolution as resolver
    publish(case)
    selected = case["schema"]
    before = deepcopy(selected)
    build_prompt_inputs(selected, case["question"], case["rules"])
    import agents.validator_agent as validator
    original_known = validator._known_fields
    known = Mock(wraps=original_known)
    monkeypatch.setattr(validator, "_known_fields", known)
    original_value = resolver.resolve_filter_value
    value_resolver = Mock(wraps=original_value)
    monkeypatch.setattr(resolver, "resolve_filter_value", value_resolver)
    name, category, _, _, _ = symbols(case)
    value = case["profiles"][name]["fields"][category]["values"][-1]["value"]
    query = {"operation": "find", "collection": name, "filter": {category: value}}
    validated = ValidatorAgent().validate(query, selected)
    assert known.call_args.args[0] is selected
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    assert resolver.apply_value_resolution(validated, selected, [])["filter"][category] == value
    assert value_resolver.called
    assert selected is case["schema"] and selected == before


def test_relevance_is_pure_and_preferences_relationships_apply(case):
    name, category, identity, moment, _ = symbols(case)
    other = list(case["profiles"])[1]
    preference = {"collection": name, "field": identity}
    relation = {"collection": name, "field": identity, "target_collection": other, "target_field": identity}
    before = deepcopy(case)
    args = (case["question"], list(case["profiles"]), case["profiles"].get, case["rules"], [preference], [object()], [relation])
    first = compute_question_relevance(*args)
    assert first == compute_question_relevance(*args) and case == before
    rows = {row["field"]: row for row in first[name]}
    assert "value" in rows[category]["reasons"] and "time" in rows[moment]["reasons"]
    assert {"preference", "relationship", "rule"} <= set(rows[identity]["reasons"])


@pytest.mark.parametrize("signal", ["context_selected", "relevant_field", "trigger", "token_overlap", "similarity", "mandatory", "preference"])
def test_each_rule_gate_signal(case, signal):
    name, category, _, _, _ = symbols(case)
    rule = {"knowledge_id": "qjxrule", "collections": [name], "fields": [], "text": "qjxunique", "type": "field_definition"}
    question, relevance = "qjxforeign", {name: []}
    if signal == "context_selected":
        rule["inclusion_reasons"] = [signal]
    elif signal == "relevant_field":
        rule["fields"] = [category]
        relevance[name] = [{"field": category, "reasons": ["value"]}]
    elif signal == "trigger":
        rule["inclusion_reasons"] = ["trigger:categorical"]
    elif signal == "token_overlap":
        question = "qjxunique"
    elif signal == "similarity":
        question = "qjxforeign"
        rule["semantic_similarity"] = .9
    elif signal == "mandatory":
        rule["metadata"] = {"mandatory": True}
    elif signal == "preference":
        rule["_preference"] = {}
    # Extra independent rules allow one unique lexical token to survive the
    # corpus-based generic-token rule without a language stopword list.
    decoys = [{**rule, "knowledge_id": f"qjxdecoy{index}", "text": f"qjxdistant{index}", "fields": [],
               "inclusion_reasons": [], "semantic_similarity": 0, "metadata": {}} for index in range(9)]
    for decoy in decoys:
        decoy.pop("_preference", None)
    assert rule in gate_rules([rule, *decoys], question, [name], relevance, {name: case["profiles"][name]}, PromptSelectionConfig())


def test_rule_gate_rejects_scope_generic_noise_and_caps(case, caplog):
    name = symbols(case)[0]
    rules = [{"knowledge_id": f"qjxrule{index}", "text": "qjxgeneric", "collections": [name], "type": "field_definition"} for index in range(10)]
    assert generic_tokens(rules, .3) == {"qjxgeneric"}
    config = PromptSelectionConfig()
    assert gate_rules(rules, "qjxgeneric", [name], {name: []}, {name: case["profiles"][name]}, config) == []
    assert "no_question_signal" in caplog.text and "text=" not in caplog.text
    assert gate_rules(rules, "qjxgeneric", [name], {name: []}, {name: case["profiles"][name]}, replace(config, rule_mode="log")) == rules
    for rule in rules:
        rule["mandatory"] = True
    assert len(gate_rules(rules, "qjxforeign", [name], {name: []}, {name: case["profiles"][name]}, config)) == 4
    rules[0]["collections"] = ["qjxunselected"]
    assert rules[0] not in gate_rules(rules, "qjxforeign", [name], {name: []}, {name: case["profiles"][name]}, config)


def test_irrelevant_rule_does_not_bootstrap_its_own_field(case):
    name = symbols(case)[0]
    field = list(case["profiles"][name]["fields"])[100]
    rule = {"text": "qjxdistant", "fields": [field], "collections": [name]}
    relevant = compute_question_relevance("qjxforeign", [name], case["profiles"].get, [rule])
    row = next(row for row in relevant[name] if row["field"] == field)
    assert "rule" not in row["reasons"]


def test_facts_scalar_relevance_junk_and_caps(case, monkeypatch):
    name, category, identity, moment, _ = symbols(case)
    result = inputs(case)
    assert "values=[\"string\"" not in result["facts"] and "undefined" not in result["facts"]
    assert all(path not in result["facts"] for path in ("payloadnode", "unknownnode", "singletonmark"))
    assert '"amber_widget"' in result["facts"] and "(+23 more values not shown)" in result["facts"]
    field_lines = [line for line in result["facts"].splitlines() if " kind=" in line]
    assert len(field_lines) <= 6 and len(result["facts"]) <= 2400
    assert any(item["reason"] == "junk_values" for item in result["diagnostics"])
    monkeypatch.setenv("FACTS_IGNORE_VALUES", "amber_widget")
    assert '"amber_widget"' not in inputs(case)["facts"]
    assert "date_format=" not in inputs(case, "qjxforeign")["facts"]
    # Observed custom type labels are also junk even when absent from the file.
    case["profiles"][name]["fields"][identity]["types"] = {"qjxcustomtype": 1}
    case["profiles"][name]["fields"][category]["values"].append({"value": "qjxcustomtype", "count": 100})
    assert '"qjxcustomtype"' not in inputs(case)["facts"].split("values=", 1)[-1].split("\n", 1)[0]


@pytest.mark.parametrize("mode", ["off", "log", "on"])
def test_collection_gate_modes_and_source(case, mode, caplog):
    names = list(case["profiles"])
    relevant = {names[0]: [{"field": "qjxfield", "reasons": ["value"]}]}
    config = replace(PromptSelectionConfig(), collection_mode=mode)
    result = collection_relevance(names, "qjxforeign", case["profiles"], relevant, config=config)
    assert result == ([names[0]] if mode == "on" else names)
    if mode != "off":
        assert "would_drop=true" in caplog.text
    rule = {"collections": [names[0]], "metadata": {"source_collection": names[1]}}
    result = collection_relevance(names, "qjxforeign", case["profiles"], relevant, [rule], config=config)
    assert names[1] in result
    assert collection_relevance(names, "qjxforeign", case["profiles"], {}, config=replace(config, collection_mode="on"))


def test_collection_gate_changes_prompt_only(case):
    before = deepcopy(case["schema"])
    case["rules"] = []
    for name, profile in list(case["profiles"].items())[1:]:
        profile["fields"][symbols(case)[1]]["values"] = []
    name, category, _, _, _ = symbols(case)
    value = case["profiles"][name]["fields"][category]["values"][-1]["value"].replace("_", " ")
    result = inputs(case, value, replace(PromptSelectionConfig(), collection_mode="on"))
    assert list(result["schema"]) == [name]
    assert case["schema"] == before


@pytest.mark.parametrize("method", ["generate", "repair", "optimize"])
@pytest.mark.parametrize("has_profile", [True, False])
def test_legacy_or_no_profile_match_prechange_golden(case, monkeypatch, method, has_profile):
    import agents.query_generator_agent as module
    if has_profile:
        monkeypatch.setenv("SCHEMA_PRUNE_ENABLED", "false")
        monkeypatch.setenv("RULE_GATING_MODE", "off")
        monkeypatch.setenv("GENERATOR_COLLECTION_GATE", "off")
        publish(case)
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(case["now"]).astimezone(tz)
    monkeypatch.setattr(module, "datetime", Frozen)
    agent = module.QueryGeneratorAgent()
    monkeypatch.setattr(agent, "_invoke_json", lambda prompt, *args, **kwargs: prompt)
    previous = {"operation": "find", "collection": symbols(case)[0], "filter": {}}
    if method == "generate":
        actual = agent.generate(case["question"], case["schema"], operation_hint="find", database_knowledge=case["rules"])
    elif method == "repair":
        actual = agent.repair(case["question"], case["schema"], previous, "Synthetic validation error", database_knowledge=case["rules"])
    else:
        actual = agent.optimize(case["question"], {}, case["schema"], previous, case["question"], database_knowledge=case["rules"])
    golden = json.loads((ROOT / "tests/fixtures/prompt_selection_golden.json").read_text(encoding="utf-8"))
    assert actual == (golden[method] if has_profile else golden[method].split("\nProfile facts:\n", 1)[0])


def test_no_profile_keeps_original_prompt(case, monkeypatch):
    from services.query.prompt_selection.builder import prompt_query_context
    from database.schema_inspector import compact_schema
    result = build_prompt_inputs(case["schema"], case["question"], case["rules"], profile_lookup=lambda _: None)
    assert result["schema"] == compact_schema(case["schema"]) and result["rules"] == case["rules"] and result["facts"] == ""
    assert not result["active"]


def test_new_modules_contain_no_fixture_names_or_values(case):
    forbidden = set(case["profiles"])
    for profile in case["profiles"].values():
        forbidden.update(profile["fields"])
        forbidden.update(row["value"] for spec in profile["fields"].values() if spec.get("kind") == "categorical" for row in spec.get("values", []))
    paths = list((ROOT / "services/query/prompt_selection").glob("*.py")) + [ROOT / "scripts/prompt_size_report.py"]
    for path in paths:
        literals = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not literals & forbidden, (path, literals & forbidden)


def test_sparse_nested_profile_never_returns_empty(case):
    name = symbols(case)[0]
    field = "qjxcontainer.qjxinner.qjxscalar"
    case["profiles"] = {name: {"collection": name, "fields": {field: {"kind": "text"}}}}
    case["schema"] = {"collections": {name: {"fields": {field: ["string"], "qjxother": ["string"]}}}}
    result = inputs(case, "qjxforeign", replace(PromptSelectionConfig(), max_fields=1))
    assert result["schema"][name]["fields"] == {field: ["string"]}


def test_parent_option_with_tiny_budget_keeps_leaf(case):
    name = symbols(case)[0]
    paths = ["qjxcontainer", "qjxcontainer.qjxinner", "qjxcontainer.qjxinner.qjxscalar"]
    for path in paths:
        case["profiles"][name]["fields"][path] = {"kind": "object"}
        case["schema"]["collections"][name]["fields"][path] = ["object"]
    result = inputs(case, "qjxscalar", replace(PromptSelectionConfig(), max_fields=1, include_parents=True))
    assert list(result["schema"][name]["fields"]) == [paths[-1]]


def test_generator_stage_reuses_existing_vector_without_embedding(case):
    from services.knowledge_retriever import KnowledgeRetriever
    from services.knowledge.retrieval.retriever import KNOWLEDGE_EMBEDDING_MODEL
    publish(case)
    name = symbols(case)[0]
    record = {"_id": "qjxvector", "type": "field_definition", "text": "qjxdistant",
              "collections": [name], "fields": [], "active": True, "status": "active",
              "embedding": [1., 0.], "embedding_model": KNOWLEDGE_EMBEDDING_MODEL}
    original = deepcopy(record)
    retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: [record]))
    def run(vector):
        return retriever.rules_for_generator("qjxforeign", [name], {}, full_schema=case["schema"], question_embedding=vector)["knowledge"]
    assert run(None) == []
    assert [rule["knowledge_id"] for rule in run([1., 0.])] == [record["_id"]]
    assert run([0., 1.]) == []
    assert record == original
    record["embedding_model"] = "qjxmismatch"
    assert run([1., 0.]) == []


@pytest.mark.parametrize("method", ["generate", "repair", "optimize"])
def test_all_generator_paths_use_prompt_copy(case, monkeypatch, method):
    import agents.query_generator_agent as module
    publish(case)
    before = deepcopy(case["schema"])
    build = Mock(wraps=module.build_prompt_inputs)
    monkeypatch.setattr(module, "build_prompt_inputs", build)
    agent = module.QueryGeneratorAgent()
    monkeypatch.setattr(agent, "_invoke_json", lambda prompt, *args, **kwargs: prompt)
    name = symbols(case)[0]
    previous = {"operation": "find", "collection": name, "filter": {}}
    if method == "generate":
        text = agent.generate(case["question"], case["schema"], database_knowledge=case["rules"])
    elif method == "repair":
        text = agent.repair(case["question"], case["schema"], previous, "qjxerror", database_knowledge=case["rules"])
    else:
        context = {"collections": list(case["profiles"]), "fields": {name: list(case["profiles"][name]["fields"])}}
        old_context = deepcopy(context)
        text = agent.optimize(case["question"], context, case["schema"], previous, case["question"], database_knowledge=case["rules"])
        assert context == old_context
        assert len(json.loads(text.split("\n", 1)[1].split("\nProfile facts:", 1)[0])["query_context"]["fields"][name]) <= 40
    assert build.call_args.args[0] is case["schema"]
    assert "more fields not shown; any field of the collection is valid" in text
    assert case["schema"] == before


def test_collection_gate_uses_published_labels_and_not_one_sided_relation(case):
    from services.profiling.accessors import publish_profiles
    from agent.config import MONGODB_DATABASE
    names = list(case["profiles"])
    name, _, identity, _, _ = symbols(case)
    case["profiles"][name]["relationships"] = [{"field": identity, "target_collection": names[1], "target_field": identity}]
    publish_profiles(MONGODB_DATABASE, {entry: {"facts": profile, "generated": {"aliases": ["qjxalias"] if entry == name else []}}
                                         for entry, profile in case["profiles"].items()})
    config = replace(PromptSelectionConfig(), collection_mode="on")
    result = build_prompt_inputs(case["schema"], "qjxalias", config=config)
    assert list(result["schema"]) == [name]
    assert " -> " not in result["facts"]
    # A relationship alone cannot turn an unsupported target into a selection.
    assert len(case["schema"]["collections"]) == 3


def test_date_trigger_requires_resolver_or_explicit_date_field(case):
    name, _, _, moment, _ = symbols(case)
    rule = {"collections": [name], "fields": [moment], "text": "qjxdistant", "inclusion_reasons": ["trigger:date"]}
    config = PromptSelectionConfig()
    profiles = {name: case["profiles"][name]}
    relevance = compute_question_relevance("qjxforeign", [name], profiles.get, [rule])
    assert gate_rules([rule], "qjxforeign", [name], relevance, profiles, config) == []
    relevance = compute_question_relevance(moment, [name], profiles.get, [rule])
    assert gate_rules([rule], moment, [name], relevance, profiles, config) == [rule]


def test_config_defaults_and_weight_overrides(monkeypatch):
    monkeypatch.setenv("QUESTION_RELEVANCE_WEIGHTS", '{"field_leaf":23}')
    assert PromptSelectionConfig.from_env().weights["field_leaf"] == 23
    assert PromptSelectionConfig.from_env().weights["value"] == 20
    monkeypatch.setenv("RULE_GATING_MODE", "qjxinvalid")
    with pytest.raises(ValueError):
        PromptSelectionConfig.from_env()


def test_all_unsafe_paths_keep_collection_metadata_without_exposure(case):
    name = symbols(case)[0]
    for profile in case["profiles"].values():
        for field in profile["fields"].values():
            field["pii_suspected"] = True
    result = inputs(case, "qjxforeign")
    assert result["schema"][name]["fields"] == {}
    assert "300 more fields not shown" in result["schema"][name]["omitted_fields"]
    assert result["facts"] == ""


def test_rule_cap_uses_existing_type_order_then_signal_strength(case):
    name = symbols(case)[0]
    weak = {"knowledge_id": "qjxweak", "collections": [name], "text": "qjxdistant", "type": "business_rule",
            "inclusion_reasons": ["context_selected"]}
    strong = {**weak, "knowledge_id": "qjxstrong", "mandatory": True}
    lower_type = {**strong, "knowledge_id": "qjxlower", "type": "field_definition"}
    result = gate_rules([lower_type, weak, strong], "qjxforeign", [name], {}, {name: case["profiles"][name]},
                        replace(PromptSelectionConfig(), max_rules=2))
    assert result == [strong, weak]


def test_refinement_preserves_join_owners_fields_and_shared_reasons(case):
    name, category, identity, moment, _ = symbols(case)
    other = list(case["profiles"])[1]
    previous = {"operation": "aggregate", "collection": name, "pipeline": [
        {"$match": {moment: {"$gte": "2026-10-15"}}},
        {"$lookup": {"from": other, "localField": identity, "foreignField": identity, "as": "qjxjoined"}}]}
    result = build_prompt_inputs(case["schema"], case["question"], profile_lookup=case["profiles"].get,
                                 config=replace(PromptSelectionConfig(), collection_mode="on"), previous_query=previous)
    rows = {row["field"]: row for row in result["relevance"][name]}
    assert {"time", "preference"} <= set(rows[moment]["reasons"])
    assert other in result["schema"] and identity in result["schema"][other]["fields"]
    result = build_prompt_inputs(case["schema"], "qjxforeign", profile_lookup=case["profiles"].get,
                                 config=replace(PromptSelectionConfig(), collection_mode="on"), previous_query=previous)
    assert name in result["schema"] and other in result["schema"]
