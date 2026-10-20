"""Relevance changes must reduce noise without losing constraints or joins."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.query.prompt_selection.builder import build_prompt_inputs
from services.query.prompt_selection.config import PromptSelectionConfig
from services.query.prompt_selection.gates import gate_rules
from services.query.prompt_selection.relevance import compute_question_relevance

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(monkeypatch):
    from services.profiling import accessors
    monkeypatch.setattr(accessors, "_profiles", {})
    monkeypatch.setattr(accessors, "_profile_labels", {})
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "true")
    monkeypatch.setenv("GENERATOR_FACTS_MAX_CHARS", "2400")
    monkeypatch.setenv("DATE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("KNOWLEDGE_CONTEXT_RELEVANCE_MODE", "on")
    return json.loads((ROOT / "tests/fixtures/prompt_selection_case.json").read_text(encoding="utf-8"))


def test_rule_references_cannot_supply_their_own_question_evidence(case):
    name = next(iter(case["profiles"]))
    path = next(iter(case["profiles"][name]["fields"]))
    rule = {"knowledge_id": "self", "collections": [name], "fields": [path], "text": "unrelated"}
    result = gate_rules([rule], "absentword", [name], {name: [{"field": path, "reasons": ["rule"]}]},
                        case["profiles"], PromptSelectionConfig())
    assert result == []


def test_dropped_rules_leave_no_fact_or_field_signal(case):
    name = next(iter(case["profiles"]))
    category = next(path for path, spec in case["profiles"][name]["fields"].items() if spec["kind"] == "categorical")
    optional = next(path for path, spec in case["profiles"][name]["fields"].items() if spec["kind"] == "text")
    rules = [
        {"knowledge_id": "keep", "type": "business_rule", "collections": [name], "fields": [category], "text": "primary", "semantic_similarity": .9},
        {"knowledge_id": "drop", "type": "field_definition", "collections": [name], "fields": [optional], "text": "secondary", "semantic_similarity": .9},
    ]
    before = deepcopy(case)
    result = build_prompt_inputs(case["schema"], case["question"], rules, profile_lookup=case["profiles"].get,
                                config=replace(PromptSelectionConfig(), max_rules=1))
    assert [rule["knowledge_id"] for rule in result["rules"]] == ["keep"]
    row = next(row for row in result["relevance"][name] if row["field"] == optional)
    assert "rule" not in row["reasons"] and name + "." + optional + " kind=" not in result["facts"]
    assert case == before


def test_constraints_and_required_facts_survive_optional_budgets(case, monkeypatch):
    name = next(iter(case["profiles"]))
    paths = [path for path, spec in case["profiles"][name]["fields"].items() if spec["kind"] == "text"][:3]
    rules = [{"knowledge_id": str(index), "collections": [name], "fields": [path], "text": "required",
              "enforcement": {"required_filters": {path: "fixed"}}} for index, path in enumerate(paths)]
    monkeypatch.setenv("GENERATOR_FACTS_MAX_CHARS", "20")
    result = build_prompt_inputs(case["schema"], "unrelated", rules, profile_lookup=case["profiles"].get,
        config=replace(PromptSelectionConfig(), max_rules=1, max_fields=1, facts_max_fields=1))
    assert len(result["rules"]) == 3
    assert set(paths) <= set(result["schema"][name]["fields"])
    assert all(name + "." + path + " kind=" in result["facts"] for path in paths)


def test_required_join_keeps_both_sources_and_join_keys(case):
    left, right = list(case["profiles"])[:2]
    key = next(path for path, spec in case["profiles"][left]["fields"].items() if spec["kind"] == "identifier")
    case["profiles"][right]["fields"][key] = deepcopy(case["profiles"][left]["fields"][key])
    case["schema"]["collections"][right]["fields"][key] = ["string"]
    rule = {"knowledge_id": "join", "type": "relationship", "collections": [left, right], "fields": [key], "text": "link"}
    result = build_prompt_inputs(case["schema"], "Connect " + left + " and " + right, [rule], profile_lookup=case["profiles"].get,
                                config=replace(PromptSelectionConfig(), max_rules=1, max_fields=1))
    assert {left, right} <= set(result["schema"])
    assert all(key in result["schema"][name]["fields"] for name in (left, right))
    assert result["rules"] == [rule]


def test_sparse_profile_does_not_hide_catalog_backed_mandatory_field(case):
    name = next(iter(case["profiles"]))
    path = next(path for path, spec in case["profiles"][name]["fields"].items() if spec["kind"] == "text")
    case["profiles"][name]["fields"].pop(path)
    rule = {"knowledge_id": "sparse", "collections": [name], "fields": [path], "text": "required", "mandatory": True}
    result = build_prompt_inputs(case["schema"], "unrelated", [rule], profile_lookup=case["profiles"].get,
                                config=replace(PromptSelectionConfig(), max_fields=1))
    assert path in result["schema"][name]["fields"]
    assert name + "." + path + " kind=" not in result["facts"]  # Do not invent missing observations.


def test_noise_reduction_is_visible_at_prompt_boundary(case):
    name = next(iter(case["profiles"]))
    # A shared categorical value is legitimate ambiguity, not a distractor.
    # Give only the intended source evidence for this question.
    for other, profile in case["profiles"].items():
        if other != name:
            profile["generated"] = {}
            for spec in profile["fields"].values():
                spec["values"] = []
    before = deepcopy(case["schema"])
    old = build_prompt_inputs(case["schema"], case["question"], case["rules"], profile_lookup=case["profiles"].get,
                             config=replace(PromptSelectionConfig(), rule_mode="off", collection_mode="off"))
    new = build_prompt_inputs(case["schema"], case["question"], case["rules"], profile_lookup=case["profiles"].get,
                              config=replace(PromptSelectionConfig(), collection_mode="on"))
    assert len(new["schema"]) < len(old["schema"])
    assert len(new["rules"]) < len(old["rules"])
    assert len(json.dumps(new["schema"])) + len(json.dumps(new["rules"])) < len(json.dumps(old["schema"])) + len(json.dumps(old["rules"]))
    assert name in new["schema"] and case["schema"] == before


def retrieve_policy(monkeypatch, scores, keywords=None, required=(), mode="on", top_k=1):
    import services.knowledge_retriever as module
    monkeypatch.setenv("KNOWLEDGE_CONTEXT_RELEVANCE_MODE", mode)
    monkeypatch.setattr(module, "KNOWLEDGE_RANK_SELECTION_ENABLED", False)
    monkeypatch.setattr(module, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", True)
    monkeypatch.setattr(module, "KNOWLEDGE_CONTEXT_FLOOR", .3)
    catalog = json.loads((ROOT / "tests/fixtures/context_fast_path_case.json").read_text(encoding="utf-8"))["catalog"]
    name = next(iter(catalog["collections"]))
    records = [{"_id": key, "text": "instruction " + key, "type": "business_rule", "scope": "database",
        "collections": [name], "fields": [], "status": "active", "active": True, "mandatory": key in required,
        "embedding_model": module.KNOWLEDGE_EMBEDDING_MODEL,
        "embedding": [score, math.sqrt(1 - score ** 2)] if score is not None else []} for key, score in scores.items()]
    retriever = module.KnowledgeRetriever(SimpleNamespace(list_active=lambda: deepcopy(records)),
        embedding_service=SimpleNamespace(ensure_embeddings=lambda record, **kwargs: deepcopy(record)))
    retriever._keyword_index.match = Mock(return_value=keywords or {})
    return retriever.rules_for_context([1., 0.], catalog, threshold=.75, top_k=top_k, question="rare evidence")


@pytest.mark.parametrize("mode,expected", [("off", 1), ("log", 1), ("on", 0)])
def test_keyword_gate_has_reversible_rollout(monkeypatch, mode, expected):
    result = retrieve_policy(monkeypatch, {"low": .01}, {"low": ["rare"]}, mode=mode)
    assert len(result["knowledge"]) == expected
    if mode != "off":
        assert result["_context_diagnostics"]["policy_rejections"][0]["reason"] == "keyword_below_similarity_floor"


def test_optional_keyword_rules_cannot_exceed_shared_budget(monkeypatch):
    result = retrieve_policy(monkeypatch, {"high": .99, "extra": .9}, {"extra": ["rare"]})
    assert [rule["knowledge_id"] for rule in result["knowledge"]] == ["high"]
    assert any(item["reason"] == "optional_rule_cap" for item in result["_context_diagnostics"]["policy_rejections"])


def test_required_rules_survive_missing_vectors_and_optional_budget(monkeypatch):
    result = retrieve_policy(monkeypatch, {"high": .99, "required1": None, "required2": .01}, required=("required1", "required2"))
    assert {rule["knowledge_id"] for rule in result["knowledge"]} == {"high", "required1", "required2"}
    assert all(rule["selection_method"] == "required" for rule in result["knowledge"] if rule["knowledge_id"].startswith("required"))


def test_missing_vectors_need_more_than_one_keyword(monkeypatch):
    assert not retrieve_policy(monkeypatch, {"low": None}, {"low": ["rare"]})["knowledge"]
    assert retrieve_policy(monkeypatch, {"low": None}, {"low": ["rare", "evidence"]})["knowledge"]
