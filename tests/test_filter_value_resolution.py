"""Data-driven spelling resolution; every catalog/model/database is fake."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.filter_value_resolution as resolver
import services.value_catalog as catalog_module
from agents.validator_agent import ValidatorAgent
from services.query_parameter_binder import parameterize_filter_values, bind_query_parameters

C1, C2 = "col_x", "col_remote"
F1, F2, F3 = "stateX", "kindY", "idZ"
CATALOG = {(C1, F1): {"Failed", "failed"}, (C1, F2): {"Échoué"}, (C1, F3): None,
           (C2, F1): {"REMOTE", "remote"}}
SCHEMA = {"collections": {
    C1: {"fields": {F1: ["string"], F2: ["string"], F3: ["string"], "nested." + F1: ["string"]}},
    C2: {"fields": {F1: ["string"], F2: ["string"]}},
}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from agent.llm import llm_router
    monkeypatch.setattr(llm_router, "invoke", Mock(side_effect=AssertionError("No real model calls")))
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_VARIANTS_POLICY", "merge")
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset())
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_SUGGESTION_CUTOFF", .8)
    monkeypatch.setattr(resolver, "get_value_catalog", lambda collection, field: CATALOG.get((collection, field)))
    monkeypatch.setattr(resolver, "get_value_catalog_status", lambda collection, field:
                        "not_categorical" if (collection, field) in CATALOG else "catalog_unavailable")


def query(condition, operation="find"):
    return {"operation": operation, "collection": C1, "filter": condition}


def apply(condition, rules=()):
    return resolver.apply_value_resolution(query(condition), SCHEMA, rules)["filter"]


def rule(text, collection=C1, field=F1, kind="coded_value", **changes):
    return {"type": kind, "text": text, "collections": [collection], "fields": [field],
            "active": True, "status": "active", **changes}


def test_single_stored_case_spelling(monkeypatch, caplog):
    monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: {"Failed"})
    assert apply({F1: "failed"}) == {F1: "Failed"}
    assert '"reason":"case_variant"' in caplog.text


def test_merge_and_exact_policies(monkeypatch):
    assert apply({F1: "FAILED"}) == {F1: {"$in": ["Failed", "failed"]}}
    assert apply({F1: "Failed"}) == {F1: {"$in": ["Failed", "failed"]}}
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_VARIANTS_POLICY", "exact")
    assert apply({F1: "Failed"}) == {F1: "Failed"}
    assert apply({F1: "FAILED"}) == {F1: {"$in": ["Failed", "failed"]}}


def test_accent_and_compatibility_normalization(caplog):
    assert apply({F2: "echoue"}) == {F2: "Échoué"}
    assert '"reason":"accent_variant"' in caplog.text
    assert apply({F1: "ＦＡＩＬＥＤ"}) == {F1: {"$in": ["Failed", "failed"]}}


def test_typo_is_only_suggested(caplog):
    assert apply({F1: "faild"}) == {F1: "faild"}
    assert "value_not_in_catalog" in caplog.text
    assert '"closest_stored_value":"Failed"' in caplog.text
    assert apply({F1: "unrelated_unknown"}) == {F1: "unrelated_unknown"}


@pytest.mark.parametrize("value", [42, None, True, {"$oid": "0123456789abcdef01234567"}, {"$date": "2026-01-01"}])
def test_typed_and_non_string_values_do_not_read_catalog(monkeypatch, value):
    getter = Mock(side_effect=AssertionError("No catalog for typed/non-string values"))
    monkeypatch.setattr(resolver, "get_value_catalog", getter)
    assert apply({F1: value}) == {F1: value}
    getter.assert_not_called()


def test_high_cardinality_field_stays_unchanged():
    assert apply({F3: "Mixed_Id"}) == {F3: "Mixed_Id"}


@pytest.mark.parametrize("condition,expected", [
    ({"$eq": "FAILED"}, {"$in": ["Failed", "failed"]}),
    ({"$ne": "FAILED"}, {"$nin": ["Failed", "failed"]}),
    ({"$in": ["FAILED", "unknown", 7]}, {"$in": ["Failed", "failed", "unknown", 7]}),
    ({"$nin": ["FAILED", "Failed"]}, {"$nin": ["Failed", "failed"]}),
])
def test_comparison_and_membership_operators(condition, expected):
    assert apply({F1: condition}) == {F1: expected}


def test_single_value_keeps_eq_and_ne():
    assert apply({F2: {"$eq": "echoue"}}) == {F2: {"$eq": "Échoué"}}
    assert apply({F2: {"$ne": "echoue"}}) == {F2: {"$ne": "Échoué"}}


def test_all_variants_are_alternatives_not_joint_requirements():
    resolved = apply({F1: {"$all": ["FAILED", "unknown"]}})
    assert resolved == {"$and": [{F1: {"$in": ["Failed", "failed"]}}, {F1: {"$all": ["unknown"]}}]}
    concrete = query(resolved)
    generated = parameterize_filter_values({"template": concrete, "parameters": {}})
    assert bind_query_parameters(generated["template"], generated["parameters"])["filter"] == resolved
    assert any(value == ["Failed", "failed"] for value in generated["parameters"].values())


def test_ne_and_nin_constraints_are_both_preserved():
    assert apply({F1: {"$ne": "FAILED", "$nin": ["unknown"]}}) == {"$and": [
        {F1: {"$nin": ["unknown"]}}, {F1: {"$nin": ["Failed", "failed"]}}]}


@pytest.mark.parametrize("condition", [
    {F1: {"$regex": "failed", "$options": "i"}},
    {F1: {"$not": {"$regex": "failed"}}},
    {"$expr": {"$eq": ["$" + F1, "failed"]}},
])
def test_regex_and_expressions_are_untouched(monkeypatch, condition):
    monkeypatch.setattr(resolver, "get_value_catalog", Mock(side_effect=AssertionError("No regex or expr lookup")))
    assert apply(condition) == condition


def test_lookup_facet_alias_and_element_paths_use_correct_collection(monkeypatch):
    catalog = {**CATALOG, (C1, "nested." + F1): {"LOCAL"}}
    reads = []
    def getter(collection, field):
        reads.append((collection, field))
        return catalog.get((collection, field))
    monkeypatch.setattr(resolver, "get_value_catalog", getter)
    original = {"operation": "aggregate", "collection": C1, "pipeline": [
        {"$lookup": {"from": C2, "as": "joined_rows", "pipeline": [
            {"$match": {F1: "REMOTE"}}, {"$facet": {"branch_rows": [{"$match": {F1: "remote"}}]}}]}},
        {"$match": {"joined_rows." + F1: "REMOTE", "nested": {"$elemMatch": {F1: "local"}}}},
    ]}
    resolved = resolver.apply_value_resolution(original, SCHEMA, [])
    assert resolved["pipeline"][0]["$lookup"]["pipeline"][0]["$match"][F1] == {"$in": ["REMOTE", "remote"]}
    assert resolved["pipeline"][0]["$lookup"]["pipeline"][1]["$facet"]["branch_rows"][0]["$match"][F1] == {"$in": ["REMOTE", "remote"]}
    assert resolved["pipeline"][1]["$match"]["joined_rows." + F1] == {"$in": ["REMOTE", "remote"]}
    assert resolved["pipeline"][1]["$match"]["nested"]["$elemMatch"][F1] == "LOCAL"
    assert (C2, F1) in reads and (C1, "nested." + F1) in reads
    assert original["pipeline"][0]["$lookup"]["pipeline"][0]["$match"][F1] == "REMOTE"


@pytest.mark.parametrize("kind", ["coded_value", "field_definition"])
def test_case_distinction_guard_preserves_exact_or_unrecognized_user_spelling(monkeypatch, kind, caplog):
    monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: {"Aaa", "aaa"})
    rules = [rule('"Aaa" denotes one class.', kind=kind), rule('"aaa" denotes another class.', kind=kind)]
    assert apply({F1: "Aaa"}, rules) == {F1: "Aaa"}
    assert apply({F1: "AAA"}, rules) == {F1: "AAA"}
    assert '"reason":"rule_guard"' in caplog.text


def test_guard_does_not_infer_two_spellings_from_one_folded_mention(monkeypatch):
    monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: {"Aaa", "aaa"})
    assert apply({F1: "AAA"}, [rule('"Aaa" is a category.')]) == {F1: {"$in": ["Aaa", "aaa"]}}
    assert apply({F1: "AAA"}, [rule('Aaa and aaa', collection=C2)]) == {F1: {"$in": ["Aaa", "aaa"]}}
    assert apply({F1: "AAA"}, [rule('Aaa and aaa', active=False)]) == {F1: {"$in": ["Aaa", "aaa"]}}


def test_sensitive_configuration_and_feature_off_do_not_read(monkeypatch):
    getter = Mock(side_effect=AssertionError("No read when disabled"))
    monkeypatch.setattr(resolver, "get_value_catalog", getter)
    monkeypatch.setattr(resolver, "KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset({C1 + "." + F1}))
    assert apply({F1: "FAILED"}) == {F1: "FAILED"}
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", False)
    original = query({F2: "echoue"})
    assert resolver.apply_value_resolution(original, SCHEMA, []) is original
    getter.assert_not_called()


def test_unavailable_catalog_is_fail_open(caplog):
    original = query({"missing_catalog_field": "FAILED"})
    assert resolver.apply_value_resolution(original, SCHEMA, []) == original
    assert '"reason":"catalog_unavailable"' in caplog.text


@pytest.mark.parametrize("condition", [
    {F1: "FAILED"}, {F1: {"$in": ["FAILED", "unknown"]}},
    {F1: {"$all": ["FAILED", "unknown"]}}, {F1: {"$ne": "FAILED", "$nin": ["unknown"]}},
])
def test_idempotence(condition):
    first = resolver.apply_value_resolution(query(condition), SCHEMA, [])
    assert resolver.apply_value_resolution(first, SCHEMA, []) == first


def test_computed_fields_do_not_resolve_against_pre_transformation_data():
    original = {"operation": "aggregate", "collection": C1, "pipeline": [
        {"$set": {F1: {"$toLower": "$" + F1}}}, {"$match": {F1: "failed"}}]}
    assert resolver.apply_value_resolution(original, SCHEMA, []) == original


def test_validated_query_reparameterizes_merged_list_as_one_parameter():
    generated = parameterize_filter_values({"template": query({F1: "FAILED"}), "parameters": {}})
    original = deepcopy(generated)
    bound = bind_query_parameters(generated["template"], generated["parameters"])
    validated = ValidatorAgent().validate(bound, SCHEMA)
    resolved = resolver.apply_value_resolution(validated, SCHEMA, [])
    rebuilt = resolver.parameterize_resolved_query(generated, resolved)
    slot = rebuilt["template"]["filter"][F1]["$in"]
    assert set(slot) == {"$param"}
    assert rebuilt["parameters"][slot["$param"]] == ["Failed", "failed"]
    assert len(rebuilt["parameters"]) == 1
    assert bind_query_parameters(rebuilt["template"], rebuilt["parameters"]) == resolved
    assert generated == original


def test_feature_off_parameterization_matches_previous_behavior(monkeypatch):
    from services.query_parameter_binder import reconcile_template
    generated = parameterize_filter_values({"template": query({F1: "FAILED"}), "parameters": {}})
    validated = ValidatorAgent().validate(bind_query_parameters(generated["template"], generated["parameters"]), SCHEMA)
    monkeypatch.setattr(resolver, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", False)
    assert resolver.parameterize_resolved_query(generated, validated) == parameterize_filter_values(reconcile_template(generated, validated))


def test_no_hardcoded_database_terms_in_new_modules():
    forbidden = {C1, C2, F1, F2, F3, "col_x", "stateX", "kindY", "idZ", "failed", "Failed", "FAILED",
                 "echoue", "Échoué", "Aaa", "aaa", "REMOTE", "remote", "LOCAL", "Mixed_Id",
                 "missing_catalog_field", "nested", "joined_rows", "branch_rows", "private_token",
                 "PrivateValue", "privatevalue", "faild", "unrelated_unknown", "unknown", "AAA", "ＦＡＩＬＥＤ"}
    for module in (resolver, catalog_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        literals = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not literals & forbidden, (module.__name__, literals & forbidden)


class Cursor:
    def __init__(self, documents):
        self.documents = documents
        self.maximum = None
    def limit(self, maximum):
        self.maximum = maximum
        return self
    def __iter__(self):
        return iter(self.documents[:self.maximum])


def value_catalog(monkeypatch, documents, clock=None):
    cursor = Cursor(documents)
    collection = SimpleNamespace(find=Mock(return_value=cursor))
    database = {C1: collection}
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_CATEGORICAL_MAX_DISTINCT", 30)
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE", 50000)
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS", 3600)
    return catalog_module.ValueCatalog(lambda: database, clock), collection, cursor


def test_bounded_nested_array_catalog_cache_and_copy(monkeypatch):
    instance, collection, cursor = value_catalog(monkeypatch, [
        {"nested": [{F1: ["Failed", "failed"]}, {F1: "FAILED"}]}, {"nested": {F1: None}}, {}])
    assert instance.get(C1, "nested." + F1) == {"Failed", "failed", "FAILED"}
    assert cursor.maximum == 50000
    args, kwargs = collection.find.call_args
    assert args == ({}, {"nested." + F1: 1})
    assert kwargs["max_time_ms"] > 0
    instance.get(C1, "nested." + F1).clear()
    assert instance.get(C1, "nested." + F1)
    collection.find.assert_called_once()


@pytest.mark.parametrize("documents", [[{F1: str(index)} for index in range(31)], [{F1: 1}], [{F1: ["Failed", 1]}], [{}]])
def test_non_categorical_data_by_distinct_count_and_actual_type(monkeypatch, documents):
    instance, _, _ = value_catalog(monkeypatch, documents)
    assert instance.get(C1, F1) is None
    assert instance.status(C1, F1) == "not_categorical"


def test_sample_size_and_ttl_are_configurable(monkeypatch):
    clock = Mock(return_value=0.)
    instance, collection, _ = value_catalog(monkeypatch, [{F1: "Failed"}, {F1: "failed"}], clock)
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE", 1)
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS", 2)
    assert instance.get(C1, F1) == {"Failed"}
    clock.return_value = 3.
    assert instance.get(C1, F1) == {"Failed"}
    assert collection.find.call_count == 2


def test_read_failure_does_not_break_query(monkeypatch, caplog):
    instance, collection, _ = value_catalog(monkeypatch, [])
    collection.find.side_effect = RuntimeError("PRIVATE_CONNECTION_DETAIL")
    monkeypatch.setattr(resolver, "get_value_catalog", instance.get)
    monkeypatch.setattr(resolver, "get_value_catalog_status", instance.status)
    original = query({F1: "FAILED"})
    assert resolver.apply_value_resolution(original, SCHEMA, []) == original
    assert "catalog_unavailable" in caplog.text
    assert "PRIVATE_CONNECTION_DETAIL" not in caplog.text


def test_catalog_feature_off_performs_no_database_read(monkeypatch):
    provider = Mock(side_effect=AssertionError("No database access when disabled"))
    monkeypatch.setattr(catalog_module, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", False)
    assert catalog_module.ValueCatalog(provider).get(C1, F1) is None
    provider.assert_not_called()


def test_array_of_strings_primitive_element_match_and_plain_all(monkeypatch):
    assert apply({F1: {"$elemMatch": {"$eq": "FAILED"}}}) == {F1: {"$elemMatch": {"$in": ["Failed", "failed"]}}}
    assert apply({F2: {"$all": ["echoue"]}}) == {F2: {"$all": ["Échoué"]}}


def test_field_scoped_values_are_redacted_by_existing_safety_policy(monkeypatch, caplog):
    sensitive_field = "private_token"
    monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: {"PrivateValue"})
    apply({sensitive_field: "privatevalue"})
    assert "PrivateValue" not in caplog.text and "privatevalue" not in caplog.text
    assert "[redacted]" in caplog.text


@pytest.mark.parametrize("available", [True, False])
def test_orchestrator_post_validation_handoff_is_resolved_or_fail_open(monkeypatch, available):
    import orchestrator
    from agent.llm import llm_router
    if not available:
        monkeypatch.setattr(resolver, "get_value_catalog", lambda *args: None)
        monkeypatch.setattr(resolver, "get_value_catalog_status", lambda *args: "catalog_unavailable")
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: deepcopy(SCHEMA))
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", lambda *args: {"catalog": SCHEMA, "telemetry": {}})
    calls = []
    def fake_invoke(role, messages, output):
        calls.append(role)
        if role == "query_context":
            payload = {"complexity": "simple", "operation": "find", "collections": [C1], "fields": {C1: [F1]}}
        else:
            payload = {"semantic_intent": "Fetch selected rows", "template": query({F1: "FAILED"}), "parameters": {}}
        return SimpleNamespace(content=json.dumps(payload))
    monkeypatch.setattr(llm_router, "invoke", fake_invoke)
    app = orchestrator.OrchestratorAgent()
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [], "_active_rules": [], "telemetry": {}}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [], "user_preferences": []}
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "fake"}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())
    result = app.run("Show selected rows")
    assert result["execution_success"], result
    expected = {"$in": ["Failed", "failed"]} if available else "FAILED"
    assert result["validated_query"]["filter"][F1] == expected
    generated = result["generated_query"]
    assert bind_query_parameters(generated["template"], generated["parameters"])["filter"][F1] == expected
    assert app.executor.execute.call_args.args[0]["filter"][F1] == expected
    assert calls == ["query_context", "query_generator"]


def test_resolved_template_reuse_keeps_cached_document_and_uses_runtime_parameters(monkeypatch):
    import services.template_decision_service as decision_module
    from database.query_template_repository import QueryTemplateRepository, template_signature
    from services.template_parameter_extractor import describe_parameters
    from services.query_history_service import QueryHistoryService
    from services.schema_bootstrap_service import schema_bootstrap_service
    from agent.config import DEFAULT_USER_ID
    source = "Find records with value FAILED"
    generated = parameterize_filter_values({"template": query({F1: "FAILED"}), "parameters": {}})
    signature = template_signature(generated["template"])
    document = {"_id": "tpl_" + signature, "template_id": "tpl_" + signature, "template_signature": signature,
        "template": generated["template"], "source_question": source, "semantic_text": "Fetch selected rows",
        "operation": "find", "collections": [C1], "status": "active", "trusted": True,
        "positive_feedback_count": 1, "success_count": 1, "schema_spec": SCHEMA,
        "parameter_schema": describe_parameters(generated["template"], generated["parameters"], source)}
    original = deepcopy(document)
    pending = Mock()
    pending.find_one_and_update.return_value = {"question": source, "cache": {"checked": True, "hit": True}}
    service = decision_module.TemplateDecisionService()
    monkeypatch.setattr(service, "collection", lambda: pending)
    monkeypatch.setattr(QueryTemplateRepository, "get", lambda self, key: deepcopy(document))
    db = Mock()
    db.list_collection_names.return_value = [C1, C2]
    monkeypatch.setattr(decision_module, "get_database", lambda: db)
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    monkeypatch.setattr(decision_module, "active_resolution_rules", lambda: [])
    monkeypatch.setattr(decision_module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "fake"}))
    monkeypatch.setattr(decision_module.PerformanceLogger, "record", Mock())
    monkeypatch.setattr(QueryHistoryService, "save_result", Mock())
    executor = Mock(return_value={"ok": True, "row_count": 0, "all_rows": []})
    monkeypatch.setattr(decision_module.ExecutorAgent, "execute", executor)
    result = service.decide("req_fake", SimpleNamespace(action="reuse", template_id=document["template_id"]))
    assert result["execution_success"]
    assert result["query_template_id"] is None
    assert result["validated_query"]["filter"][F1] == {"$in": ["Failed", "failed"]}
    assert bind_query_parameters(result["generated_query"]["template"], result["query_parameters"]) == result["validated_query"]
    assert document == original
