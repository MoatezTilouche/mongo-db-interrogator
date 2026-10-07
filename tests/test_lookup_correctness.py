from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from agents.validator_agent import ValidatorAgent, QueryValidationError
from services.aggregation_validation import normalize_lookups
from services.mongo_capabilities import MongoCapabilities
from services.query_parameter_binder import bind_query_parameters
from agent.output_schemas import GeneratedQuery

SCHEMA = {"collections": {"orders": {"fields": {}}, "customers": {"fields": {}}}}
SIMPLE = {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}
PIPELINE = {"from": "customers", "let": {"cid": "$customer_id"}, "pipeline": [
    {"$match": {"$expr": {"$eq": ["$_id", "$$cid"]}}}], "as": "customer"}


def query(lookup):
    return {"operation": "aggregate", "collection": "orders", "pipeline": [{"$lookup": deepcopy(lookup)}]}


@pytest.mark.parametrize("lookup", [SIMPLE, PIPELINE, {"from": "customers", "as": "customer", "pipeline": []}])
def test_valid_lookup_forms(lookup):
    assert ValidatorAgent().validate(query(lookup), SCHEMA) == query(lookup)


@pytest.mark.parametrize("lookup", [
    None, [], "customers", {**SIMPLE, "pipeline": []},
    {k: v for k, v in SIMPLE.items() if k != "foreignField"},
    {k: v for k, v in SIMPLE.items() if k != "localField"},
    {**PIPELINE, "pipeline": {}}, {**PIPELINE, "let": []},
    {k: v for k, v in SIMPLE.items() if k != "from"},
    {k: v for k, v in SIMPLE.items() if k != "as"},
    {**SIMPLE, "from": " "}, {**SIMPLE, "as": 5},
    {**SIMPLE, "localField": "$customer_id"}, {**SIMPLE, "let": {}},
    {"from": "customers", "as": "customer"},
])
def test_invalid_lookup_has_structured_error(lookup):
    with pytest.raises(QueryValidationError) as caught:
        ValidatorAgent(MongoCapabilities()).validate(query(lookup), SCHEMA)
    error = caught.value.to_dict()
    assert error["code"] == "invalid_lookup_shape"
    assert error["stage_index"] == 0 and error["operator"] == "$lookup" and error["repairable"]


def test_mixed_lookup_capability_is_explicit():
    mixed = query({**SIMPLE, "pipeline": []})
    assert ValidatorAgent(MongoCapabilities(True)).validate(mixed, SCHEMA) == mixed
    assert normalize_lookups(mixed, MongoCapabilities(True)) == mixed
    assert "Never combine pipeline" in MongoCapabilities().lookup_guidance()
    assert "is supported" in MongoCapabilities(True).lookup_guidance()


@pytest.mark.parametrize("child", [[{}], [{"$match": {}, "$limit": 1}], [{"$lookup": None}], [{"$out": "x"}], "bad"])
def test_nested_lookup_pipeline_is_validated(child):
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate(query({**PIPELINE, "pipeline": child}), SCHEMA)


def test_facet_lookup_validation_and_nested_path():
    candidate = {"operation": "aggregate", "collection": "orders", "pipeline": [
        {"$facet": {"joined": [{"$lookup": {**SIMPLE, "pipeline": []}}]}}]}
    with pytest.raises(QueryValidationError) as caught:
        ValidatorAgent().validate(candidate, SCHEMA)
    assert "$facet.joined" in caught.value.to_dict()["path"]
    ValidatorAgent().validate(normalize_lookups(candidate), SCHEMA)


def test_empty_pipeline_normalization_is_lossless_and_immutable():
    original = query({**SIMPLE, "pipeline": []})
    original["pipeline"].extend([{"$unwind": "$customer"}, {"$sort": {"_id": 1}}, {"$limit": 2}])
    snapshot = deepcopy(original)
    normalized = normalize_lookups(original)
    assert normalized["pipeline"][0] == {"$lookup": SIMPLE}
    assert normalized["pipeline"][1:] == original["pipeline"][1:]
    assert original == snapshot
    assert normalize_lookups(normalized) == normalized
    ValidatorAgent().validate(normalized, SCHEMA)


@pytest.mark.parametrize("extra", [{"pipeline": [{"$match": {"active": True}}]}, {"pipeline": [], "let": {}}, {"pipeline": [], "localField": "$bad"}])
def test_ambiguous_normalization_defers_to_repair(extra):
    original = query({**SIMPLE, **extra})
    assert normalize_lookups(original) == original


@pytest.mark.parametrize("method", ["generate", "optimize", "repair"])
def test_all_llm_paths_share_capabilities_and_template_contract(monkeypatch, method):
    import agents.query_generator_agent as module
    output = {"semantic_intent": "Join related records", "template": query(PIPELINE), "parameters": {}}
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(output)))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    agent = module.QueryGeneratorAgent(MongoCapabilities())
    if method == "generate":
        result = agent.generate("Join records", SCHEMA)
    elif method == "optimize":
        result = agent.optimize("Join records", {}, SCHEMA, query(SIMPLE), "Filter related records")
    else:
        result = agent.repair("Join records", SCHEMA, query({**SIMPLE, "pipeline": []}), "Invalid lookup")
    assert "Never combine pipeline" in invoke.call_args.args[1][0].content
    assert invoke.call_args.args[2] is GeneratedQuery
    concrete = bind_query_parameters(result["template"], result["parameters"])
    assert concrete == output["template"]
    ValidatorAgent().validate(concrete, SCHEMA)


def test_raw_repair_output_does_not_match_contract():
    with pytest.raises(ValueError):
        GeneratedQuery.model_validate(query(SIMPLE))


def fresh_pipeline(monkeypatch, generated, repaired=None):
    import orchestrator as module
    monkeypatch.setattr(module.schema_bootstrap_service, "get_catalog", lambda: SCHEMA)
    monkeypatch.setattr(module.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module.schema_bootstrap_service, "retrieve", lambda *args: {"catalog": SCHEMA, "telemetry": {}})
    app = module.OrchestratorAgent()
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.retrieve.return_value = {"knowledge": [], "telemetry": {}}
    app.query_context_agent = Mock()
    app.query_context_agent.build.return_value = {"schema_spec": SCHEMA, "query_context": {
        "complexity": "complex", "operation": "aggregate", "collections": ["orders", "customers"]}}
    app.query_generator = Mock()
    from services.query_parameter_binder import template_from_concrete
    app.query_generator.generate.return_value = template_from_concrete(generated)
    app.query_generator.repair.return_value = template_from_concrete(repaired or generated)
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={
        "status": "safe", "query_signature": "sig", "estimated_rows": 0}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    return app


def test_fresh_generated_mixed_lookup_normalizes_before_execution(monkeypatch):
    app = fresh_pipeline(monkeypatch, query({**SIMPLE, "pipeline": []}))
    result = app.run("Join related records")
    assert result["execution_success"]
    assert result["validated_query"] == query(SIMPLE)
    app.query_generator.repair.assert_not_called()
    app.executor.execute.assert_called_once_with(query(SIMPLE))


def test_fresh_invalid_repair_stops_with_structured_error(monkeypatch):
    bad = query({**SIMPLE, "pipeline": [{"$match": {"active": True}}]})
    app = fresh_pipeline(monkeypatch, bad)
    result = app.run("Join related records")
    assert not result["execution_success"]
    assert result["execution_error_code"] == "invalid_lookup_shape"
    assert len(result["validation_errors"]) == 2
    app.query_generator.repair.assert_called_once()
    app.executor.execute.assert_not_called()


def test_fresh_invalid_execution_repair_stops_before_second_execution(monkeypatch):
    app = fresh_pipeline(monkeypatch, query(SIMPLE), query({**SIMPLE, "pipeline": "bad"}))
    app.executor.execute.return_value.update(ok=False, repairable=True, error="Structural failure")
    result = app.run("Join related records")
    assert not result["execution_success"]
    assert result["execution_error_code"] == "invalid_lookup_shape"
    app.query_generator.repair.assert_called_once()
    app.executor.execute.assert_called_once()


def test_repair_agent_invalid_envelope_has_safe_structured_error(monkeypatch):
    import agents.query_generator_agent as module
    monkeypatch.setattr(module.llm_router, "invoke", Mock(return_value=SimpleNamespace(content=json.dumps(query(SIMPLE)))))
    with pytest.raises(QueryValidationError) as caught:
        module.QueryGeneratorAgent().repair("Join records", SCHEMA, query(SIMPLE), "Bad structure")
    assert caught.value.to_dict()["code"] == "invalid_repair_output"
    assert not caught.value.repairable


@pytest.mark.parametrize("code,repairable", [(2, True), (9, True), (4570, True), (50, False), (13, False), (18, False), (None, False)])
def test_structural_execution_error_classification(code, repairable):
    from agents.executor_agent import is_repairable_execution_error
    from pymongo.errors import OperationFailure
    assert is_repairable_execution_error(OperationFailure("Generic command failure", code=code)) == repairable
