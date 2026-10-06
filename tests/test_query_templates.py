from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from bson import ObjectId
from agent.output_schemas import GeneratedQuery
from services.query_parameter_binder import bind_query_parameters, reconcile_template, parameterize_filter_values
from database.query_template_repository import QueryTemplateRepository, template_signature


def generated(limit=3, status="Failed"):
    return {"template": {"operation": "find", "collection": "orders",
        "filter": {"status": status}, "projection": {"_id": 0, "description": 1},
        "sort": [["executionDate", -1]], "limit": {"$param": "limit"}, "skip": 0},
        "parameters": {"limit": limit}}


def test_binding_and_reuse():
    three, ten = generated(), generated(10)
    assert template_signature(three["template"]) == template_signature(ten["template"])
    assert template_signature(three["template"]) != template_signature(generated(status="Successful")["template"])
    for item in (three, ten):
        before = deepcopy(item)
        bound = bind_query_parameters(**item)
        assert type(bound["limit"]) is int
        assert bound["limit"] == item["parameters"]["limit"]
        assert bound["filter"] == {"status": "Failed"}
        assert item == before
        assert GeneratedQuery.model_validate(item).template.collection == "orders"


def test_filter_values_share_signature_and_preserve_current_values():
    templates = []
    for command_id, status in [("CMD-000009350132", "Failed"), ("CMD-000009350133", "Successful")]:
        item = generated()
        item["template"]["filter"] = {"commandId": command_id, "status": status}
        output = parameterize_filter_values(item)
        assert output["parameters"] == {"limit": 3, "filter_commandId": command_id, "filter_status": status}
        assert output["template"]["filter"] == {"commandId": {"$param": "filter_commandId"}, "status": {"$param": "filter_status"}}
        assert bind_query_parameters(**output)["filter"] == item["template"]["filter"]
        assert parameterize_filter_values(output) == output
        GeneratedQuery.model_validate(output)
        templates.append(output["template"])
    assert template_signature(templates[0]) == template_signature(templates[1])
    collection = Mock()
    repository = QueryTemplateRepository({"_system_query_templates": collection})
    for command_id in ("CMD-000009350132", "CMD-000009350133"):
        item = {"template": {"operation": "find", "collection": "orders", "filter": {"commandId": command_id}}, "parameters": {}}
        repository.record_success(parameterize_filter_values(item), "req_test")
    calls = collection.update_one.call_args_list
    assert calls[0].args[0] == calls[1].args[0]
    assert calls[0].args[1]["$setOnInsert"]["parameter_schema"] == {"filter_commandId": {"type": "string"}}
    assert "CMD-" not in json.dumps(calls[0].args[1], default=str)


def test_filter_types_operators_and_nested_matches():
    oid, date = ObjectId(), datetime.now(timezone.utc)
    predicates = {"$and": [{"amount": {"$gte": 12.5, "$lt": 100}}, {"status": {"$in": ["Failed", "Successful"]}}],
        "enabled": True, "deleted": None, "id": oid, "date": date,
        "name": {"$regex": "^CMD-", "$options": "i"}, "present": {"$exists": True},
        "items": {"$elemMatch": {"quantity": {"$gt": 2}}},
        "scores": {"$elemMatch": {"$gt": 3}},
        "$expr": {"$eq": ["$commandId", "$$commandId"]}}
    concrete = {"operation": "aggregate", "collection": "orders", "pipeline": [
        {"$match": predicates}, {"$lookup": {"from": "other", "let": {"commandId": "$commandId"},
            "pipeline": [{"$match": {"status": "Failed", "$expr": {"$eq": ["$id", "$$commandId"]}}}], "as": "joined"}},
        {"$facet": {"active": [{"$match": {"enabled": True}}]}}, {"$sort": {"amount": -1}}]}
    output = parameterize_filter_values({"template": concrete, "parameters": {}})
    assert bind_query_parameters(**output) == concrete
    assert output["template"]["pipeline"][0]["$match"]["$expr"] == predicates["$expr"]
    assert output["template"]["pipeline"][0]["$match"]["present"] == {"$exists": True}
    assert output["template"]["pipeline"][0]["$match"]["name"]["$options"] == "i"
    assert output["template"]["pipeline"][-1] == concrete["pipeline"][-1]
    assert any(isinstance(value, ObjectId) for value in output["parameters"].values())
    assert any(isinstance(value, datetime) for value in output["parameters"].values())


@pytest.mark.parametrize("value", [{"$ne": None}, {"$where": "code"}, {"$param": "other"}, [{"$gt": 0}]])
def test_filter_operator_injection_rejected(value):
    with pytest.raises(ValueError):
        bind_query_parameters({"filter": {"commandId": {"$param": "command_id"}}}, {"command_id": value})


@pytest.mark.parametrize("template", [
    {"filter": {"$param": "filter"}},
    {"filter": {"status": {"$exists": {"$param": "value"}}}},
    {"pipeline": [{"$match": {"$expr": {"$param": "value"}}}]},
    {"projection": {"description": {"$param": "value"}}},
])
def test_filter_structure_cannot_be_parameterized(template):
    with pytest.raises(ValueError): bind_query_parameters(template, {"filter": {}, "value": True})


def test_parameter_names_are_independent_of_llm_names():
    first = {"template": {"filter": {"commandId": {"$param": "command_id"}}}, "parameters": {"command_id": "CMD-1"}}
    second = {"template": {"filter": {"commandId": {"$param": "id"}}}, "parameters": {"id": "CMD-2"}}
    assert parameterize_filter_values(first)["template"] == parameterize_filter_values(second)["template"]


def test_filter_parameterization_does_not_add_limit():
    from services.query_parameter_binder import template_from_concrete
    output = template_from_concrete({"operation": "find", "collection": "orders", "filter": {"commandId": "CMD-1"}, "limit": None})
    assert "limit" not in output["parameters"]
    assert output["template"]["limit"] is None


def test_filter_missing_parameter_and_invalid_membership():
    with pytest.raises(ValueError, match="Missing parameter"):
        bind_query_parameters({"filter": {"commandId": {"$param": "command_id"}}}, {})
    with pytest.raises(ValueError, match="Membership"):
        bind_query_parameters({"filter": {"status": {"$in": {"$param": "statuses"}}}}, {"statuses": "Failed"})


def test_extended_json_and_collision_names():
    query = {"filter": {"id": {"$oid": "507f1f77bcf86cd799439011"},
        "date": {"$date": "2026-10-06T00:00:00Z"}, "a.b": 1, "a_b": 2}}
    output = parameterize_filter_values({"template": query, "parameters": {}})
    assert bind_query_parameters(**output) == query
    reversed_query = {"filter": dict(reversed(list(query["filter"].items())))}
    reordered = parameterize_filter_values({"template": reversed_query, "parameters": {}})
    assert template_signature(output["template"]) == template_signature(reordered["template"])


@pytest.mark.parametrize("template,parameters", [
    ({"limit": {"$param": "limit"}}, {}),
    ({"limit": {"$param": "limit", "extra": 1}}, {"limit": 3}),
    ({"limit": {"$param": 3}}, {"limit": 3}),
    ({"filter": {"status": {"$param": "limit"}}}, {"limit": 3}),
    ({"sort": [[{"$param": "limit"}, -1]]}, {"limit": 3}),
    ({"skip": {"$param": "limit"}}, {"limit": 3}),
    ({"limit": "{limit}"}, {"limit": 3}),
    ({"limit": 3}, {"limit": 3}),
    ({"limit": {"$param": "limit"}}, {"limit": "3"}),
    ({"limit": {"$param": "limit"}}, {"limit": True}),
    ({"limit": {"$param": "limit"}}, {"limit": 0}),
    ({"limit": {"$param": "limit"}}, {"limit": -1}),
])
def test_invalid_binding(template, parameters):
    with pytest.raises(ValueError): bind_query_parameters(template, parameters)


def test_bson_types_and_aggregation():
    oid, date = ObjectId(), datetime.now(timezone.utc)
    template = {"filter": {"id": oid, "date": date}, "pipeline": [{"$limit": {"$param": "limit"}}]}
    result = bind_query_parameters(template, {"limit": 3})
    assert result["filter"] == {"id": oid, "date": date}
    assert result["pipeline"] == [{"$limit": 3}]


def test_atomic_deduplication():
    collection = Mock()
    repository = QueryTemplateRepository({"_system_query_templates": collection})
    assert repository.record_success(generated(), "req_3") == repository.record_success(generated(10), "req_10")
    first, second = collection.update_one.call_args_list
    assert first.args[0] == second.args[0]
    assert first.args[1]["$inc"] == {"success_count": 1}
    assert second.kwargs == {"upsert": True}
    assert "parameters" not in first.args[1]["$setOnInsert"]
    assert first.args[1]["$setOnInsert"]["parameter_schema"] == {"limit": {"type": "integer"}}


def test_pipeline_concrete_boundaries(monkeypatch):
    import orchestrator as module
    schema = {"collections": {"orders": {"fields": {"status": ["string"], "description": ["string"], "executionDate": ["date"]}}}}
    bootstrap = Mock()
    bootstrap.get_catalog.return_value = schema
    bootstrap.status.return_value = {}
    bootstrap.retrieve.return_value = {"catalog": schema, "telemetry": {}}
    monkeypatch.setattr(module, "schema_bootstrap_service", bootstrap)
    persist = Mock(return_value="tpl_test")
    monkeypatch.setattr(module, "persist_successful_template", persist)
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    app = module.OrchestratorAgent()
    app.preference_service.retrieve_for_query = Mock(return_value=[])
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.retrieve.return_value = {"knowledge": [], "telemetry": {}}
    app.query_context_agent.build = Mock(return_value={"query_context": {"complexity": "simple", "operation": "find", "collections": ["orders"]}, "schema_spec": schema})
    app.query_generator.generate = Mock(return_value=parameterize_filter_values(generated()))
    validate = app.validator.validate
    seen = []
    app.validator.validate = lambda query, schema: (seen.append(deepcopy(query)) or validate(query, schema))
    monkeypatch.setattr(module.QueryCostGuard, "analyze", lambda self, query: (seen.append(deepcopy(query)) or {"status": "safe", "query_signature": "sig"}))
    app.executor.execute = lambda query: (seen.append(deepcopy(query)) or {"ok": True, "all_rows": [], "row_count": 0})
    result = app.run("Give me the latest 3 failed order descriptions")
    assert result["execution_success"]
    assert len(seen) == 3
    assert all(query["limit"] == 3 and "$param" not in json.dumps(query) for query in seen)
    assert result["query_template_id"] == "tpl_test"
    assert result["query_parameters"] == {"limit": 3, "filter_status": "Failed"}
    persist.assert_called_once()
    app.executor.execute = lambda query: {"ok": False, "all_rows": [], "row_count": 0}
    app.run("Give me the latest 3 failed order descriptions")
    persist.assert_called_once()


def test_default_and_reconciliation():
    item = generated()
    concrete = bind_query_parameters(**item)
    concrete["projection"] = {"description": 1}
    updated = reconcile_template(item, concrete)
    assert updated["parameters"] == {"limit": 3}
    assert updated["template"]["projection"] == {"description": 1}
    default = {"template": {"operation": "find", "collection": "orders", "limit": None}, "parameters": {}}
    assert bind_query_parameters(**default)["limit"] is None


def test_internal_collection_excluded():
    from services.schema_retriever import SchemaRetriever
    from agents.validator_agent import ValidatorAgent, QueryValidationError
    schema = {"collections": {"orders": {}, "_system_query_templates": {}, "v2DB._system_query_templates": {}}}
    assert list(SchemaRetriever(schema).collections) == ["orders"]
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({"operation": "find", "collection": "_system_query_templates"}, schema)
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({"operation": "find", "collection": "v2DB._system_query_templates"}, schema)


@pytest.mark.parametrize("natural", [False, True])
def test_optimization_reuses_context_and_binds(monkeypatch, natural):
    import services.query_optimization_service as module
    original = generated(100)
    schema = {"collections": {"orders": {"fields": {"status": ["string"], "executionDate": ["date"]}}}}
    pending = {"state": "pending", "question": "Latest 100 failed orders", "result": {
        "validated_query": bind_query_parameters(**original), "generated_query": original,
        "schema_spec": schema, "query_context": {"collections": ["orders"]},
        "cost_analysis": {"estimated_rows": 100}, "suggestions": [{"id": "latest_100"}, {"id": "refine"}]}}
    collection = Mock()
    collection.find_one_and_update.return_value = pending
    service = module.QueryOptimizationService()
    monkeypatch.setattr(service, "collection", lambda: collection)
    monkeypatch.setattr(service, "save_history", Mock())
    optimize = Mock(return_value=parameterize_filter_values(generated(20)))
    monkeypatch.setattr(module.QueryGeneratorAgent, "optimize", optimize)
    cost = Mock(return_value={"status": "safe", "query_signature": "sig", "estimated_rows": 20})
    monkeypatch.setattr(module.QueryCostGuard, "analyze", cost)
    executor = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0, "execution_time_ms": 1})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    persist = Mock(return_value="tpl_reused")
    monkeypatch.setattr(module, "persist_successful_template", persist)
    result = service.continue_query("req_test", SimpleNamespace(optimization="refine" if natural else "latest_100", value=20, page=1, refinement="Latest 20"))
    assert result["execution_success"]
    assert result["generated_query"]["template"]["limit"] == {"$param": "limit"}
    assert result["query_parameters"] == {"limit": 20, "filter_status": "Failed"}
    assert result["query_template_id"] == "tpl_reused"
    assert cost.call_args.args[-1]["limit"] == 20
    assert executor.call_args.args[-1]["limit"] == 20
    assert "$param" not in json.dumps(executor.call_args.args[-1])
    assert optimize.call_count == int(natural)
    if natural:
        assert optimize.call_args.args[2] == schema
        assert optimize.call_args.args[1] == pending["result"]["query_context"]
    persist.assert_called_once()


def test_generator_modes_return_templates(monkeypatch):
    import agents.query_generator_agent as module
    output = parameterize_filter_values(generated())
    captured = []
    def invoke(role, messages, schema):
        assert schema is GeneratedQuery
        captured.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(output))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    agent = module.QueryGeneratorAgent()
    schema = {"collections": {"orders": {"fields": {"status": ["string"]}}}}
    assert agent.generate("Latest 3 failed orders", schema) == output
    assert agent.repair("Latest 3 failed orders", schema, bind_query_parameters(**output), "bad projection") == output
    assert agent.optimize("Latest 10 failed orders", {}, schema, bind_query_parameters(**output), "Latest 3") == output
    assert all('"$param":"limit"' in prompt for prompt in captured)
