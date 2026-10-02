import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import orchestrator as module
from orchestrator import OrchestratorAgent
from agents.schema_agent import SchemaAgent
from agents.query_generator_agent import QueryGeneratorAgent
from agents.query_router import QueryRouter
from agents.query_context_agent import QueryContextAgent
from agent.llm import LLMRouter
from agent.telemetry import capture_usage, totals

CATALOG = {"collections": {"customers": {"fields": {"_id": ["ObjectId"], "name": ["string"], "country": ["string"]}, "sample_documents": [{"secret": "PRIVATE_SAMPLE"}]}, "orders": {"fields": {"customerId": ["ObjectId"], "total": ["float"]}}}}
SELECTED = {"collections": {"customers": {"fields": CATALOG["collections"]["customers"]["fields"]}}}

def pipeline(monkeypatch, complex=False, operation="find"):
    monkeypatch.setattr(module, "inspect_database_schema", lambda: CATALOG)
    app = OrchestratorAgent()
    app.query_context_agent = Mock()
    context = {"complexity": "complex" if complex else "simple", "operation": "aggregate" if complex else operation,
               "collections": list(SELECTED["collections"]), "fields": {name: list(info["fields"]) for name, info in SELECTED["collections"].items()}}
    app.query_context_agent.build.return_value = {"query_context": context, "schema_spec": SELECTED}
    app.query_generator = Mock()
    query = {"operation": "aggregate" if complex else ("count_documents" if operation == "count" else operation), "collection": "customers", "filter": {}, "limit": 2}
    if complex:
        query["pipeline"] = [{"$lookup": {"from": "customers", "as": "joined", "localField": "_id", "foreignField": "_id"}}]
    app.query_generator.generate.return_value = query
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 2, "all_rows": [{"name": "Alice"}, {"name": "Sara"}], "sample_rows": []}
    app.summary_agent = Mock()
    app.summary_agent.summarize.return_value = {"summary_text": "Analysis"}
    return app

@pytest.mark.parametrize("question,operation", [("Give me 2 customers", "find"), ("Find customers from Tunisia", "find"), ("Count all customers", "count"), ("Show the 5 most expensive products", "find")])
def test_simple_uses_two_agents_and_skips_summary(monkeypatch, question, operation):
    app = pipeline(monkeypatch, operation=operation)
    if operation == "count":
        app.executor.execute.return_value.update(all_rows=[{"count": 42}], row_count=1)
    if "expensive" in question:
        app.query_generator.generate.return_value.update(sort=[["name", -1]], limit=5)
    result = app.run(question)
    assert result["execution_success"]
    app.query_context_agent.build.assert_called_once()
    app.summary_agent.summarize.assert_not_called()
    assert app.query_generator.generate.call_args.kwargs["operation_hint"] == ("count" if operation == "count" else "find")
    assert result["execution_path"]["agents_called"] == ["query_context", "query_generator"]
    assert len(result["execution_path"]["agents_called"]) == 2

@pytest.mark.parametrize("question", ["Top customers by spending across orders", "Join customers and orders with $lookup"])
def test_complex_uses_context_and_generator_without_planner(monkeypatch, question):
    app = pipeline(monkeypatch, complex=True)
    result = app.run(question)
    assert result["execution_success"]
    app.query_context_agent.build.assert_called_once()
    app.query_generator.generate.assert_called_once()
    assert result["query_plan"] is None
    app.summary_agent.summarize.assert_called_once()


def test_selection_and_prompt_boundaries(monkeypatch):
    import agents.schema_agent as schema_module
    import agents.query_generator_agent as query_module
    captured = []
    def select(role, messages, schema):
        captured.append(messages[0].content)
        return SimpleNamespace(content=json.dumps({"collections": ["customers"], "fields": {"customers": ["_id", "name"]}}))
    monkeypatch.setattr(schema_module.llm_router, "invoke", select)
    selected = SchemaAgent().identify_collections("Give me 2 customers", CATALOG)
    assert list(selected["collections"]) == ["customers"]
    assert "PRIVATE_SAMPLE" not in captured[0]
    def generate(role, messages, schema):
        captured.append(messages[0].content)
        return SimpleNamespace(content='{"operation":"find","collection":"customers"}')
    monkeypatch.setattr(query_module.llm_router, "invoke", generate)
    agent = QueryGeneratorAgent()
    agent.generate("Give me 2 customers", selected, plan={"private": "do not forward"}, operation_hint="find")
    assert "orders" not in captured[-1]
    assert "Plan:" not in captured[-1]
    assert "PRIVATE_SAMPLE" not in captured[-1]
    assert "do not forward" not in captured[-1]
    agent.repair("Give me 2 customers", selected, {"operation": "find"}, "Unknown field")
    assert "Plan:" not in captured[-1]
    assert "orders" not in captured[-1]
    assert "Unknown field" in captured[-1]


def test_query_context_combines_routing_and_selection_and_drops_unknown_fields(monkeypatch):
    import agents.query_context_agent as context_module
    captured = []
    response = {"complexity": "simple", "operation": "find", "collections": ["customers"],
                "fields": {"customers": ["name", "notInCatalog"]}}
    def invoke(role, messages, schema):
        assert role == "query_context"
        captured.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(response))
    monkeypatch.setattr(context_module.llm_router, "invoke", invoke)
    built = QueryContextAgent().build("Give me 2 customers", CATALOG)
    assert built["query_context"]["complexity"] == "simple"
    assert built["query_context"]["operation"] == "find"
    assert built["query_context"]["collections"] == ["customers"]
    assert built["query_context"]["fields"]["customers"] == ["name"]
    assert list(built["schema_spec"]["collections"]) == ["customers"]
    assert "PRIVATE_SAMPLE" not in captured[0]
    assert "orders" in captured[0]  # Full compact catalog is context-agent-only.


def test_query_context_complex_cross_collection_selection(monkeypatch):
    import agents.query_context_agent as context_module
    response = {"complexity": "complex", "operation": "aggregate", "collections": ["orders", "customers"],
                "fields": {"orders": ["customerId", "total"], "customers": ["name"]}}
    monkeypatch.setattr(context_module.llm_router, "invoke", lambda *args: SimpleNamespace(content=json.dumps(response)))
    built = QueryContextAgent().build("Top customers by order spending", CATALOG)
    assert set(built["schema_spec"]["collections"]) == {"orders", "customers"}
    assert built["query_context"]["complexity"] == "complex"
    assert built["query_context"]["operation"] == "aggregate"


def test_multiple_selected_collections_are_promoted_without_a_planner(monkeypatch):
    import agents.query_context_agent as context_module
    response = {"complexity": "simple", "operation": "find", "collections": ["customers", "orders"],
                "fields": {"customers": ["name"], "orders": ["total"]}}
    monkeypatch.setattr(context_module.llm_router, "invoke", lambda *args: SimpleNamespace(content=json.dumps(response)))
    built = QueryContextAgent().build("Customers with their order totals", CATALOG)
    assert built["query_context"]["complexity"] == "complex"
    assert built["query_context"]["operation"] == "aggregate"


def test_unknown_selection_does_not_expand_catalog(monkeypatch):
    import agents.schema_agent as schema_module
    monkeypatch.setattr(schema_module.llm_router, "invoke", lambda *args: SimpleNamespace(content='{"collections":["missing"],"fields":{"missing":[]}}'))
    with pytest.raises(ValueError):
        SchemaAgent().identify_collections("question", CATALOG)


def test_semantic_router_contract(monkeypatch):
    import agents.query_router as router_module
    route = dict(complexity="simple", operation="find", needs_planner=False, needs_schema_selection=True, needs_summary=False)
    monkeypatch.setattr(router_module.llm_router, "invoke", lambda *args: SimpleNamespace(content=json.dumps(route)))
    assert QueryRouter().route("Show the most expensive products") == route


def test_telemetry_counts_retries_and_tokens(monkeypatch):
    import agent.llm as llm_module
    monkeypatch.setattr(llm_module, "LLM_ROLE_CONFIG", {"summary": "openrouter/free"})
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 1)
    router = LLMRouter()
    client = Mock()
    client.invoke.side_effect = [TimeoutError("timed out"), SimpleNamespace(content="answer", usage_metadata={"input_tokens":12,"output_tokens":3})]
    monkeypatch.setattr(router, "_client", lambda model: client)
    with capture_usage() as records:
        router.invoke("summary", [])
    usage = totals(records)
    assert usage["total_llm_calls"] == 2
    assert usage["total_input_tokens"] == 12
    assert usage["total_output_tokens"] == 3
    assert not records[-1]["fallback_used"]
    assert usage["total_latency_ms"] >= 0


def test_connection_error_does_not_repair(monkeypatch):
    app = pipeline(monkeypatch)
    app.executor.execute.return_value = {"ok": False, "error": "connection failed"}
    result = app.run("Give me 2 customers")
    assert not result["execution_success"]
    app.query_generator.repair.assert_not_called()


def test_validation_repair_is_bounded(monkeypatch):
    app = pipeline(monkeypatch)
    app.query_generator.generate.return_value = {"operation": "find", "collection": "missing"}
    app.query_generator.repair.return_value = {"operation": "find", "collection": "customers"}
    app.executor.execute.return_value = {"ok": False, "error": "invalid query", "repairable": True}
    result = app.run("Give me 2 customers")
    assert not result["execution_success"]
    app.query_generator.repair.assert_called_once()


def test_generated_aggregation_does_not_invoke_planner(monkeypatch):
    app = pipeline(monkeypatch)
    app.query_generator.generate.return_value = {"operation": "aggregate", "collection": "customers", "pipeline": []}
    result = app.run("Analyze customers")
    assert result["execution_success"]
    app.query_context_agent.build.assert_called_once()
    app.query_generator.generate.assert_called_once()
    assert result["query_plan"] is None
    app.summary_agent.summarize.assert_called_once()


def test_complex_context_can_select_multiple_collections_without_planner(monkeypatch):
    app = pipeline(monkeypatch)
    app.query_context_agent.build.return_value = {
        "query_context": {"complexity": "complex", "operation": "aggregate", "collections": list(CATALOG["collections"]),
                          "fields": {name: list(data["fields"]) for name, data in CATALOG["collections"].items()}},
        "schema_spec": {"collections": CATALOG["collections"]},
    }
    result = app.run("Customers and their orders")
    assert result["execution_success"]
    app.query_context_agent.build.assert_called_once()
    app.query_generator.generate.assert_called_once()
    assert result["query_plan"] is None


def test_deterministic_limit_correction_skips_repair(monkeypatch):
    app = pipeline(monkeypatch)
    app.query_generator.generate.return_value["limit"] = 100000
    result = app.run("Give me customers")
    assert result["execution_success"]
    from agent.config import MAX_LIMIT
    assert result["validated_query"]["limit"] == MAX_LIMIT
    app.query_generator.repair.assert_not_called()
