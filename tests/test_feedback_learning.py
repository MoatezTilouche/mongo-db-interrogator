from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from agents.feedback_analysis_agent import FeedbackAnalysis, FeedbackAnalysisAgent
from api.app import create_app
from api.routers.feedback import get_feedback_service
from api.routers.query import get_query_history_service
from database.feedback_repository import FeedbackRepository
from database.query_history_repository import QueryHistoryRepository
from services.feedback_service import FeedbackService
from services.query_history_service import QueryHistoryService


class MemoryCollection:
    """Small Mongo interface fake to exercise real repository update semantics."""
    def __init__(self):
        self.records = {}

    def insert_one(self, record):
        assert record["_id"] not in self.records
        self.records[record["_id"]] = deepcopy(record)

    def find_one(self, query, projection=None):
        record = self.records.get(query["_id"])
        if record is None:
            return None
        result = deepcopy(record)
        if projection and projection.get("_id") == 0:
            result.pop("_id")
        return result

    def update_one(self, query, update, upsert=False):
        record = self.records.get(query["_id"])
        if record is not None and any(record.get(key) != value for key, value in query.items()):
            return SimpleNamespace(matched_count=0)
        if record is None:
            if not upsert:
                return SimpleNamespace(matched_count=0)
            record = {"_id": query["_id"], **deepcopy(update.get("$setOnInsert", {}))}
            self.records[query["_id"]] = record
        record.update(deepcopy(update.get("$set", {})))
        return SimpleNamespace(matched_count=1)


@pytest.fixture
def repositories():
    from agent.config import FEEDBACK_COLLECTION, QUERY_HISTORY_COLLECTION
    database = {QUERY_HISTORY_COLLECTION: MemoryCollection(), FEEDBACK_COLLECTION: MemoryCollection()}
    return QueryHistoryRepository(database), FeedbackRepository(database), database


@pytest.fixture
def analysis():
    return FeedbackAnalysis(
        feedback_type="wrong_metric", cause="Customers were ranked by order count.",
        corrected_meaning="Rank customers by total spending.", reusable_preference=True,
        preference="Interpret top customers as ranking by total spending.", scope="user", confidence=0.95,
    )


def result(success=True):
    return {
        "execution_success": success, "execution_error": None if success else "Unknown field",
        "execution_result": [{"name": f"Customer {i}"} for i in range(10)] if success else [],
        "final_response": "Found customers." if success else None,
        "route": {"complexity": "simple"}, "query_plan": None,
        "schema_spec": {"collections": {"customers": {"fields": {"name": ["string"]}}}, "available_collections": ["private"]},
        "generated_query": {"operation": "find", "collection": "customers", "limit": 10},
        "validated_query": {"operation": "find", "collection": "customers", "limit": 10},
        "execution_metadata": {"row_count": 10 if success else 0, "execution_time_ms": 80},
        "telemetry": {"total_llm_calls": 3, "total_input_tokens": 900, "total_output_tokens": 180, "total_latency_ms": 1500},
    }


def history(history_repository):
    response = QueryHistoryService(history_repository, lambda _: result()).run("Show top customers")
    return response["request_id"]


def client_for(monkeypatch, query_service=None, feedback_service=None):
    monkeypatch.delenv("API_KEY", raising=False)
    app = create_app()
    if query_service is not None:
        app.dependency_overrides[get_query_history_service] = lambda: query_service
    if feedback_service is not None:
        app.dependency_overrides[get_feedback_service] = lambda: feedback_service
    return TestClient(app)


@pytest.mark.parametrize("success", [True, False])
def test_query_endpoint_returns_unique_id_and_persists_history(monkeypatch, repositories, success):
    repo, _, _ = repositories
    service = QueryHistoryService(repo, lambda _: result(success), preview_size=2)
    with client_for(monkeypatch, query_service=service) as client:
        first = client.post("/query", json={"question": "Give me customers"})
        second = client.post("/query", json={"question": "Give me customers"})
    assert first.status_code == second.status_code == 200
    payload = first.json()
    assert payload["request_id"].startswith("req_")
    assert payload["request_id"] != second.json()["request_id"]
    assert payload["execution_success"] is success
    record = repo.get_by_request_id(payload["request_id"])
    assert record["execution_success"] is success
    assert record["query_plan"] is None
    assert record["route"] == "simple"
    assert len(record["result_preview"]) == (2 if success else 0)
    assert record["result_metadata"]["row_count"] == (10 if success else 0)
    assert record["result_metadata"]["execution_time_ms"] == 80
    assert record["llm_usage"]["total_calls"] == 3
    assert record["created_at"].tzinfo == timezone.utc
    assert "execution_result" not in record
    assert "available_collections" not in record["selected_schema"]


def test_history_failure_does_not_break_query_or_log_secrets(monkeypatch, caplog):
    secret = "DO-NOT-LOG-this-key"
    repo = Mock()
    repo.save.side_effect = RuntimeError(secret)
    service = QueryHistoryService(repo, lambda _: result())
    with caplog.at_level(logging.INFO), client_for(monkeypatch, query_service=service) as client:
        response = client.post("/query", json={"question": "Give me customers"})
    assert response.status_code == 200
    assert response.json()["execution_success"]
    assert response.json()["request_id"]
    assert "query_history_save_failed" in caplog.text
    assert secret not in caplog.text


def test_pipeline_exception_is_persisted(repositories):
    repo, _, _ = repositories
    pipeline = Mock(side_effect=RuntimeError("private connection URI"))
    response = QueryHistoryService(repo, pipeline).run("question")
    record = repo.get_by_request_id(response["request_id"])
    assert not response["execution_success"]
    assert not record["execution_success"]
    assert record["execution_error"] == "Query pipeline failed."


@pytest.mark.parametrize("preview_size", [0, 1, 5])
def test_preview_is_bounded_by_count_and_bytes(repositories, preview_size):
    repo, _, _ = repositories
    output = result()
    output["execution_result"] = [{"large": "x" * 100000}] * 10
    response = QueryHistoryService(repo, lambda _: output, preview_size=preview_size, preview_max_bytes=100).run("question")
    preview = repo.get_by_request_id(response["request_id"])["result_preview"]
    assert len(preview) <= preview_size
    assert len(json.dumps(preview).encode()) <= 100


def test_positive_feedback_has_no_llm_call(monkeypatch, repositories):
    queries, feedback, _ = repositories
    request_id = history(queries)
    analyzer = Mock()
    service = FeedbackService(queries, feedback, analyzer)
    with client_for(monkeypatch, feedback_service=service) as client:
        response = client.post("/feedback", json={"request_id": request_id, "liked": True})
    assert response.status_code == 200
    assert response.json()["feedback_saved"]
    assert not response.json()["analysis_created"]
    record = feedback.get_by_request_id(request_id)
    assert record["liked"] is True
    assert record["analysis"] is None
    assert record["comment"] is None
    analyzer.analyze.assert_not_called()


def test_negative_feedback_is_analyzed_and_persisted(monkeypatch, repositories, analysis):
    queries, feedback, _ = repositories
    request_id = history(queries)
    analyzer = Mock()
    analyzer.analyze.return_value = analysis
    with client_for(monkeypatch, feedback_service=FeedbackService(queries, feedback, analyzer)) as client:
        response = client.post("/feedback", json={"request_id": request_id, "liked": False, "comment": "Rank by spending"})
    assert response.status_code == 200
    assert response.json()["analysis_created"]
    analyzer.analyze.assert_called_once()
    assert feedback.get_by_request_id(request_id)["analysis"] == analysis.model_dump()


@pytest.mark.parametrize("payload", [
    {"request_id": "req_unknown", "liked": False},
    {"request_id": "req_unknown", "liked": False, "comment": "   "},
    {"request_id": "req_unknown", "liked": False, "comment": None},
    {"request_id": "req_unknown", "liked": "false", "comment": "wrong"},
    {"request_id": "req_unknown", "liked": 1},
    {"liked": True},
    {"request_id": " ", "liked": True},
])
def test_invalid_feedback_is_rejected(monkeypatch, payload):
    service = Mock()
    with client_for(monkeypatch, feedback_service=service) as client:
        assert client.post("/feedback", json=payload).status_code == 422
    service.submit.assert_not_called()


def test_unknown_request_is_404(monkeypatch, repositories):
    queries, feedback, _ = repositories
    analyzer = Mock()
    with client_for(monkeypatch, feedback_service=FeedbackService(queries, feedback, analyzer)) as client:
        response = client.post("/feedback", json={"request_id": "req_unknown", "liked": True})
    assert response.status_code == 404
    assert feedback.get_by_request_id("req_unknown") is None
    analyzer.analyze.assert_not_called()


def test_duplicate_feedback_updates_one_record_and_clears_old_analysis(repositories, analysis):
    from agent.config import FEEDBACK_COLLECTION
    queries, feedback, database = repositories
    request_id = history(queries)
    analyzer = Mock()
    analyzer.analyze.return_value = analysis
    service = FeedbackService(queries, feedback, analyzer)
    service.submit(request_id, False, "Wrong metric")
    first = feedback.get_by_request_id(request_id)
    service.submit(request_id, True)
    second = feedback.get_by_request_id(request_id)
    assert len(database[FEEDBACK_COLLECTION].records) == 1
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["analysis"] is None
    assert second["liked"] is True


def test_raw_feedback_survives_analysis_failure(monkeypatch, repositories, caplog):
    queries, feedback, _ = repositories
    request_id = history(queries)
    analyzer = Mock()
    analyzer.analyze.side_effect = RuntimeError("secret provider credentials")
    with caplog.at_level(logging.INFO), client_for(monkeypatch, feedback_service=FeedbackService(queries, feedback, analyzer)) as client:
        response = client.post("/feedback", json={"request_id": request_id, "liked": False, "comment": "Wrong metric"})
    assert response.status_code == 200
    assert response.json()["feedback_saved"]
    assert not response.json()["analysis_created"]
    assert response.json()["analysis_error"]
    record = feedback.get_by_request_id(request_id)
    assert record["comment"] == "Wrong metric"
    assert record["analysis"] is None
    assert "feedback_analysis_failed" in caplog.text
    assert "secret provider credentials" not in caplog.text


def test_analysis_storage_failure_keeps_raw_feedback(repositories, analysis):
    queries, feedback, _ = repositories
    request_id = history(queries)
    analyzer = Mock()
    analyzer.analyze.return_value = analysis
    feedback.save_analysis = Mock(side_effect=RuntimeError("storage failed"))
    response = FeedbackService(queries, feedback, analyzer).submit(request_id, False, "Wrong metric")
    assert response["feedback_saved"]
    assert not response["analysis_created"]
    assert feedback.get_by_request_id(request_id)["analysis"] is None


def test_stale_analysis_cannot_overwrite_new_feedback(repositories, analysis):
    queries, feedback, _ = repositories
    request_id = history(queries)
    analyzer = Mock()
    service = FeedbackService(queries, feedback, analyzer)
    def concurrent_update(*args):
        service.submit(request_id, True)
        return analysis
    analyzer.analyze.side_effect = concurrent_update
    response = service.submit(request_id, False, "Wrong metric")
    assert not response["analysis_created"]
    assert feedback.get_by_request_id(request_id)["liked"] is True
    assert feedback.get_by_request_id(request_id)["analysis"] is None


@pytest.mark.parametrize("failing_stage", ["read", "write"])
def test_feedback_persistence_failure_is_503(monkeypatch, repositories, failing_stage):
    queries, feedback, _ = repositories
    request_id = history(queries)
    if failing_stage == "read":
        queries.get_by_request_id = Mock(side_effect=RuntimeError("secret"))
    else:
        feedback.save = Mock(side_effect=RuntimeError("secret"))
    with client_for(monkeypatch, feedback_service=FeedbackService(queries, feedback)) as client:
        response = client.post("/feedback", json={"request_id": request_id, "liked": True})
    assert response.status_code == 503
    assert "secret" not in response.text


def test_feedback_agent_uses_central_router_with_minimal_context(monkeypatch, analysis):
    import agents.feedback_analysis_agent as module
    captured = []
    def invoke(role, messages, schema):
        assert role == "feedback_analysis"
        assert schema is FeedbackAnalysis
        captured.append(messages[0].content)
        return SimpleNamespace(content=analysis.model_dump_json())
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    record = {
        "question": "Top customers", "selected_schema": {"customers": {"fields": {"name": ["string"]}}},
        "query_plan": None, "generated_query": {"operation": "find"}, "validated_query": {},
        "final_response": "Found customers", "full_schema": {"private": "FULL_SCHEMA_MARKER"},
        "result_preview": [{"data": "PREVIEW_MARKER"}], "execution_result": ["FULL_RESULT_MARKER"],
        "telemetry": {"marker": "TELEMETRY_MARKER"}, "other_users": "OTHER_USER_MARKER",
    }
    output = FeedbackAnalysisAgent().analyze(record, "Rank by spending")
    assert output == analysis
    for marker in ("FULL_SCHEMA_MARKER", "PREVIEW_MARKER", "FULL_RESULT_MARKER", "TELEMETRY_MARKER", "OTHER_USER_MARKER"):
        assert marker not in captured[0]
    assert "Rank by spending" in captured[0]


@pytest.mark.parametrize("patch", [{"confidence": 1.5}, {"feedback_type": "invented"}, {"scope": "global"}, {"reusable_preference": "true"}, {"extra": "field"}])
def test_feedback_analysis_contract_is_strict(analysis, patch):
    with pytest.raises(ValidationError):
        FeedbackAnalysis.model_validate({**analysis.model_dump(), **patch})


def test_potential_global_is_stored_without_changing_business_data(repositories, analysis):
    from agent.config import QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION
    queries, feedback, database = repositories
    request_id = history(queries)
    analyzer = Mock()
    candidate = analysis.model_copy(update={"scope": "potential_global", "reusable_preference": False, "preference": None})
    analyzer.analyze.return_value = candidate
    original = deepcopy(database[QUERY_HISTORY_COLLECTION].records)
    FeedbackService(queries, feedback, analyzer).submit(request_id, False, "Exclude refunds")
    assert feedback.get_by_request_id(request_id)["analysis"]["scope"] == "potential_global"
    assert database[QUERY_HISTORY_COLLECTION].records == original
    assert set(database) == {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION}


def test_secrets_are_redacted_from_history_feedback_and_prompts(monkeypatch, repositories, analysis):
    import agents.feedback_analysis_agent as agent_module
    secret = "test-openrouter-credential-not-real"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("API_KEY", "test-auth-credential-not-real")
    queries, feedback, _ = repositories
    output = result()
    output.update(final_response=secret, execution_error=secret)
    output["execution_result"] = [{"name": "Alice", "password": "hidden-db-password", "authorization": "hidden-auth", "note": secret}]
    output["generated_query"]["filter"] = {"api_key": secret}
    response = QueryHistoryService(queries, lambda _: output).run(f"question {secret}")
    record = queries.get_by_request_id(response["request_id"])
    text = json.dumps(record, default=str)
    assert secret not in text
    assert "hidden-db-password" not in text
    assert "hidden-auth" not in text
    captured = []
    def invoke(role, messages, schema):
        captured.append(messages[0].content)
        return SimpleNamespace(content=analysis.model_dump_json())
    monkeypatch.setattr(agent_module.llm_router, "invoke", invoke)
    FeedbackService(queries, feedback).submit(response["request_id"], False, f"wrong {secret}")
    assert secret not in captured[0]
    assert secret not in json.dumps(feedback.get_by_request_id(response["request_id"]), default=str)


def test_internal_collections_are_excluded_from_schema(monkeypatch):
    import database.schema_inspector as inspector
    from agent.config import QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION
    db = Mock()
    db.name = "test"
    db.list_collection_names.return_value = [QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, "customers"]
    customers = Mock()
    customers.find.return_value = [{"name": "Alice"}]
    db.__getitem__ = Mock(return_value=customers)
    monkeypatch.setattr(inspector, "get_database", lambda: db)
    schema = inspector.inspect_database_schema()
    assert set(schema["collections"]) == {"customers"}
    db.__getitem__.assert_called_once_with("customers")


def test_internal_collections_are_rejected_by_validator():
    from agent.config import QUERY_HISTORY_COLLECTION
    from agents.validator_agent import QueryValidationError, ValidatorAgent
    schema = {"collections": {QUERY_HISTORY_COLLECTION: {"fields": {}}, "customers": {"fields": {}}}}
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate({"operation": "find", "collection": QUERY_HISTORY_COLLECTION}, schema)
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate({"operation": "aggregate", "collection": "customers", "pipeline": [{"$lookup": {"from": QUERY_HISTORY_COLLECTION}}]}, schema)


def test_feedback_endpoint_keeps_existing_api_key_protection(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-protected-key")
    with TestClient(create_app()) as client:
        assert client.post("/feedback", json={"request_id": "req_x", "liked": True}).status_code == 401


def test_short_credentials_do_not_corrupt_request_ids(monkeypatch, repositories):
    monkeypatch.setenv("MONGODB_URI", "mongodb://user:a@localhost/test")
    queries, _, _ = repositories
    request_id = history(queries)
    assert queries.get_by_request_id(request_id) is not None


def test_untrusted_request_id_cannot_log_configured_secrets(monkeypatch, repositories, caplog):
    from services.feedback_service import HistoryNotFoundError
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-provider-secret")
    queries, feedback, _ = repositories
    with caplog.at_level(logging.INFO), pytest.raises(HistoryNotFoundError):
        FeedbackService(queries, feedback).submit("test-not-a-real-provider-secret", True)
    assert "test-not-a-real-provider-secret" not in caplog.text


def test_query_id_is_available_to_llm_logs_and_context_is_reset(repositories):
    from agent.telemetry import current_request_id
    queries, _, _ = repositories
    captured = []
    def pipeline(question):
        captured.append(current_request_id())
        return result()
    response = QueryHistoryService(queries, pipeline).run("question")
    assert captured == [response["request_id"]]
    assert current_request_id() is None


def test_query_context_failure_is_persisted(monkeypatch, repositories):
    import orchestrator as module
    queries, _, _ = repositories
    schema = {"collections": {"customers": {"fields": {"name": ["string"]}}}}
    app = module.OrchestratorAgent()
    monkeypatch.setattr(module, "inspect_database_schema", lambda: schema)
    app.query_context_agent = Mock()
    app.query_context_agent.build.side_effect = RuntimeError("context failed")
    response = QueryHistoryService(queries, app.run).run("Top customers")
    record = queries.get_by_request_id(response["request_id"])
    assert not record["execution_success"]
    assert record["route"] is None
    assert record["selected_schema"] == {}


def test_query_context_is_saved_in_history(repositories):
    queries, _, _ = repositories
    context = {"complexity": "simple", "operation": "find", "collections": ["customers"],
               "fields": {"customers": ["name"]}}
    output = result()
    output["query_context"] = context
    output["execution_path"] = {"agents_called": ["query_context", "query_generator"],
                                "summary_skipped": True, "repair_used": False}
    response = QueryHistoryService(queries, lambda _: output).run("Give me 2 customers")
    record = queries.get_by_request_id(response["request_id"])
    assert record["query_context"] == context
    assert record["query_plan"] is None
    assert record["agents_called"] == ["query_context", "query_generator"]
    assert record["summary_skipped"] is True


def test_feedback_analysis_receives_query_context(monkeypatch, analysis):
    import agents.feedback_analysis_agent as module
    context = {"complexity": "complex", "operation": "aggregate", "collections": ["orders"], "fields": {"orders": ["total"]}}
    captured = []
    monkeypatch.setattr(module.llm_router, "invoke", lambda role, messages, schema: (
        captured.append(messages[0].content) or SimpleNamespace(content=analysis.model_dump_json())
    ))
    FeedbackAnalysisAgent().analyze({"question": "Total spending", "query_context": context}, "Wrong total")
    assert json.loads(captured[0].split("Context: ", 1)[1].split("\n", 1)[0])["query_context"] == context


def test_executor_reports_actual_execution_latency(monkeypatch):
    import agents.executor_agent as module
    database = {"customers": Mock()}
    database["customers"].count_documents.return_value = 42
    monkeypatch.setattr(module, "get_database", lambda: database)
    executed = module.ExecutorAgent().execute({"operation": "count_documents", "collection": "customers"})
    assert executed["ok"]
    assert executed["all_rows"] == [{"count": 42}]
    assert executed["execution_time_ms"] >= 0


def test_nested_join_cannot_read_internal_feedback():
    from agent.config import FEEDBACK_COLLECTION
    from agents.validator_agent import QueryValidationError, ValidatorAgent
    query = {"operation": "aggregate", "collection": "customers", "pipeline": [
        {"$facet": {"data": [{"$lookup": {"from": FEEDBACK_COLLECTION}}]}}
    ]}
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate(query, {"collections": {"customers": {"fields": {}}}})
