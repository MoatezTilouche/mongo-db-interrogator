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
from database.user_preference_repository import UserPreferenceRepository
from services.feedback_service import FeedbackService
from services.query_history_service import QueryHistoryService
from services.user_preference_service import (
    UserPreferenceService,
    apply_preferences_to_query,
    preference_from_analysis,
)


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

    def find(self, query, projection=None):
        records = [deepcopy(record) for record in self.records.values()
                   if all(record.get(key) == value for key, value in query.items())]
        if projection and projection.get("_id") == 0:
            for record in records:
                record.pop("_id", None)
        return MemoryCursor(records)

    def update_one(self, query, update, upsert=False):
        record = self.records.get(query.get("_id")) if "_id" in query else next(
            (item for item in self.records.values()
             if all(item.get(key) == value for key, value in query.items())), None,
        )
        if record is not None and any(record.get(key) != value for key, value in query.items()):
            return SimpleNamespace(matched_count=0)
        created = record is None
        if record is None:
            if not upsert:
                return SimpleNamespace(matched_count=0)
            record = {"_id": query["_id"], **deepcopy(update.get("$setOnInsert", {}))}
            self.records[query["_id"]] = record
        record.update(deepcopy(update.get("$set", {})))
        return SimpleNamespace(matched_count=0 if created else 1,
                               upserted_id=query["_id"] if created else None)

    def update_many(self, query, update):
        modified = 0
        for record in self.records.values():
            if all(record.get(key) == value for key, value in query.items()):
                record.update(deepcopy(update.get("$set", {})))
                modified += 1
        return SimpleNamespace(modified_count=modified)


class MemoryCursor:
    def __init__(self, records):
        self.records = records

    def sort(self, key, direction):
        self.records.sort(key=lambda record: record.get(key) or datetime.min.replace(tzinfo=timezone.utc),
                          reverse=direction < 0)
        return self

    def limit(self, count):
        self.records = self.records[:count]
        return self

    def __iter__(self):
        return iter(self.records)


@pytest.fixture
def repositories():
    from agent.config import FEEDBACK_COLLECTION, QUERY_HISTORY_COLLECTION, USER_PREFERENCES_COLLECTION
    database = {QUERY_HISTORY_COLLECTION: MemoryCollection(), FEEDBACK_COLLECTION: MemoryCollection(),
                USER_PREFERENCES_COLLECTION: MemoryCollection()}
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


def preference_catalog():
    return {"collections": {"orders_proc_hist": {"fields": {
        "status": ["string"], "executionDate": ["date"], "total": ["number"],
    }}}}


def test_feedback_analysis_normalizes_contextual_latest_limit():
    analysis = FeedbackAnalysis(
        feedback_type="format_preference",
        cause=None,
        corrected_meaning=None,
        reusable_preference=True,
        preference="For failed-order requests, return only the 5 most recent failed orders.",
        scope="user",
        confidence=0.94,
    )
    normalized = FeedbackAnalysisAgent._normalize_preference_attributes(
        analysis, {"question": "Show failed orders"}, analysis.preference or "",
    )
    assert normalized.preference_type == "result_limit_preference"
    assert normalized.limit == 5
    assert normalized.latest is True
    assert normalized.apply_mode == "contextual"
    assert normalized.condition == "failed-order requests"
    assert normalized.direction == "desc"


def test_preference_normalization_preserves_structured_fields_and_identity():
    schema = preference_catalog()
    base = {
        "scope": "user", "reusable_preference": True,
        "preference_type": "result_limit_preference",
        "preference": "For failed orders, return 5 latest.",
        "collection": "orders_proc_hist", "field": "executionDate",
        "direction": "desc", "limit": 5, "latest": True,
        "apply_mode": "contextual", "condition": "failed orders",
        "confidence": 0.94,
    }
    failed = preference_from_analysis(
        {"validated_query": {"collection": "orders_proc_hist"},
         "selected_schema": schema["collections"]},
        base, "default", "req_failed", schema,
    )
    successful = preference_from_analysis(
        {"validated_query": {"collection": "orders_proc_hist"},
         "selected_schema": schema["collections"]},
        {**base, "preference": "For successful orders, return 20.",
         "condition": "successful orders", "limit": 20},
        "default", "req_success", schema,
    )
    assert failed["type"] == "result_limit_preference"
    assert failed["field"] == "executionDate"
    assert failed["latest"] is True
    assert failed["limit"] == 5
    assert failed["direction"] == -1
    assert failed["apply_mode"] == "contextual"
    assert failed["condition"] == "failed orders"
    assert failed["_id"] != successful["_id"]


def test_contextual_retrieval_and_explicit_overrides():
    collection = MemoryCollection()
    database = {"user_preferences": collection}
    repository = UserPreferenceRepository(database)
    schema = preference_catalog()
    for record in (
        preference_from_analysis(
            {"validated_query": {"collection": "orders_proc_hist"},
             "selected_schema": schema["collections"]},
            {"scope": "user", "reusable_preference": True,
             "preference_type": "result_limit_preference",
             "preference": "For failed orders, return 5 latest.",
             "collection": "orders_proc_hist", "field": "executionDate",
             "direction": "desc", "limit": 5, "latest": True,
             "apply_mode": "contextual", "condition": "failed orders", "confidence": 0.94},
            "default", "req_failed", schema,
        ),
        preference_from_analysis(
            {"validated_query": {"collection": "orders_proc_hist"},
             "selected_schema": schema["collections"]},
            {"scope": "user", "reusable_preference": True,
             "preference_type": "result_limit_preference",
             "preference": "For successful orders, return 20.",
             "collection": "orders_proc_hist", "limit": 20,
             "apply_mode": "contextual", "condition": "successful orders", "confidence": 0.94},
            "default", "req_success", schema,
        ),
    ):
        repository.upsert(record)
    service = UserPreferenceService(repository, "default")
    failed = service.retrieve_for_query("show failed orders", schema, selected_collections={"orders_proc_hist"})
    successful = service.retrieve_for_query("show successful orders", schema, selected_collections={"orders_proc_hist"})
    assert [item["limit"] for item in failed] == [5]
    assert [item["limit"] for item in successful] == [20]

    query = {"operation": "find", "collection": "orders_proc_hist", "limit": None, "sort": []}
    applied = apply_preferences_to_query(query, failed, question="show 20 failed orders")
    assert query["limit"] == 20
    assert query["sort"] == []
    assert applied == []

    query = {"operation": "find", "collection": "orders_proc_hist", "limit": None, "sort": []}
    apply_preferences_to_query(query, failed, question="show oldest failed orders")
    assert query["limit"] == 5
    assert query["sort"] == [["executionDate", 1]]


def test_legacy_limit_preference_is_normalized_at_retrieval():
    collection = MemoryCollection()
    repository = UserPreferenceRepository({"user_preferences": collection})
    now = datetime.now(timezone.utc)
    repository.upsert({
        "_id": "legacy", "user_id": "default", "scope": "user",
        "type": "format_preference", "collection": "orders_proc_hist",
        "source_collection": "orders_proc_hist", "field": None, "direction": None,
        "limit": None, "latest": False,
        "preference": "For failed-order requests, return only the 5 most recent failed orders.",
        "confidence": 0.94, "active": True, "source_request_id": "req_legacy",
        "created_at": now, "updated_at": now,
    })
    schema = preference_catalog()
    preferences = UserPreferenceService(repository, "default").retrieve_for_query(
        "show failed orders", schema, selected_collections={"orders_proc_hist"},
    )
    assert preferences[0]["type"] == "result_limit_preference"
    assert preferences[0]["limit"] == 5
    assert preferences[0]["latest"] is True
    assert preferences[0]["field"] == "executionDate"
    query = {"operation": "find", "collection": "orders_proc_hist", "limit": None, "sort": []}
    apply_preferences_to_query(query, preferences, question="show failed orders")
    assert query["limit"] == 5
    assert query["sort"] == [["executionDate", -1]]


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


def test_reusable_sort_feedback_uses_full_catalog_when_selected_schema_omits_target(monkeypatch, repositories):
    from agent.config import USER_PREFERENCES_COLLECTION
    from agents.feedback_analysis_agent import FeedbackAnalysis
    import database.schema_inspector as inspector
    queries, feedback, database = repositories
    request_id = "req_products_sort"
    queries.save({
        "request_id": request_id, "user_id": "default", "question": "Give me top 2 products",
        # QueryContext selected the incorrect field, so history lacks the correction target.
        "selected_schema": {"products": {"fields": {"price": ["int"]}}},
        "generated_query": {"operation": "find", "collection": "products", "sort": [["price", -1]]},
        "validated_query": {"operation": "find", "collection": "products", "sort": [["price", -1]]},
    })
    analysis = FeedbackAnalysis(
        feedback_type="wrong_sorting", cause="sorted by price instead of stock count",
        corrected_meaning="products should be ranked by stock count", reusable_preference=True,
        preference="rank products by stock count", scope="user", confidence=0.9,
    )
    analyzer = Mock()
    analyzer.analyze.return_value = analysis
    monkeypatch.setattr(inspector, "inspect_database_schema", lambda: {
        "collections": {"products": {"fields": {"price": ["int"], "stock": ["int"]}}},
    })
    service = FeedbackService(queries, feedback, analyzer)
    first = service.submit(request_id, False, "Rank by stock")
    second = service.submit(request_id, False, "Rank by stock")
    records = list(database[USER_PREFERENCES_COLLECTION].records.values())
    assert first["preference_saved"] and first["preference_action"] == "created"
    assert second["preference_action"] == "updated"
    assert len(records) == 1
    assert {key: records[0][key] for key in ("user_id", "type", "collection", "field", "direction")} == {
        "user_id": "default", "type": "sorting_preference", "collection": "products",
        "field": "stock", "direction": -1,
    }


def test_user_preference_retrieval_filters_unrelated_and_explicit_sort(repositories):
    from agent.config import USER_PREFERENCES_COLLECTION
    _, _, database = repositories
    repository = UserPreferenceRepository(database)
    for preference_id, collection, field in (("stock", "products", "stock"), ("name", "customers", "name")):
        repository.upsert({
            "_id": preference_id, "user_id": "default", "scope": "user", "type": "sorting_preference",
            "collection": collection, "field": field, "direction": -1,
            "preference": f"rank {collection} by {field}", "confidence": 0.9,
            "active": True, "source_request_id": preference_id,
            "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
        })
    service = UserPreferenceService(repository)
    catalog = {"collections": {
        "products": {"fields": {"stock": ["int"], "price": ["int"]}},
        "customers": {"fields": {"name": ["string"]}},
    }}
    assert [p["field"] for p in service.retrieve_for_query("Give me top 2 products", catalog)] == ["stock"]
    assert service.retrieve_for_query("Give me top 2 products by price", catalog) == []
    assert service.retrieve_for_query("Give me top 2 customers", catalog)[0]["field"] == "name"
    assert USER_PREFERENCES_COLLECTION in database


def test_retrieval_repairs_stale_field_from_preference_text(repositories):
    from agent.config import USER_PREFERENCES_COLLECTION
    _, _, database = repositories
    repository = UserPreferenceRepository(database)
    repository.upsert({
        "_id": "products_sort", "user_id": "default", "scope": "user", "type": "sorting_preference",
        "collection": "products", "field": "price", "direction": -1,
        "preference": "products.stock_count", "confidence": 0.9,
        "active": True, "source_request_id": "req_feedback",
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
    })
    catalog = {"collections": {"products": {"fields": {"stock": ["int"], "price": ["int"]}}}}
    preferences = UserPreferenceService(repository).retrieve_for_query("give me top 3 products", catalog)
    assert preferences[0]["field"] == "stock"
    assert list(database[USER_PREFERENCES_COLLECTION].records.values())[0]["field"] == "stock"
    query = {"operation": "find", "collection": "products", "sort": [["price", -1]], "limit": 3}
    assert apply_preferences_to_query(query, preferences) == [{
        "type": "sorting_preference", "collection": "products", "field": "stock", "direction": -1,
    }]
    assert query["sort"] == [["stock", -1]]


def test_learned_sort_is_passed_to_agents_applied_and_saved_in_history(monkeypatch, repositories):
    from agent.config import USER_PREFERENCES_COLLECTION
    from agents.feedback_analysis_agent import FeedbackAnalysis
    from database.user_preference_repository import UserPreferenceRepository
    import agents.query_context_agent as context_module
    import agents.query_generator_agent as generator_module
    import orchestrator as orchestrator_module

    queries, _, database = repositories
    preference_repo = UserPreferenceRepository(database)
    preference_repo.upsert({
        "_id": "products_sort", "user_id": "default", "scope": "user", "type": "sorting_preference",
        "collection": "products", "field": "stock", "direction": -1,
        "preference": "rank products by stock count", "confidence": 0.9,
        "active": True, "source_request_id": "req_feedback",
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
    })
    catalog = {"database": "test", "collections": {"products": {"fields": {
        "name": ["string"], "price": ["int"], "stock": ["int"],
    }}}}
    monkeypatch.setattr(orchestrator_module, "inspect_database_schema", lambda: catalog)
    captured = {}

    def invoke(role, messages, schema):
        captured[role] = messages[0].content
        if role == "query_context":
            return SimpleNamespace(content=json.dumps({
                "complexity": "simple", "operation": "find", "collections": ["products"],
                "fields": {"products": ["name", "price"]},
            }))
        return SimpleNamespace(content=json.dumps({
            "operation": "find", "collection": "products", "filter": {},
            "projection": None, "sort": [["price", -1]], "limit": 2,
        }))

    monkeypatch.setattr(context_module.llm_router, "invoke", invoke)
    monkeypatch.setattr(generator_module.llm_router, "invoke", invoke)
    app = orchestrator_module.OrchestratorAgent()
    app.preference_service = UserPreferenceService(preference_repo)
    app.executor.execute = Mock(return_value={
        "ok": True, "row_count": 2, "all_rows": [{"name": "A", "stock": 50}, {"name": "B", "stock": 30}],
        "execution_time_ms": 1,
    })
    response = QueryHistoryService(queries, app.run).run("Give me top 2 products")
    record = queries.get_by_request_id(response["request_id"])
    assert response["generated_query"]["sort"] == [["stock", -1]]
    assert response["applied_preferences"] == [{
        "type": "sorting_preference", "collection": "products", "field": "stock", "direction": -1,
    }]
    assert '"field":"stock"' in captured["query_context"]
    assert '"field":"stock"' in captured["query_generator"]
    assert record["applied_preferences"] == response["applied_preferences"]
    assert USER_PREFERENCES_COLLECTION in database


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
    from agent.config import FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION
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
    stored_preferences = list(database[USER_PREFERENCES_COLLECTION].records.values())
    assert stored_preferences and all(not preference["active"] for preference in stored_preferences)


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
        assert schema.model_fields is FeedbackAnalysis.model_fields
        captured.append("\n".join(message.content for message in messages))
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


def test_feedback_agent_promotes_clear_ranking_correction(monkeypatch):
    import agents.feedback_analysis_agent as module
    analysis = FeedbackAnalysis(
        feedback_type="wrong_sorting", cause="Query sorted by price instead of stock count",
        corrected_meaning="Products should be ranked by stock count", reusable_preference=False,
        preference=None, scope="one_off", confidence=0.9,
    )
    monkeypatch.setattr(module.llm_router, "invoke", lambda *args: SimpleNamespace(content=analysis.model_dump_json()))
    output = FeedbackAnalysisAgent().analyze(
        {"question": "Give me the top 2 products"},
        "Query sorted by price instead of stock count",
    )
    assert output.reusable_preference is True
    assert output.preference == "Products should be ranked by stock count"
    assert output.scope == "user"


@pytest.mark.parametrize("comment", [
    "Only this request",
    "For this report only",
    "Just this once",
])
def test_feedback_agent_keeps_explicit_one_off_ranking_correction(monkeypatch, comment):
    import agents.feedback_analysis_agent as module
    analysis = FeedbackAnalysis(
        feedback_type="wrong_sorting", cause="Sorted by price instead of stock",
        corrected_meaning="Rank products by stock", reusable_preference=False,
        preference=None, scope="one_off", confidence=0.9,
    )
    monkeypatch.setattr(module.llm_router, "invoke", lambda *args: SimpleNamespace(content=analysis.model_dump_json()))
    output = FeedbackAnalysisAgent().analyze({"question": "Show top products"}, comment)
    assert output.reusable_preference is False
    assert output.preference is None
    assert output.scope == "one_off"


def test_feedback_agent_never_promotes_potential_global_rule():
    analysis = FeedbackAnalysis(
        feedback_type="wrong_sorting", cause="Shared reporting rule", corrected_meaning="Rank by stock",
        reusable_preference=False, preference=None, scope="potential_global", confidence=0.99,
    )
    output = FeedbackAnalysisAgent._normalize_clear_ranking_correction(
        analysis, {"question": "Show top products"}, "Sort by stock",
    )
    assert output.scope == "potential_global"
    assert output.reusable_preference is False


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
    from agent.config import USER_PREFERENCES_COLLECTION
    assert set(database) == {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION}
    assert database[USER_PREFERENCES_COLLECTION].records == {}


def test_secrets_are_redacted_from_history_feedback_and_prompts(monkeypatch, repositories, analysis):
    import agents.feedback_analysis_agent as agent_module
    secret = "test-openrouter-credential-not-real"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
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
    from agent.config import QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION
    db = Mock()
    db.name = "test"
    db.list_collection_names.return_value = [QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION,
                                             USER_PREFERENCES_COLLECTION, "customers"]
    customers = Mock()
    customers.find.return_value = [{"name": "Alice"}]
    db.__getitem__ = Mock(return_value=customers)
    monkeypatch.setattr(inspector, "get_database", lambda: db)
    schema = inspector.inspect_database_schema()
    assert set(schema["collections"]) == {"customers"}
    db.__getitem__.assert_called_once_with("customers")


def test_internal_collections_are_rejected_by_validator():
    from agent.config import QUERY_HISTORY_COLLECTION, USER_PREFERENCES_COLLECTION
    from agents.validator_agent import QueryValidationError, ValidatorAgent
    schema = {"collections": {QUERY_HISTORY_COLLECTION: {"fields": {}}, "customers": {"fields": {}}}}
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate({"operation": "find", "collection": QUERY_HISTORY_COLLECTION}, schema)
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate({"operation": "aggregate", "collection": "customers", "pipeline": [{"$lookup": {"from": QUERY_HISTORY_COLLECTION}}]}, schema)
    with pytest.raises(QueryValidationError, match="Internal"):
        ValidatorAgent().validate({"operation": "find", "collection": USER_PREFERENCES_COLLECTION}, schema)


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
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-provider-secret")
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


def test_feedback_analysis_receives_only_collection_names_from_query_context(monkeypatch, analysis):
    import agents.feedback_analysis_agent as module
    context = {"complexity": "complex", "operation": "aggregate", "collections": ["orders"], "fields": {"orders": ["total"]}}
    captured = []
    monkeypatch.setattr(module.llm_router, "invoke", lambda role, messages, schema: (
        captured.append(messages[-1].content) or SimpleNamespace(content=analysis.model_dump_json())
    ))
    FeedbackAnalysisAgent().analyze({"question": "Total spending", "query_context": context}, "Wrong total")
    compact = json.loads(captured[0])
    assert compact["cols"] == ["orders"]
    assert "query_context" not in compact and "complexity" not in captured[0]


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
