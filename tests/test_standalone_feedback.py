"""Standalone learning reaches normal retrieval without questions or chat calls."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from agent.config import DEFAULT_USER_ID
from agent.llm import llm_router
from api.app import create_app
from api.routers.feedback import get_standalone_feedback_service
from services.knowledge_ingestion_service import KnowledgeIngestionService
from services.knowledge_retriever import KnowledgeRetriever
from services.standalone_feedback_service import StandaloneFeedbackService
from services.user_preference_service import UserPreferenceService, apply_preferences_to_query


CATALOG = {"collections": {
    "orders": {"fields": {"_id": ["ObjectId"], "status": ["string"], "createdAt": ["date"], "total": ["number"]}},
    "customers": {"fields": {"name": ["string"]}},
}}
PREFERENCE = {"scope": "user", "type": "result_limit_preference", "text": "Return 5 latest failed orders.",
              "collection": "orders", "field": "createdAt", "limit": 5, "latest": True,
              "apply_mode": "contextual", "condition": "failed orders"}
RULE = {"scope": "database", "type": "business_rule", "text": "Exclude refunded orders in revenue.",
        "collections": ["orders"], "fields": ["status"]}


class Preferences:
    def __init__(self):
        self.records = {}

    def upsert(self, record):
        action = "updated" if record["_id"] in self.records else "created"
        self.records[record["_id"]] = deepcopy(record)
        return action

    def list_active(self, user_id, limit):
        return [deepcopy(record) for record in self.records.values()
                if record["user_id"] == user_id and record["active"]][:limit]


class Knowledge:
    def __init__(self):
        self.records = {}

    def get_by_key(self, key):
        return deepcopy(self.records.get(key))

    def add_or_confirm(self, record):
        key = record["_id"]
        if key in self.records:
            self.records[key]["confirmation_count"] += 1
            return "confirmation_added", self.get_by_key(key)
        self.records[key] = deepcopy(record)
        return "created", self.get_by_key(key)

    def set_status(self, key, status, active):
        self.records[key].update(status=status, active=active)
        return True

    def list_active(self):
        return [deepcopy(record) for record in self.records.values() if record["active"]]

    def find_active_with_same_scope(self, record):
        return [other for other in self.list_active() if other["_id"] != record["_id"]
                and all(other[key] == record[key] for key in ("type", "collections", "fields"))]

    def mark_conflict(self, key, other):
        self.records[key]["conflict_with"] = other

    def save_embedding_if_missing(self, key, vector, model):
        self.records[key].update(embedding=vector, embedding_model=model)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    chat = Mock(side_effect=AssertionError("Standalone feedback must not call a chat model"))
    monkeypatch.setattr(llm_router, "invoke", chat)
    # If the standalone route accidentally constructs the question-bound service,
    # fail before it can touch MongoDB or query history.
    monkeypatch.setattr("services.feedback_service.QueryHistoryRepository",
                        Mock(side_effect=AssertionError("Question-bound feedback is forbidden")))
    schema = SimpleNamespace(get_catalog=Mock(return_value=deepcopy(CATALOG)))
    embeddings = SimpleNamespace(embed_documents=Mock(return_value=[[1.0, 0.0]]))
    preferences, knowledge = Preferences(), Knowledge()
    ingestion = KnowledgeIngestionService(knowledge, schema, embeddings)
    service = StandaloneFeedbackService(preferences, ingestion, schema)
    app = create_app()
    app.dependency_overrides[get_standalone_feedback_service] = lambda: service
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, preferences=preferences, knowledge=knowledge,
                              schema=schema, embeddings=embeddings, chat=chat)
    chat.assert_not_called()


def test_preference_is_retrieved_and_applied_without_tokens(setup):
    response = setup.client.post("/feedback/standalone", json=PREFERENCE)
    assert response.status_code == 200
    result = response.json()
    assert result["feedback_id"].startswith("fb_")
    assert result["feedback_saved"] is True and result["learning_active"] is True
    record = setup.preferences.records[result["learning_id"]]
    assert record["user_id"] == DEFAULT_USER_ID
    assert record["source"] == "standalone_feedback"
    assert record["source_feedback_id"] == result["feedback_id"]
    service = UserPreferenceService(setup.preferences)
    applicable = service.retrieve_for_query("show failed orders", CATALOG, selected_collections={"orders"})
    assert len(applicable) == 1
    query = {"operation": "find", "collection": "orders", "limit": 100}
    assert apply_preferences_to_query(query, applicable, question="show failed orders")
    assert query["limit"] == 5 and query["sort"] == [["createdAt", -1]]
    assert service.retrieve_for_query("show successful orders", CATALOG, selected_collections={"orders"}) == []
    apply_preferences_to_query(query, applicable, question="show 12 failed orders")
    assert query["limit"] == 12
    setup.embeddings.embed_documents.assert_not_called()


def test_repeated_preference_updates_one_identity(setup):
    first = setup.client.post("/feedback/standalone", json=PREFERENCE).json()
    second = setup.client.post("/feedback/standalone", json={**PREFERENCE, "limit": 10, "text": "Return 10 latest failed orders."}).json()
    assert first["learning_id"] == second["learning_id"]
    assert second["action"] == "updated"
    assert len(setup.preferences.records) == 1
    assert setup.preferences.records[first["learning_id"]]["limit"] == 10


def test_structured_sort_field_and_direction_win_over_text_heuristics(setup):
    body = {"scope": "user", "type": "sorting_preference", "collection": "orders", "field": "total",
            "direction": "desc", "text": "Sort total ascending; createdAt was the old ordering field."}
    result = setup.client.post("/feedback/standalone", json=body)
    assert result.status_code == 200
    preferences = UserPreferenceService(setup.preferences).retrieve_for_query("show orders", CATALOG)
    assert preferences[0]["field"] == "total" and preferences[0]["direction"] == -1


def test_rule_is_active_and_retrievable_with_one_embedding(setup, monkeypatch):
    monkeypatch.setattr("services.knowledge_ingestion_service.KNOWLEDGE_AUTO_PROMOTION_ENABLED", False)
    response = setup.client.post("/feedback/standalone", json=RULE)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "active" and result["learning_active"] is True
    record = setup.knowledge.records[result["learning_id"]]
    assert record["source"] == "standalone_feedback" and record["created_from_feedback"] is True
    assert record["source_feedback_id"] == result["feedback_id"]
    retrieved = KnowledgeRetriever(setup.knowledge).retrieve("revenue from orders", CATALOG, [1.0, 0.0])
    assert retrieved["knowledge"][0]["knowledge_id"] == result["learning_id"]
    assert retrieved["knowledge"][0]["created_from_feedback"] is True
    setup.embeddings.embed_documents.assert_called_once()
    duplicate = setup.client.post("/feedback/standalone", json=RULE).json()
    assert duplicate["learning_id"] == result["learning_id"]
    assert len(setup.knowledge.records) == 1
    setup.embeddings.embed_documents.assert_called_once()


def test_conflicting_rule_is_saved_but_not_activated(setup):
    first = setup.client.post("/feedback/standalone", json=RULE).json()
    second = setup.client.post("/feedback/standalone", json={**RULE, "text": "Include refunded orders in revenue."}).json()
    assert second["feedback_saved"] is True
    assert second["learning_active"] is False and second["learning_error"]
    assert setup.knowledge.records[second["learning_id"]]["conflict_with"] == first["learning_id"]
    assert len(setup.knowledge.list_active()) == 1


def test_batch_saves_rules_and_preferences_in_request_order(setup):
    response = setup.client.post("/feedback/standalone/batch", json={"items": [RULE, PREFERENCE]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert (result["total"], result["succeeded"], result["failed"]) == (2, 2, 0)
    assert [item["scope"] for item in result["results"]] == ["database", "user"]
    assert all(item["learning_active"] for item in result["results"])
    rule, preference = result["results"]
    assert rule["feedback_id"] != preference["feedback_id"]
    assert setup.knowledge.records[rule["learning_id"]]["text"] == RULE["text"]
    assert setup.preferences.records[preference["learning_id"]]["limit"] == PREFERENCE["limit"]
    setup.embeddings.embed_documents.assert_called_once()


def test_batch_preserves_conflict_review_for_individual_rules(setup):
    conflicting = {**RULE, "text": "Include refunded orders in revenue."}
    response = setup.client.post("/feedback/standalone/batch", json={"items": [RULE, conflicting]})
    assert response.status_code == 200, response.text
    first, second = response.json()["results"]
    assert first["learning_active"] is True
    assert second["status"] == "needs_review" and second["learning_active"] is False
    assert second["learning_error"]
    assert setup.knowledge.records[second["learning_id"]]["conflict_with"] == first["learning_id"]


def test_batch_rejects_unknown_collection_without_writing(setup):
    response = setup.client.post("/feedback/standalone/batch", json={
        "items": [{**RULE, "collections": ["missing"]}]})
    assert response.status_code == 422
    assert not setup.knowledge.records and not setup.preferences.records
    setup.embeddings.embed_documents.assert_not_called()


def test_batch_embedding_failure_returns_sanitized_503(setup):
    setup.embeddings.embed_documents.side_effect = RuntimeError("private dependency details")
    response = setup.client.post("/feedback/standalone/batch", json={"items": [RULE]})
    assert response.status_code == 503
    assert "private dependency details" not in response.text
    assert not setup.knowledge.records


@pytest.mark.parametrize("payload", [
    {**PREFERENCE, "text": "   "}, {**PREFERENCE, "collection": "  "},
    {**PREFERENCE, "request_id": "req_unneeded"}, {**PREFERENCE, "question": "unneeded"},
    {**PREFERENCE, "scope": "one_off"}, {**PREFERENCE, "limit": True},
    {**PREFERENCE, "limit": 50001}, {**PREFERENCE, "limit": None},
    {**PREFERENCE, "latest": "true"}, {**PREFERENCE, "field": None},
    {**PREFERENCE, "condition": " "}, {**PREFERENCE, "apply_mode": "default"},
    {"scope": "user", "type": "sorting_preference", "text": "Sort orders", "collection": "orders"},
    {**RULE, "collections": []}, {**RULE, "collections": [" "]},
    {**RULE, "fields": [" "]}, {**RULE, "type": "sorting_preference"},
    {**RULE, "text": "x" * 1201}, {**RULE, "scope": "user"},
    {**PREFERENCE, "collection": "missing"}, {**PREFERENCE, "field": "missing"},
    {**RULE, "collections": ["orders", "missing"]},
    {**RULE, "fields": ["status", "name"]},
])
def test_invalid_inputs_never_write_or_embed(setup, payload):
    assert setup.client.post("/feedback/standalone", json=payload).status_code == 422
    assert not setup.preferences.records and not setup.knowledge.records
    setup.embeddings.embed_documents.assert_not_called()


def test_api_key_is_required(setup, monkeypatch):
    monkeypatch.setenv("API_KEY", "standalone-test-key")
    assert setup.client.post("/feedback/standalone", json=PREFERENCE).status_code == 401
    setup.schema.get_catalog.assert_not_called()
    response = setup.client.post("/feedback/standalone", json=PREFERENCE, headers={"X-API-Key": "standalone-test-key"})
    assert response.status_code == 200


@pytest.mark.parametrize("failure", ["schema", "preferences", "knowledge", "embedding"])
def test_unavailable_dependencies_return_sanitized_503(setup, failure):
    error = RuntimeError("private dependency details")
    payload = RULE
    if failure == "schema":
        setup.schema.get_catalog.side_effect = error
    elif failure == "preferences":
        setup.preferences.upsert = Mock(side_effect=error)
        payload = PREFERENCE
    elif failure == "knowledge":
        setup.knowledge.add_or_confirm = Mock(side_effect=error)
    else:
        setup.embeddings.embed_documents.side_effect = error
    response = setup.client.post("/feedback/standalone", json=payload)
    assert response.status_code == 503
    assert "private dependency details" not in response.text
    assert not setup.preferences.records and not setup.knowledge.records


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [float("nan")], [float("inf")]])
def test_invalid_embeddings_do_not_create_unretrievable_rules(setup, vector):
    setup.embeddings.embed_documents.return_value = [vector]
    assert setup.client.post("/feedback/standalone", json=RULE).status_code == 503
    assert not setup.knowledge.records


def test_new_route_has_discriminated_schema(setup):
    schema = setup.client.app.openapi()
    body = schema["paths"]["/feedback/standalone"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["discriminator"]["propertyName"] == "scope"
