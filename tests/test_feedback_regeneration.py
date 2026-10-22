"""Backend feedback application independent of classification and active knowledge."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from pymongo import ReturnDocument
import pytest

from agent.config import DEFAULT_USER_ID, FEEDBACK_COLLECTION, QUERY_HISTORY_COLLECTION
from agents.feedback_analysis_agent import FeedbackAnalysis, FeedbackAnalysisAgent
from api.app import create_app
from api.routers.feedback import get_feedback_service
from api.routers.query import get_query_history_service
from database.feedback_repository import FeedbackRepository
from database.query_history_repository import QueryHistoryRepository
from services.feedback_service import FeedbackService
from services.query_history_service import QueryHistoryService

QUESTION = "Show recharge orders"
COMMENT = "orders should be fetched from orders_proc_hist collection with actionCode = Recharge"


def get_path(record, key):
    value = record
    for name in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def matches(record, query):
    for key, value in query.items():
        actual = get_path(record, key)
        if isinstance(value, dict):
            if "$gt" in value and (actual is None or actual <= value["$gt"]):
                return False
        elif actual != value:
            return False
    return True


def set_path(record, key, value):
    parts = key.split(".")
    for name in parts[:-1]:
        record = record.setdefault(name, {})
    record[parts[-1]] = deepcopy(value)


class Cursor(list):
    def sort(self, field, direction):
        return Cursor(sorted(self, key=lambda item: get_path(item, field), reverse=direction < 0))

    def limit(self, count):
        return Cursor(self[:count])


class Collection:
    """Implements the Mongo operations under test, including atomic claims."""
    def __init__(self):
        self.records = {}
        self.lock = RLock()

    def create_index(self, *args, **kwargs):
        pass

    def insert_one(self, record):
        with self.lock:
            self.records[record["_id"]] = deepcopy(record)

    def find(self, query, projection=None):
        with self.lock:
            return Cursor(deepcopy(item) for item in self.records.values() if matches(item, query))

    def find_one(self, query, projection=None):
        items = self.find(query)
        if not items:
            return None
        item = items[0]
        if projection and projection.get("_id") == 0:
            item.pop("_id", None)
        return item

    def update_one(self, query, update, upsert=False):
        with self.lock:
            record = next((item for item in self.records.values() if matches(item, query)), None)
            existed = record is not None
            if not existed:
                if not upsert:
                    return SimpleNamespace(matched_count=0)
                record = {"_id": query["_id"], **deepcopy(update.get("$setOnInsert", {}))}
                self.records[record["_id"]] = record
            for key, value in update.get("$set", {}).items():
                set_path(record, key, value)
            return SimpleNamespace(matched_count=int(existed), modified_count=1)

    def find_one_and_update(self, query, update, return_document=None):
        with self.lock:
            before = self.find_one(query)
            if before is None:
                return None
            self.update_one(query, update)
            return self.find_one({"_id": before["_id"]}) if return_document == ReturnDocument.AFTER else before


def one_off():
    return FeedbackAnalysis(feedback_type="wrong_collection", cause="Wrong source and code case",
        corrected_meaning="For this request, use orders_proc_hist with actionCode = Recharge",
        reusable_preference=False, preference=None, knowledge_candidate=False, knowledge_type=None,
        collections=["orders_proc_hist"], fields=["actionCode"], scope="one_off", confidence=0.94)


@pytest.fixture
def setup():
    database = {QUERY_HISTORY_COLLECTION: Collection(), FEEDBACK_COLLECTION: Collection()}
    history = QueryHistoryRepository(database)
    feedback = FeedbackRepository(database)
    history.save({"request_id": "req_original", "question": QUESTION, "user_id": DEFAULT_USER_ID})
    analyzer = Mock()
    analyzer.analyze.return_value = one_off()
    router = Mock()
    router.route.return_value = {"route": "one_off"}
    service = FeedbackService(history, feedback, analyzer, learning_router=router, template_repository=Mock())
    service.preference_service = Mock()
    pipeline = Mock(return_value={"execution_success": True, "execution_result": [], "applied_knowledge": []})
    query = QueryHistoryService(history, pipeline, feedback_repository=feedback)
    return SimpleNamespace(database=database, history=history, feedback=feedback,
                           analyzer=analyzer, router=router, service=service, pipeline=pipeline, query=query)


@pytest.mark.parametrize("classification", ["one_off", "candidate", "analysis_failure", "learning_failure"])
def test_saved_feedback_applies_even_when_not_active_knowledge(setup, classification):
    if classification == "candidate":
        setup.analyzer.analyze.return_value = one_off().model_copy(update={
            "scope": "database", "knowledge_candidate": True, "knowledge_type": "business_rule"})
        setup.router.route.return_value = {"route": "database_knowledge", "knowledge": {
            "knowledge_id": "candidate", "status": "candidate", "active": False}}
    elif classification == "analysis_failure":
        setup.analyzer.analyze.side_effect = RuntimeError("provider unavailable")
    elif classification == "learning_failure":
        setup.router.route.side_effect = RuntimeError("learning unavailable")
    saved = setup.service.submit("req_original", False, COMMENT)
    assert saved["feedback_saved"] and saved["regeneration_ready"]
    assert saved["learning_active"] is False
    result = setup.query.run(QUESTION)
    assert result["execution_success"]
    effective = setup.pipeline.call_args.args[0]
    assert QUESTION in effective and COMMENT in effective
    assert "overrides conflicting earlier details" in effective
    assert "actionCode = Recharge" in effective
    assert result["original_question"] == QUESTION
    assert result["effective_question"] == effective
    context = result["request_context"]
    assert context["original_question"] == QUESTION
    assert context["corrections"] == [COMMENT]
    assert COMMENT in context["resolved_question"]
    assert "Apply the latest correction" not in context["resolved_question"]
    assert result["applied_feedback"]["source_request_id"] == "req_original"
    assert result["applied_knowledge"] == []
    record = setup.history.get_by_request_id(result["request_id"])
    assert record["question"] == QUESTION
    assert record["applied_feedback"] == result["applied_feedback"]
    assert record["request_context"] == context
    assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "applied"


def test_feedback_is_consumed_once_and_unrelated_question_does_not_consume_it(setup):
    setup.service.submit("req_original", False, COMMENT)
    setup.query.run("Show customers")
    assert setup.pipeline.call_args.args[0] == "Show customers"
    assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "pending"
    assert setup.query.run(QUESTION).get("applied_feedback")
    assert not setup.query.run(QUESTION).get("applied_feedback")
    assert setup.pipeline.call_args.args[0] == QUESTION


@pytest.mark.parametrize("failure", ["exception", "unsuccessful_result"])
def test_failed_generation_can_retry_same_correction(setup, failure):
    setup.service.submit("req_original", False, COMMENT)
    setup.pipeline.side_effect = [RuntimeError("failed") if failure == "exception" else {
        "execution_success": False, "execution_error": "invalid field"}, {"execution_success": True}]
    assert not setup.query.run(QUESTION)["execution_success"]
    assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "pending"
    assert setup.query.run(QUESTION)["execution_success"]
    assert setup.pipeline.call_args_list[0].args[0] == setup.pipeline.call_args_list[1].args[0]


def test_positive_feedback_cancels_pending_correction(setup):
    setup.service.submit("req_original", False, COMMENT)
    setup.service.submit("req_original", True)
    assert not setup.query.run(QUESTION).get("applied_feedback")
    assert setup.pipeline.call_args.args[0] == QUESTION


def test_latest_revision_replaces_old_comment(setup):
    setup.service.submit("req_original", False, COMMENT)
    old_revision = setup.feedback.get_by_request_id("req_original")["revision"]
    latest = "Use orders_proc_hist with actionCode = OtherCode"
    setup.service.submit("req_original", False, latest)
    result = setup.query.run(QUESTION)
    assert latest in setup.pipeline.call_args.args[0] and COMMENT not in setup.pipeline.call_args.args[0]
    assert result["applied_feedback"]["revision"] != old_revision


def test_repeated_corrections_preserve_prior_question_scope(setup):
    setup.service.submit("req_original", False, COMMENT)
    first = setup.query.run(QUESTION)
    second_comment = "Limit the corrected results to December 2025"
    setup.service.submit(first["request_id"], False, second_comment)
    second = setup.query.run(QUESTION)
    assert second["original_question"] == QUESTION
    assert COMMENT in second["effective_question"] and second_comment in second["effective_question"]
    assert second["request_context"]["corrections"] == [COMMENT, second_comment]
    assert second["request_context"]["resolved_question"].count(COMMENT) == 1
    assert "Apply the latest correction" not in second["request_context"]["resolved_question"]


def test_older_pending_feedback_does_not_resurface_after_newer_is_used(setup):
    setup.service.submit("req_original", False, "older correction")
    setup.history.save({"request_id": "req_newer", "question": QUESTION, "user_id": DEFAULT_USER_ID})
    setup.service.submit("req_newer", False, COMMENT)
    # BSON would truncate both dates to milliseconds. Nanosecond ordering
    # still identifies the newest distinct request deterministically.
    for record in setup.database[FEEDBACK_COLLECTION].records.values():
        record["regeneration"]["submitted_at"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert setup.query.run(QUESTION)["applied_feedback"]["source_request_id"] == "req_newer"
    assert not setup.query.run(QUESTION).get("applied_feedback")


def test_expired_feedback_and_other_users_are_not_applied(setup):
    setup.service.submit("req_original", False, COMMENT)
    assert setup.feedback.claim_correction(QUESTION, "other-user", "claim") is None
    raw = setup.database[FEEDBACK_COLLECTION].records["req_original"]
    raw["regeneration"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert not setup.query.run(QUESTION).get("applied_feedback")


def test_naive_bson_dates_are_supported(setup):
    setup.service.submit("req_original", False, COMMENT)
    raw = setup.database[FEEDBACK_COLLECTION].records["req_original"]
    # The comparison inside the fake needs UTC aware dates; exercise the
    # repository's naive-date eligibility using a captured Mongo update call.
    naive = deepcopy(raw)
    naive["regeneration"]["expires_at"] = naive["regeneration"]["expires_at"].replace(tzinfo=None)
    collection = Mock()
    collection.find.return_value = Cursor([naive])
    collection.find_one_and_update.return_value = raw
    repository = FeedbackRepository({FEEDBACK_COLLECTION: collection})
    assert repository.claim_correction(QUESTION, DEFAULT_USER_ID, "claim") == raw


def test_only_one_concurrent_generation_can_claim_revision(setup):
    setup.service.submit("req_original", False, COMMENT)
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(lambda index: setup.feedback.claim_correction(
            QUESTION, DEFAULT_USER_ID, f"claim-{index}"), range(4)))
    assert sum(item is not None for item in outcomes) == 1


def test_finishing_old_claim_does_not_overwrite_new_revision(setup):
    setup.service.submit("req_original", False, COMMENT)
    claimed = setup.feedback.claim_correction(QUESTION, DEFAULT_USER_ID, "claim")
    setup.service.submit("req_original", False, "New correction")
    setup.feedback.finish_correction("req_original", claimed["revision"], "claim", True)
    assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "pending"


def test_explicit_feedback_reference_cannot_apply_to_another_question(setup):
    setup.service.submit("req_original", False, COMMENT)
    result = setup.query.run("Show customers", feedback_request_id="req_original")
    assert result["execution_error_code"] == "feedback_correction_unavailable"
    setup.pipeline.assert_not_called()


def test_regeneration_skips_template_reuse_only_for_the_corrected_request(setup):
    setup.service.submit("req_original", False, COMMENT)
    setup.query.template_cache = Mock()
    setup.query.template_cache.lookup.side_effect = AssertionError("Rejected query must not be reused")
    result = setup.query.run(QUESTION)
    assert result["execution_success"]
    setup.query.template_cache.lookup.assert_not_called()
    assert result["semantic_template_cache"]["user_decision"] == "feedback_regenerate"
    setup.query.template_cache.lookup.return_value = (None, {"checked": True, "hit": False})
    setup.query.template_cache.lookup.side_effect = None
    setup.query.run(QUESTION)
    setup.query.template_cache.lookup.assert_called_once_with(QUESTION)


@pytest.mark.parametrize("pause_failure", [False, True])
def test_cost_guard_continuation_keeps_correction_and_pause_failure_can_retry(setup, monkeypatch, pause_failure):
    from services.query_optimization_service import QueryOptimizationService
    setup.service.submit("req_original", False, COMMENT)
    setup.pipeline.return_value = {"status": "optimization_required", "execution_success": False}
    pause = Mock(side_effect=RuntimeError("storage unavailable") if pause_failure else None)
    monkeypatch.setattr(QueryOptimizationService, "pause", pause)
    if pause_failure:
        with pytest.raises(RuntimeError):
            setup.query.run(QUESTION)
        assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "pending"
    else:
        result = setup.query.run(QUESTION)
        assert COMMENT in pause.call_args.args[1]
        assert result["status"] == "optimization_required"
        assert setup.feedback.get_by_request_id("req_original")["regeneration"]["state"] == "applied"


@pytest.mark.parametrize("route", ["user_preference", "database_knowledge"])
def test_response_reports_active_learning_separately_from_regeneration(setup, route):
    if route == "user_preference":
        setup.router.route.return_value = {"route": route, "preference_action": "created"}
    else:
        setup.router.route.return_value = {"route": route, "knowledge": {
            "knowledge_id": "active", "status": "active", "active": True}}
    result = setup.service.submit("req_original", False, COMMENT)
    assert result["learning_active"] and result["regeneration_ready"]


def test_latest_positive_feedback_prevents_older_pending_correction_from_resurfacing(setup):
    setup.service.submit("req_original", False, COMMENT)
    setup.history.save({"request_id": "req_newer", "question": QUESTION, "user_id": DEFAULT_USER_ID})
    setup.service.submit("req_newer", True)
    assert not setup.query.run(QUESTION).get("applied_feedback")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("explicit_reference", [False, True])
def test_current_api_flow_applies_feedback_without_frontend_changes(setup, monkeypatch, streaming, explicit_reference):
    monkeypatch.delenv("API_KEY", raising=False)
    app = create_app()
    app.dependency_overrides[get_feedback_service] = lambda: setup.service
    app.dependency_overrides[get_query_history_service] = lambda: setup.query
    with TestClient(app) as client:
        saved = client.post("/feedback", json={"request_id": "req_original", "liked": False, "comment": COMMENT})
        assert saved.status_code == 200 and saved.json()["regeneration_ready"]
        payload = {"question": QUESTION}
        if explicit_reference:
            payload["feedback_request_id"] = "req_original"
        response = client.post("/query/stream" if streaming else "/query", json=payload)
        assert response.status_code == 200
        if streaming:
            frame = next(frame for frame in response.text.split("\n\n") if frame.startswith("event: result"))
            data = json.loads(frame.split("data: ", 1)[1])
        else:
            data = response.json()
        assert data["applied_feedback"]["source_request_id"] == "req_original"
        assert COMMENT in setup.pipeline.call_args.args[0]


@pytest.mark.parametrize("scope", ["database", "user"])
def test_explicit_one_off_words_override_incorrect_reusable_classification(scope):
    analysis = one_off().model_copy(update={"scope": scope,
        "knowledge_candidate": scope == "database", "knowledge_type": "business_rule" if scope == "database" else None,
        "reusable_preference": scope == "user", "preference": "Use orders_proc_hist" if scope == "user" else None})
    fixed = FeedbackAnalysisAgent._normalize_explicit_one_off(analysis, "For this query only, " + COMMENT)
    assert fixed.scope == "one_off"
    assert not fixed.knowledge_candidate and not fixed.reusable_preference
    assert fixed.scope_evidence == "For this query only"


def test_model_invented_one_off_words_do_not_establish_user_scope_restriction():
    analysis = one_off()
    assert FeedbackAnalysisAgent._normalize_explicit_one_off(analysis, COMMENT) is analysis


def test_classifier_prompt_describes_reusable_source_rules_and_original_evidence(monkeypatch):
    import agents.feedback_analysis_agent as module
    captured = []
    monkeypatch.setattr(module.llm_router, "invoke", lambda role, messages, schema: (
        captured.append("\n".join(message.content for message in messages)) or SimpleNamespace(content=one_off().model_dump_json())))
    FeedbackAnalysisAgent().analyze({"question": QUESTION, "effective_question": "Earlier corrected request"}, COMMENT)
    prompt = captured[0]
    assert 'Absence of words like "save", "remember", "always" does NOT imply one_off' in prompt
    assert "Do not invent" in prompt and "scope_evidence" in prompt
    assert '"Take recharge orders from orders_proc_hist where actionCode = Recharge." -> database' in prompt
    assert '"For this query only, take recharge orders from orders_proc_hist." -> one_off' in prompt
    assert "A source-only mapping has knowledge_type = business_rule" in prompt
    assert "field_definition explains the meaning or representation of a named FIELD" in prompt
    assert "Earlier corrected request" in prompt


def test_explicit_one_off_is_enforced_through_actual_analysis_method(monkeypatch):
    import agents.feedback_analysis_agent as module
    misclassified = one_off().model_copy(update={"scope": "database", "knowledge_candidate": True,
                                               "knowledge_type": "business_rule"})
    monkeypatch.setattr(module.llm_router, "invoke", lambda *args: SimpleNamespace(content=misclassified.model_dump_json()))
    result = FeedbackAnalysisAgent().analyze({"question": QUESTION}, "For this query only, " + COMMENT)
    assert result.scope == "one_off" and not result.knowledge_candidate


def test_raw_correction_reaches_both_agent_prompts_and_changes_validated_query(setup, monkeypatch):
    import orchestrator
    from agent.llm import llm_router
    catalog = {"collections": {
        "orders": {"fields": {"actionCode": ["string"]}},
        "orders_proc_hist": {"fields": {"actionCode": ["string"]}},
    }}
    prompts = {}

    def invoke(role, messages, schema):
        prompts[role] = messages[0].content
        if role == "query_context":
            output = {"complexity": "simple", "operation": "find", "collections": ["orders_proc_hist"],
                      "fields": {"orders_proc_hist": ["actionCode"]}}
        else:
            output = {"semantic_intent": "Fetch recharge orders", "template": {
                "operation": "find", "collection": "orders_proc_hist",
                "filter": {"actionCode": {"$param": "action_code"}}, "limit": None},
                "parameters": {"action_code": "Recharge"}}
        return SimpleNamespace(content=json.dumps(output))

    monkeypatch.setattr(llm_router, "invoke", invoke)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: catalog)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    schema_queries = []
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", lambda question, preferences: (
        schema_queries.append(question) or {"catalog": catalog, "telemetry": {}}))
    app = orchestrator.OrchestratorAgent()
    app.preference_service = Mock(user_id=DEFAULT_USER_ID)
    app.preference_service.retrieve_for_query.return_value = []
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [], "telemetry": {}, "_active_rules": []}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [], "user_preferences": []}
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={
        "status": "safe", "query_signature": "sig", "estimated_rows": 0}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())
    setup.query.pipeline = app.run
    setup.service.submit("req_original", False, COMMENT)
    result = setup.query.run(QUESTION)
    assert result["execution_success"], result
    assert COMMENT in schema_queries[0]
    assert COMMENT in prompts["query_context"] and COMMENT in prompts["query_generator"]
    assert result["validated_query"]["collection"] == "orders_proc_hist"
    assert result["validated_query"]["filter"] == {"actionCode": "Recharge"}
    assert result["applied_knowledge"] == []
