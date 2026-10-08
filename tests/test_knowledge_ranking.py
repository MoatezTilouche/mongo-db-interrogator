"""Relevance gates, bounded feedback priority, and the actual prompt handoff."""
from copy import deepcopy
import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.knowledge_retriever as module
from services.context_builder import ContextBuilder


CATALOG = {"collections": {
    "orders": {"fields": {"_id": ["ObjectId"], "status": ["string"], "total": ["number"]}},
    "customers": {"fields": {"_id": ["ObjectId"], "name": ["string"]}},
}}
CANDIDATES = {"collections": {"orders": deepcopy(CATALOG["collections"]["orders"])}}


def rule(key, similarity, source="manual", **changes):
    # Unit vectors give a known, real cosine without paid network calls.
    record = {"_id": key, "source": source, "type": "business_rule",
              "text": f"Orders rule {key}: Failed orders must exclude retry status X",
              "collections": ["orders"], "fields": ["status"],
              "status": "active", "active": True, "confidence": 0.98,
              "embedding": [similarity, math.sqrt(1 - similarity ** 2)],
              "embedding_model": module.KNOWLEDGE_EMBEDDING_MODEL}
    return {**record, **changes}


@pytest.fixture(autouse=True)
def ranking_config(monkeypatch):
    # Environment-independent contract; production values remain configurable.
    monkeypatch.setattr(module, "KNOWLEDGE_MIN_SEMANTIC_SIMILARITY", 0.65)
    monkeypatch.setattr(module, "KNOWLEDGE_FEEDBACK_PRIORITY_BONUS", 0.15)
    monkeypatch.setattr(module, "KNOWLEDGE_SCOPE_MATCH_BONUS", 0.02)
    monkeypatch.setattr(module, "KNOWLEDGE_COLLECTION_MATCH_BONUS", 0.05)
    monkeypatch.setattr(module, "KNOWLEDGE_FIELD_MATCH_BONUS", 0.05)


def retrieve(records, candidates=CANDIDATES, full_schema=CATALOG, embedding=None, top_k=5):
    repository = SimpleNamespace(list_active=lambda: records)
    return module.KnowledgeRetriever(repository, top_k).retrieve(
        "show failed orders", candidates, [1.0, 0.0] if embedding is None else embedding, full_schema)


def ids(result):
    return [item["knowledge_id"] for item in result["knowledge"]]


def test_relevant_feedback_ranks_first_and_revenue_rule_is_excluded():
    feedback = rule("feedback", 0.86, "negative_feedback")
    manual = rule("manual", 0.90)
    revenue = rule("revenue", 0.40, "negative_feedback", text="Revenue uses completed orders only")
    assert ids(retrieve([manual, revenue, feedback])) == ["feedback", "manual"]


def test_calibrated_default_admits_measured_same_intent_but_excludes_revenue(monkeypatch):
    monkeypatch.setattr(module, "KNOWLEDGE_MIN_SEMANTIC_SIMILARITY", 0.50)
    assert ids(retrieve([rule("feedback", 0.535943, "negative_feedback"),
                         rule("revenue", 0.477192, "negative_feedback")])) == ["feedback"]


def test_irrelevant_feedback_cannot_jump_above_relevant_manual_rule():
    assert ids(retrieve([rule("unrelated", 0.10, "negative_feedback"), rule("manual", 0.82)])) == ["manual"]


def test_feedback_bonus_never_bypasses_semantic_threshold(monkeypatch):
    monkeypatch.setattr(module, "KNOWLEDGE_FEEDBACK_PRIORITY_BONUS", 100)
    assert ids(retrieve([rule("below", 0.64, "negative_feedback")])) == []
    assert ids(retrieve([rule("boundary", 0.65, "negative_feedback")])) == ["boundary"]


def test_clearly_more_relevant_manual_rule_can_outrank_feedback():
    assert ids(retrieve([rule("feedback", 0.70, "negative_feedback"), rule("manual", 0.99)])) == ["manual", "feedback"]


def test_feedback_is_ranked_before_default_top_five_truncation():
    records = [rule(f"manual-{index}", 0.90 - index / 100) for index in range(8)]
    for index, record in enumerate(records):
        record["text"] = f"Distinct orders rule {index}"
    records.append(rule("feedback", 0.86, "negative_feedback"))
    result = module.KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).retrieve(
        "show failed orders", CANDIDATES, [1.0, 0.0], CATALOG)
    assert ids(result) == ["feedback", "manual-0", "manual-1", "manual-2", "manual-3"]
    assert result["telemetry"]["knowledge_retrieval_top_k"] == 5
    assert result["telemetry"]["knowledge_candidates_returned"] == 5


def test_raw_cosine_is_preserved_and_ranking_score_exposes_bonuses():
    record = rule("feedback", 0.84, "negative_feedback")
    original = deepcopy(record)
    item = retrieve([record])["knowledge"][0]
    assert item["semantic_similarity"] == module._cosine([1.0, 0.0], record["embedding"])
    assert item["semantic_similarity"] == pytest.approx(0.84)
    assert item["ranking_score"] == pytest.approx(0.84 + 0.15 + 0.02 + 0.05 + 0.05)
    assert record == original


@pytest.mark.parametrize("changes", [
    {"status": "candidate", "active": False},
    {"status": "confirmed", "active": False},
    {"status": "rejected", "active": False},
    {"status": "superseded", "active": False},
    {"status": "active", "active": False},
    {"status": "candidate", "active": True},
])
def test_nonactive_feedback_is_excluded_even_if_repository_returns_it(changes):
    assert ids(retrieve([rule("bad", 1.0, "negative_feedback", **changes)])) == []


@pytest.mark.parametrize("changes", [
    {"collections": ["missing_orders"]},
    {"fields": ["removed_status"]},
    {"collections": [], "fields": ["removed_status"]},
    {"collections": ["customers"], "fields": ["status"]},
    {"collections": "orders"}, {"fields": "status"},
    {"scope": "user"}, {"text": ""},
])
def test_schema_invalid_or_malformed_knowledge_is_excluded(changes):
    assert ids(retrieve([rule("bad", 1.0, "negative_feedback", **changes)])) == []


def test_valid_but_incompatible_collection_scope_is_excluded():
    record = rule("customer-feedback", 0.99, "negative_feedback", collections=["customers"], fields=["name"])
    assert ids(retrieve([record])) == []


def test_schema_validation_is_enforced_without_full_catalog():
    assert ids(retrieve([rule("bad", 0.90, fields=["missing"])], full_schema=None)) == []


@pytest.mark.parametrize("embedding", [[], [0.0, 0.0], [1.0], [float("nan"), 0.0],
                                       [float("inf"), 0.0], ["bad", 0.0], 123])
def test_missing_invalid_or_incompatible_embeddings_cannot_prove_relevance(embedding):
    assert ids(retrieve([rule("bad", 0.99, "negative_feedback", embedding=embedding)])) == []


def test_query_embedding_failure_does_not_retrieve_by_scope_alone():
    assert ids(retrieve([rule("feedback", 0.99, "negative_feedback")], embedding=[])) == []


def test_model_mismatch_is_excluded():
    assert ids(retrieve([rule("feedback", 0.99, "negative_feedback", embedding_model="different-model")])) == []


def test_missing_source_metadata_has_no_feedback_bonus():
    record = rule("legacy", 0.84)
    record.pop("source")
    record.pop("embedding_model")
    item = retrieve([record])["knowledge"][0]
    assert item["source"] == "manual"
    assert item["created_from_feedback"] is False
    assert item["ranking_score"] == pytest.approx(0.84 + 0.02 + 0.05 + 0.05)


def test_legacy_user_feedback_origin_is_recognized_without_migration():
    assert ids(retrieve([rule("manual", 0.90), rule("legacy-feedback", 0.84, "user_feedback")])) == ["legacy-feedback", "manual"]


def test_bonuses_are_configurable_and_similarity_is_unchanged(monkeypatch):
    records = [rule("feedback", 0.84, "negative_feedback"), rule("manual", 0.90)]
    monkeypatch.setattr(module, "KNOWLEDGE_FEEDBACK_PRIORITY_BONUS", 0.0)
    assert ids(retrieve(records)) == ["manual", "feedback"]
    monkeypatch.setattr(module, "KNOWLEDGE_COLLECTION_MATCH_BONUS", 0.3)
    monkeypatch.setattr(module, "KNOWLEDGE_SCOPE_MATCH_BONUS", 0.1)
    monkeypatch.setattr(module, "KNOWLEDGE_FIELD_MATCH_BONUS", 0.2)
    assert retrieve(records)["knowledge"][0]["ranking_score"] == pytest.approx(0.90 + 0.3 + 0.1 + 0.2)


def test_duplicate_ids_are_removed_before_top_k():
    repeated = rule("feedback", 0.84, "negative_feedback")
    assert ids(retrieve([repeated] * 8 + [rule("manual", 0.82)])) == ["feedback", "manual"]


def test_multiple_feedback_rules_keep_semantic_order_with_deterministic_ties():
    records = [rule("c", 0.82, "negative_feedback"), rule("b", 0.86, "negative_feedback"),
               rule("a", 0.86, "negative_feedback")]
    for record in records:
        record["text"] = f"Distinct orders rule {record['_id']}"
    assert ids(retrieve(records)) == ["a", "b", "c"]


def test_same_rule_with_different_ids_is_not_duplicated():
    records = [rule("manual-copy", 0.90), rule("feedback", 0.86, "negative_feedback")]
    records[0]["text"] = records[1]["text"]
    assert ids(retrieve(records)) == ["feedback"]


def test_active_repository_does_not_cap_candidates_before_ranking():
    from agent.config import DATABASE_KNOWLEDGE_COLLECTION
    from database.knowledge_repository import KnowledgeRepository
    collection = Mock()
    cursor = Mock()
    records = [rule(str(index), 0.10) for index in range(1001)] + [rule("new-feedback", 0.86, "negative_feedback")]
    cursor.__iter__ = Mock(return_value=iter(records))
    collection.find.return_value = cursor
    repository = KnowledgeRepository({DATABASE_KNOWLEDGE_COLLECTION: collection})
    result = module.KnowledgeRetriever(repository).retrieve("show failed orders", CANDIDATES, [1.0, 0.0], CATALOG)
    assert ids(result) == ["new-feedback"]
    cursor.limit.assert_not_called()


def test_conflicting_active_rules_are_kept_separate_in_priority_order():
    records = [rule("manual", 0.88, text="Failed orders must include retry status X"),
               rule("feedback", 0.85, "negative_feedback")]
    items = retrieve(records)["knowledge"]
    assert [item["knowledge_id"] for item in items] == ["feedback", "manual"]
    assert items[0]["text"] != items[1]["text"]


def test_truncated_fields_and_corrected_collection_are_restored_to_their_owner():
    candidates = {"collections": {"customers": {"fields": {"_id": ["ObjectId"]}}}}
    record = rule("relationship", 0.90, "negative_feedback",
                  collections=["orders", "customers"], fields=["status", "name"])
    result = retrieve([record], candidates=candidates)
    assert ids(result) == ["relationship"]
    context = ContextBuilder().build("show failed orders", candidates, result["knowledge"], full_catalog=CATALOG)
    assert "status" in context["schema"]["collections"]["orders"]["fields"]
    assert "name" in context["schema"]["collections"]["customers"]["fields"]
    assert "status" not in context["schema"]["collections"]["customers"]["fields"]
    assert "name" not in context["schema"]["collections"]["orders"]["fields"]


def test_logs_explain_scores_exclusions_and_final_order(caplog):
    with caplog.at_level(logging.INFO, logger=module.__name__):
        retrieve([rule("feedback", 0.86, "negative_feedback"), rule("below", 0.40, "negative_feedback")])
    assert "knowledge_candidate knowledge_id=feedback source=negative_feedback semantic_similarity=0.860000" in caplog.text
    assert "feedback_bonus=0.150000" in caplog.text
    assert "ranking_score=1.130000" in caplog.text
    assert "reason=below_semantic_threshold" in caplog.text
    assert "ordered_knowledge_ids=['feedback'] top_k=5 returned_count=1" in caplog.text


def test_retrieval_order_reaches_both_actual_agent_prompts_and_orchestrator(monkeypatch):
    import orchestrator
    from agent.llm import llm_router
    prompts = {}

    def invoke(role, messages, schema):
        prompts[role] = messages[0].content
        if role == "query_context":
            output = {"complexity": "simple", "operation": "find", "collections": ["orders"],
                      "fields": {"orders": ["status"]}}
        else:
            output = {"semantic_intent": "Find failed orders", "template": {
                "operation": "find", "collection": "orders", "filter": {}, "limit": None}, "parameters": {}}
        return SimpleNamespace(content=json.dumps(output))

    records = [rule("a-manual", 0.90), rule("z-feedback", 0.86, "negative_feedback")]
    records[0]["text"] = "Failed orders must include retry status X"
    monkeypatch.setattr(llm_router, "invoke", invoke)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: CATALOG)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", lambda *args: {
        "catalog": CANDIDATES, "telemetry": {}, "_query_embedding": [1.0, 0.0]})
    app = orchestrator.OrchestratorAgent()
    app.knowledge_retriever = module.KnowledgeRetriever(SimpleNamespace(list_active=lambda: records))
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={
        "status": "safe", "query_signature": "sig", "estimated_rows": 0}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())
    result = app.run("show failed orders; explicitly include retry status X")
    assert result["execution_success"], result
    for role in ("query_context", "query_generator"):
        prompt = prompts[role]
        assert prompt.index('"knowledge_id":"z-feedback"') < prompt.index('"knowledge_id":"a-manual"')
        assert prompt.count('"knowledge_id":"z-feedback"') == 1
        assert "explicitly include retry status X" in prompt
        assert "explicit current request" in prompt
        assert "Do not merge contradictory rules" in prompt
    assert result["telemetry"]["knowledge_candidates_returned"] == 2


@pytest.mark.parametrize("liked", [False, True])
def test_negative_origin_persists_through_service_router_promotion_and_repository(monkeypatch, liked):
    from agent.config import DATABASE_KNOWLEDGE_COLLECTION
    from agents.feedback_analysis_agent import FeedbackAnalysis
    from database.knowledge_repository import KnowledgeRepository
    from services.feedback_learning_router import FeedbackLearningRouter
    from services.feedback_service import FeedbackService
    import services.knowledge_ingestion_service as ingestion

    stored = {}
    collection = Mock()
    collection.insert_one.side_effect = lambda record: stored.update({record["_id"]: deepcopy(record)})
    collection.find_one.side_effect = lambda query: deepcopy(stored.get(query["_id"]))

    def update(query, changes):
        stored[query["_id"]].update(changes["$set"])
        return SimpleNamespace(matched_count=1)

    def find(query):
        matching = [deepcopy(item) for item in stored.values()
                    if all(item.get(key) == value for key, value in query.items())]
        class Cursor(list):
            def limit(self, count):
                return self[:count]
        return Cursor(matching)

    collection.update_one.side_effect = update
    collection.find.side_effect = find
    repository = KnowledgeRepository({DATABASE_KNOWLEDGE_COLLECTION: collection})
    monkeypatch.setattr(ingestion, "KNOWLEDGE_LEARNING_MODE", "immediate")
    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", True)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE", 0.9)
    service = ingestion.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: CATALOG),
        embedding_provider=SimpleNamespace(embed_documents=lambda *args, **kwargs: [[1.0, 0.0]]))
    analyzer = Mock()
    analyzer.analyze.return_value = FeedbackAnalysis(
        feedback_type="wrong_business_rule", cause="Retry orders were incorrectly included",
        corrected_meaning="Failed orders must exclude retry status X", reusable_preference=False,
        preference=None, scope="database", knowledge_candidate=True, knowledge_type="business_rule",
        collections=["orders"], fields=["status"], confidence=0.98)
    history = Mock()
    history.get_by_request_id.return_value = {"question": "show failed orders", "user_id": "test"}
    feedback = Mock()
    feedback.save_analysis.return_value = True
    feedback_service = FeedbackService(history_repository=history, feedback_repository=feedback,
        analyzer=analyzer, learning_router=FeedbackLearningRouter(Mock(), service), template_repository=Mock())
    feedback_service.preference_service = Mock()
    result = feedback_service.submit("req_negative", liked, "Exclude retry status X")
    if liked:
        assert stored == {}
        analyzer.analyze.assert_not_called()
        return
    assert result["knowledge_status"] == "active"
    saved = stored[result["knowledge_id"]]
    assert saved["source"] == "negative_feedback"
    assert saved["created_from_feedback"] is True
    assert saved["source_request_id"] == "req_negative"
    assert saved["source_request_ids"] == ["req_negative"]
    items = module.KnowledgeRetriever(repository).retrieve("show failed orders", CANDIDATES, [1.0, 0.0], CATALOG)["knowledge"]
    assert items[0]["knowledge_id"] == result["knowledge_id"]
    collection.find.assert_called_with({"status": "active", "active": True})
