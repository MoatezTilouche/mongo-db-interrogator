"""Stage retrieval is local, deterministic, and uses only fake repositories."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.knowledge_retriever import KnowledgeRetriever
import services.knowledge_retriever as module
from services.knowledge_embeddings import definition_text, embedding_fields


NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
CATALOG = {"collections": {
    "orders": {"fields": {"id": ["string"], "executionDate": ["string"],
                          "status": ["string"], "code": ["int"], "total": ["number"]}},
    "Recharge": {"fields": {"id": ["string"], "rechargeDate": ["date"], "status": ["string"]}},
}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import database.mongo_client as mongo
    import services.knowledge_ingestion_service as ingestion
    from agent.llm import llm_router

    def forbidden(*args, **kwargs):
        raise AssertionError("Retrieval must not call a model or a real database")

    monkeypatch.setattr(mongo, "get_database", forbidden)
    monkeypatch.setattr("database.knowledge_repository.get_database", forbidden)
    monkeypatch.setattr(llm_router, "invoke", forbidden)
    monkeypatch.setattr(llm_router, "embed_documents", forbidden)
    monkeypatch.setattr(module, "KNOWLEDGE_CONTEXT_MIN_SIMILARITY", 0.75)
    monkeypatch.setattr(module, "KNOWLEDGE_CONTEXT_TOP_K", 3)
    monkeypatch.setattr(module, "KNOWLEDGE_RANK_SELECTION_ENABLED", False)
    monkeypatch.setattr(module, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", False)
    monkeypatch.setattr(module, "KNOWLEDGE_GENERATOR_TOP_K", 8)
    monkeypatch.setattr(module, "KNOWLEDGE_STATUS_TERMS", ("failed", "success", "succeeded", "échoué", "réussi"))

    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(ingestion, "datetime", FrozenClock)


def rule(key, kind="business_rule", **changes):
    record = {"_id": key, "type": kind, "text": f"Rule {key}", "collections": ["orders"],
            "fields": ["status"], "scope": "database", "active": True, "status": "active",
            "confidence": 0.9, "created_at": NOW, "updated_at": NOW,
            "embedding": [0.9, math.sqrt(1 - 0.9 ** 2)], **changes}
    record.update(embedding_fields(record["embedding"], definition_text(record), module.KNOWLEDGE_EMBEDDING_MODEL))
    return record


def generator(records, question="Show orders", fields=None, preferences=None):
    return KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        question, ["orders"], fields or {"orders": ["id"]},
        full_schema=CATALOG, user_preferences=preferences)


def ids(result):
    return [item["knowledge_id"] for item in result["knowledge"]]


def test_low_similarity_date_definition_is_scoped_without_embeddings():
    records = [rule("date", "field_definition", fields=["executionDate"],
                    text="orders.executionDate is stored as DD/MM/YYYY; days start at midnight.",
                    embedding=[0.01, 1.0])]
    assert ids(generator(records)) == ["date"]
    assert generator(records)["knowledge"][0]["inclusion_reasons"] == ["scope"]
    assert ids(KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_context(
        [1.0, 0.0], CATALOG)) == []


@pytest.mark.parametrize("question", ["today", "yesterday", "last week", "this month", "last 7 days",
    "2026-10-13", "13/10/2026", "October 13, 2026", "aujourd'hui", "aujourd’hui", "hier",
    "la semaine dernière", "ce mois", "les 5 derniers jours", "5 jours", "13 octobre 2026"])
def test_date_trigger_in_english_and_french(question):
    record = rule("date", "field_definition", collections=[], fields=["executionDate"])
    assert ids(generator([record], question)) == ["date"]
    assert generator([record], question)["knowledge"][0]["inclusion_reasons"] == ["trigger:date"]


def test_selected_date_field_triggers_and_empty_scope_is_not_universal():
    records = [rule("date", "field_definition", collections=[], fields=["executionDate"]),
               rule("empty", collections=[], fields=[])]
    assert ids(generator(records)) == []
    assert ids(generator(records, fields={"orders": ["executionDate"]})) == ["date"]


@pytest.mark.parametrize("question", ["failed", "success", "succeeded", "échoué", "réussi"])
def test_status_trigger_matches_empty_scope_relevant_fields(question):
    records = [rule("code", "coded_value", collections=[], fields=["status"], embedding=[]),
               rule("no-field", "coded_value", collections=[], fields=[])]
    result = generator(records, question)
    assert ids(result) == ["code"]
    assert result["knowledge"][0]["inclusion_reasons"] == ["trigger:status"]
    assert ids(generator(records, "unsuccessful")) == []


def test_status_terms_and_regexes_are_configurable(monkeypatch):
    monkeypatch.setattr(module, "KNOWLEDGE_STATUS_TERMS", ("KO",))
    assert ids(generator([rule("code", "coded_value", collections=[])], "KO")) == ["code"]
    monkeypatch.setattr(module, "KNOWLEDGE_TIME_EXPRESSION_PATTERN", r"\bpayday\b")
    monkeypatch.setattr(module, "KNOWLEDGE_DATE_FIELD_PATTERN", r"total")
    assert ids(generator([rule("date", "field_definition", collections=[], fields=["total"])], "payday")) == ["date"]


def test_recharge_scope_is_never_bypassed_by_triggers():
    records = [rule("recharge", "field_definition", collections=["Recharge"], fields=["rechargeDate"]),
               rule("recharge-status", "coded_value", collections=["Recharge"], fields=["status"])]
    assert ids(generator(records, "failed orders today")) == []


@pytest.mark.parametrize("changes", [{"fields": ["missing"]}, {"collections": ["missing"]},
    {"fields": ["rechargeDate"]}, {"status": "needs_review"}, {"active": False}])
def test_invalid_targets_and_inactive_rules_are_dropped(changes, caplog):
    assert ids(generator([rule("bad", **changes)])) == []
    assert "knowledge_id=bad reason=" in caplog.text


def test_flat_relationship_fields_are_validated_against_their_actual_owners():
    record = rule("join", "relationship", collections=["orders", "Recharge"], fields=["total", "rechargeDate"])
    assert ids(generator([record])) == ["join"]
    record["fields"].append("typo")
    assert ids(generator([record])) == []


@pytest.mark.parametrize("fields", [[], ["status"]])
def test_equivalent_duplicates_choose_confidence_then_recency(fields):
    records = [rule("old", text="Status meaning", fields=fields, confidence=0.8),
               rule("best", text="Status meaning", fields=fields, confidence=0.99),
               rule("new", text="  STATUS meaning. ", fields=fields, confidence=0.99,
                    updated_at=datetime(2026, 10, 14, tzinfo=timezone.utc))]
    assert ids(generator(records)) == ["new"]
    assert ids(generator(list(reversed(records)))) == ["new"]


@pytest.mark.parametrize("stage", ["context", "generator"])
@pytest.mark.parametrize("reverse", [False, True])
def test_latest_request_status_coexists_with_distinct_collection_rule(stage, reverse, caplog):
    from services.knowledge_ingestion_service import _candidate_id
    from services.prompt_context import knowledge_texts

    text = "The latest status of a request should be retrieved from the request_proc_hist collection"
    key = _candidate_id("business_rule", text, ["request_proc_hist"], [])
    assert key == "fa7ffe7f652c115f33fee58f44a08ebb676936a9678690f012887163ddc1b918"
    records = [rule(key, text=text, collections=["request_proc_hist"], fields=[], confidence=0.9),
               rule("other-instruction", text="Count processing events individually in request_proc_hist.",
                    collections=["request_proc_hist"], fields=[], confidence=0.99)]
    if reverse:
        records.reverse()
    catalog = {"collections": {"request_proc_hist": {"fields": {
        "requestId": ["string"], "status": ["string"]}}}}
    original = deepcopy(records)
    retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records))
    if stage == "context":
        result = retriever.rules_for_context([1.0, 0.0], catalog)
    else:
        result = retriever.rules_for_generator("What is the latest status of a request?",
            ["request_proc_hist"], {"request_proc_hist": ["requestId", "status"]}, full_schema=catalog)
    assert set(ids(result)) == {key, "other-instruction"}
    assert text in knowledge_texts(result["knowledge"])
    assert "reason=duplicate" not in caplog.text
    assert records == original


def test_conflicts_use_database_scope_then_confirmation_time_and_log_drops(caplog):
    records = [rule("old", confidence=1.0),
               rule("new", confidence=0.6, updated_at="2026-10-14T12:00:00Z"),
               rule("preference", scope="user", updated_at="2026-10-15T12:00:00Z")]
    with caplog.at_level(logging.INFO):
        assert ids(generator(records)) == ["new"]
    assert "knowledge_id=old kept_id=new reason=most_recent_confirmation" in caplog.text
    assert "knowledge_id=preference kept_id=new reason=database_scope_precedence" in caplog.text


def test_context_latest_record_rule_survives_newer_case_insensitive_rule(caplog):
    latest = rule("latest", text="Evaluate only the latest record per commandId, ordered by executionDate.",
                  fields=["id", "executionDate", "status"], confidence=0.5)
    case = rule("case", text="Text search is case-insensitive for status.", fields=["status"],
                updated_at="2026-10-15T12:00:00Z", confidence=1.0)
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: [latest, case])).rules_for_generator(
        "Show command status", ["orders"], {"orders": ["id", "executionDate", "status"]},
        full_schema=CATALOG, context_rules=[latest])
    assert ids(result) == ["latest"]
    assert result["knowledge"][0]["text"] == latest["text"]
    assert "context_selected" in result["knowledge"][0]["inclusion_reasons"]
    assert "knowledge_id=case reason=not_selected_in_context" in caplog.text
    assert "knowledge_id=latest kept_id=" not in caplog.text


def test_context_order_and_rules_survive_small_generator_cap(monkeypatch, caplog):
    monkeypatch.setattr(module, "KNOWLEDGE_GENERATOR_TOP_K", 1)
    records = [rule("relationship", "relationship", fields=[], confidence=0.1),
               rule("business", fields=[], confidence=0.99),
               rule("date", "field_definition", fields=["executionDate"])]
    selected = [records[0], records[1]]
    original = deepcopy((records, selected))
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        "orders", ["orders"], {"orders": ["executionDate"]}, full_schema=CATALOG, context_rules=selected)
    assert ids(result) == ["relationship", "business"]
    assert "effective_cap=2" in caplog.text
    assert (records, selected) == original


def test_context_rules_can_coexist_despite_overlapping_fields_and_different_confirmation_times():
    records = [rule("latest", fields=["status"], text="Only latest records", updated_at="2025-01-01"),
               rule("case", fields=["status"], text="Case-insensitive status", updated_at="2026-01-01")]
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        "status", ["orders"], {"orders": ["status"]}, full_schema=CATALOG, context_rules=records)
    assert ids(result) == ["latest", "case"]


def test_context_handoff_keeps_applicable_definitions_and_codes_but_excludes_recharge(caplog):
    source = rule("source", fields=[], text="Command status comes from orders")
    records = [source,
        rule("date", "field_definition", fields=["executionDate"], text="executionDate is an ISO string"),
        rule("status", "coded_value", fields=["status"], text="Failed means status Failed"),
        rule("recharge", "coded_value", fields=["code"], text="Recharge orders have code = 'Recharge'"),
        rule("unselected-business", fields=[], text="Use a date range for today")]
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        "give me status about this commandId: CMD-123", ["orders"],
        {"orders": ["id", "executionDate", "status", "code"]}, full_schema=CATALOG, context_rules=[source])
    assert ids(result) == ["source", "date", "status"]
    assert "knowledge_id=recharge reason=not_applicable_to_question" in caplog.text
    assert "knowledge_id=unselected-business reason=not_selected_in_context" in caplog.text


@pytest.mark.parametrize("question", ["Show Recharge orders", "Filter code Recharge", "Show code values"])
def test_context_handoff_includes_explicitly_requested_code_rules(question):
    records = [rule("recharge", "coded_value", fields=["code"], text="Recharge orders have code = 'Recharge'")]
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        question, ["orders"], {"orders": ["id", "code"]}, full_schema=CATALOG, context_rules=[])
    assert ids(result) == ["recharge"]


def test_context_rules_are_not_duplicated_and_conflicting_preferences_cannot_replace_them():
    source = rule("source", fields=["status"], text="Use latest records")
    preference = {"_id": "preference", "collection": "orders", "field": "status",
                  "type": "sorting_preference", "preference": "Sort status alphabetically"}
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: [source])).rules_for_generator(
        "status", ["orders"], {"orders": ["status"]}, full_schema=CATALOG,
        context_rules=[source, source], user_preferences=[preference])
    assert ids(result) == ["source"]
    assert result["user_preferences"] == []


@pytest.mark.parametrize("changes", [{"active": False}, {"collections": ["missing"]}, {"fields": ["missing"]}])
def test_preserved_context_rules_still_require_active_valid_schema_references(changes):
    bad = rule("bad", **changes)
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: [])).rules_for_generator(
        "orders", ["orders"], {"orders": ["id"]}, full_schema=CATALOG, context_rules=[bad])
    assert ids(result) == []


def test_partial_scope_conflicts_and_empty_triggered_scope_are_resolved(caplog):
    records = [rule("wide", collections=["orders", "Recharge"]),
               rule("date", "field_definition", fields=["executionDate"]),
               rule("empty-date", "field_definition", collections=[], fields=["executionDate"],
                    updated_at=datetime(2026, 10, 14, tzinfo=timezone.utc))]
    assert ids(generator(records, "today")) == ["wide", "empty-date"]
    assert "knowledge_id=date kept_id=empty-date" in caplog.text


def test_fixed_order_and_total_cap_include_preferences(caplog):
    records = [rule("source", text="Fetch orders from source collection", fields=[], confidence=0.95),
               rule("date", "field_definition", fields=["executionDate"], confidence=1.0),
               rule("coded", "coded_value", confidence=1.0),
               *[rule(f"other-{i}", fields=[f"f{i}"]) for i in range(8)]]
    catalog = deepcopy(CATALOG)
    catalog["collections"]["orders"]["fields"].update({f"f{i}": ["string"] for i in range(8)})
    preferences = [{"_id": "user", "collection": "orders", "field": "total", "type": "sorting_preference",
                    "preference": "Sort totals descending", "direction": -1}]
    result = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records)).rules_for_generator(
        "today", ["orders"], {"orders": ["id"]}, full_schema=catalog, user_preferences=preferences)
    assert ids(result) == ["source", *[f"other-{i}" for i in range(7)]]
    assert len(result["knowledge"]) == 8 and result["user_preferences"] == []
    assert "reason=total cap=8" in caplog.text


@pytest.mark.parametrize("reverse", [False, True])
def test_collection_matched_business_rules_lead_all_other_types_without_similarity(reverse):
    records = [rule("business", fields=["total"], confidence=0.2, embedding=[]),
               rule("relationship", "relationship", fields=[], confidence=1.0),
               rule("date", "field_definition", fields=["executionDate"], confidence=1.0),
               rule("coded", "coded_value", confidence=1.0),
               rule("misc", "custom_rule", fields=["id"], confidence=1.0),
               rule("unrelated", collections=["Recharge"], fields=["status"], confidence=1.0)]
    if reverse:
        records.reverse()
    preferences = [{"_id": "user", "collection": "orders", "field": "id", "type": "sorting_preference",
                    "preference": "Sort identifiers descending", "direction": -1, "confidence": 1.0}]
    result = generator(records, "failed orders today", preferences=preferences)
    assert ids(result) == ["business", "relationship", "date", "coded", "misc"]
    assert result["knowledge"][0]["inclusion_reasons"] == ["scope"]
    assert [item["_id"] for item in result["user_preferences"]] == ["user"]


def test_coded_value_cap_counts_shared_fields_across_different_rule_scopes(caplog):
    records = [rule("a", "coded_value", text="Status code meaning", fields=["status"]),
               rule("b", "coded_value", text="Status code meaning", fields=["status", "code"]),
               rule("c", "coded_value", text="Status code meaning", fields=["status", "total"])]
    assert ids(generator(records)) == ["a", "b"]
    assert "knowledge_id=c reason=coded_value_per_field cap=2" in caplog.text


def test_context_only_meaning_rules_sorted_by_raw_score_with_configurable_limits():
    records = [rule("a", fields=[]), rule("b", "relationship", fields=[], embedding=[1.0, 0.0]),
               rule("c", "field_definition", embedding=[1.0, 0.0]),
               rule("low", embedding=[0.70, math.sqrt(1 - 0.70 ** 2)]),
               rule("d", fields=["total"], embedding=[0.8, 0.6]),
               rule("e", fields=["id"], embedding=[0.76, math.sqrt(1 - 0.76 ** 2)])]
    retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records))
    assert ids(retriever.rules_for_context([1.0, 0.0], CATALOG)) == ["b", "a", "d"]
    assert ids(retriever.rules_for_context([1.0, 0.0], CATALOG, threshold=0.95, top_k=1)) == ["b"]
    assert ids(retriever.rules_for_context([], CATALOG)) == []
    assert ids(retriever.rules_for_context([1.0, 0.0], CATALOG, top_k=0)) == []


def test_lower_context_threshold_and_six_rules_admit_source_correction(monkeypatch):
    records = [rule(f"higher-{index}", fields=[], embedding=[score, math.sqrt(1 - score ** 2)])
               for index, score in enumerate([0.90, 0.85, 0.80, 0.70, 0.65])]
    records += [rule("latest-status-source", fields=[],
                     text="Retrieve latest order status from orders.",
                     embedding=[0.535, math.sqrt(1 - 0.535 ** 2)]),
                rule("over-budget", fields=[], embedding=[0.52, math.sqrt(1 - 0.52 ** 2)]),
                rule("below-threshold", fields=[], embedding=[0.49, math.sqrt(1 - 0.49 ** 2)])]
    retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: records))
    before = retriever.rules_for_context([1.0, 0.0], CATALOG, threshold=0.60, top_k=3)
    assert "latest-status-source" not in ids(before)
    monkeypatch.setattr(module, "KNOWLEDGE_CONTEXT_MIN_SIMILARITY", 0.50)
    monkeypatch.setattr(module, "KNOWLEDGE_CONTEXT_TOP_K", 6)
    result = retriever.rules_for_context([1.0, 0.0], CATALOG)
    assert len(result["knowledge"]) == 6
    assert ids(result)[-1] == "latest-status-source"
    assert "over-budget" not in ids(result) and "below-threshold" not in ids(result)
    assert result["telemetry"]["knowledge_retrieval_top_k"] == 6


@pytest.mark.parametrize("limit", [2, 12])
def test_generator_total_cap_is_configurable_and_counts_preferences(monkeypatch, caplog, limit):
    monkeypatch.setattr(module, "KNOWLEDGE_GENERATOR_TOP_K", limit)
    records = [rule(f"business-{index:02}", fields=[]) for index in range(13)]
    preferences = [{"_id": "user", "collection": "orders", "field": "total", "type": "sorting_preference",
                    "preference": "Sort totals descending", "direction": -1}]
    result = generator(records, preferences=preferences)
    assert len(result["knowledge"]) + len(result["user_preferences"]) == limit
    assert ids(result) == [f"business-{index:02}" for index in range(limit)]
    assert f"reason=total cap={limit}" in caplog.text


def test_source_can_change_collection_before_generator_gets_low_similarity_field_rule(monkeypatch, caplog):
    import orchestrator
    from agent.llm import llm_router

    records = [rule("source", text="Fetch orders from Recharge collection", collections=["Recharge"], fields=[]),
               rule("date", "field_definition", text="Recharge.rechargeDate is stored as a BSON date.",
                    collections=["Recharge"], fields=["rechargeDate"], embedding=[])]
    original = deepcopy((records, CATALOG))
    repository = SimpleNamespace(list_active=Mock(return_value=records))
    prompts = {}

    def invoke(role, messages, schema):
        prompts[role] = messages[0].content
        if role == "query_context":
            assert records[0]["text"] in prompts[role]
            assert records[1]["text"] not in prompts[role]
            assert '"Recharge":' in prompts[role]
            output = {"complexity": "simple", "operation": "find", "collections": ["Recharge"],
                      "fields": {"Recharge": ["id"]}}
        else:
            assert records[1]["text"] in prompts[role]
            output = {"semantic_intent": "Find orders", "template": {
                "operation": "find", "collection": "Recharge", "filter": {}, "limit": None}, "parameters": {}}
        return SimpleNamespace(content=json.dumps(output))

    monkeypatch.setattr(llm_router, "invoke", invoke)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: CATALOG)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", lambda *args: {
        "catalog": {"collections": {"orders": CATALOG["collections"]["orders"]}},
        "telemetry": {}, "_query_embedding": [1.0, 0.0]})
    app = orchestrator.OrchestratorAgent()
    app.knowledge_retriever = KnowledgeRetriever(repository)
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={
        "status": "safe", "query_signature": "sig", "estimated_rows": 0}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())
    result = app.run("Show orders today")
    assert result["execution_success"], result
    assert result["validated_query"]["collection"] == "Recharge"
    assert "rechargeDate" in result["schema_spec"]["collections"]["Recharge"]["fields"]
    repository.list_active.assert_called_once()
    for prompt in prompts.values():
        for metadata in ("knowledge_id", "inclusion_reasons", "confidence", "created_at", "semantic_similarity"):
            assert f'"{metadata}"' not in prompt
    assert "stage=context knowledge_id=source reasons=similarity" in caplog.text
    assert "stage=generator knowledge_id=date reasons=scope,trigger:date" in caplog.text
    assert (records, CATALOG) == original


@pytest.mark.parametrize("enabled", [True, False])
def test_creation_overlap_requires_review_even_without_auto_promotion(monkeypatch, enabled):
    import services.knowledge_ingestion_service as ingestion
    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", enabled)
    existing = rule("existing", "field_definition", fields=["executionDate"], text="Date is MM/DD/YYYY")
    stored = {}
    repository = Mock()
    repository.get_by_key.return_value = None
    repository.find_active_with_same_scope.return_value = [existing]
    repository.find_overlapping.return_value = [existing]

    def save(record):
        stored.update(record)
        return "created", stored

    repository.add_or_confirm.side_effect = save
    repository.set_status.side_effect = lambda key, status, active: stored.update(status=status, active=active) or True
    service = ingestion.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: CATALOG),
        embedding_provider=SimpleNamespace(embed_documents=lambda *args, **kwargs: [[1.0, 0.0]]))
    result = service.ingest({"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "field_definition", "collections": ["orders"], "fields": ["executionDate"],
        "corrected_meaning": "Date is DD/MM/YYYY", "confidence": 0.99}, "fake-feedback")
    assert result["status"] == "needs_review" and result["active"] is False
    assert stored["created_at"] == NOW
    repository.mark_conflict.assert_called_once_with(result["knowledge_id"], "existing")
    assert "knowledge_type" not in stored  # Existing storage field remains `type`.


def test_no_network_write_or_new_fields_in_read_only_repository():
    from database.knowledge_repository import KnowledgeRepository
    from agent.config import DATABASE_KNOWLEDGE_COLLECTION
    collection = Mock()
    collection.find.return_value = []
    repository = KnowledgeRepository({DATABASE_KNOWLEDGE_COLLECTION: collection})
    assert repository.list_active() == []
    assert [call[0] for call in collection.mock_calls] == ["find"]


def test_creation_finds_partial_overlap_with_inactive_candidate_using_real_repository(monkeypatch):
    from database.knowledge_repository import KnowledgeRepository
    from agent.config import DATABASE_KNOWLEDGE_COLLECTION
    import services.knowledge_ingestion_service as ingestion

    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", True)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_LEARNING_MODE", "immediate")
    pending = rule("pending", "field_definition", collections=["orders", "Recharge"],
                   fields=["executionDate", "rechargeDate"], status="candidate", active=False,
                   text="Dates use MM/DD/YYYY")

    class Collection:
        def __init__(self):
            self.records = {"pending": deepcopy(pending)}

        def find_one(self, query):
            return deepcopy(self.records.get(query["_id"]))

        def insert_one(self, record):
            self.records[record["_id"]] = deepcopy(record)

        def update_one(self, query, change):
            record = self.records[query["_id"]]
            record.update(change["$set"])
            return SimpleNamespace(matched_count=1)

        def find(self, query):
            def matches(record, filters):
                for key, value in filters.items():
                    if key == "$or":
                        if not any(matches(record, subquery) for subquery in value):
                            return False
                    elif isinstance(value, dict) and "$in" in value:
                        if not set(record.get(key, [])) & set(value["$in"]):
                            return False
                    elif record.get(key) != value:
                        return False
                return True
            return [deepcopy(record) for record in self.records.values() if matches(record, query)]

    collection = Collection()
    repository = KnowledgeRepository({DATABASE_KNOWLEDGE_COLLECTION: collection})
    service = ingestion.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: CATALOG),
        embedding_provider=SimpleNamespace(embed_documents=lambda *args, **kwargs: [[1.0, 0.0]]))
    analysis = {"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "field_definition", "collections": ["orders"], "fields": ["executionDate"],
        "corrected_meaning": "Dates use DD/MM/YYYY", "confidence": 0.99}
    result = service.ingest(analysis, "new-feedback")
    assert result["status"] == "needs_review" and result["active"] is False
    assert collection.records[result["knowledge_id"]]["conflict_with"] == "pending"
    # Non-overlapping rules retain the configured promotion behavior.
    independent = service.ingest({**analysis, "fields": ["total"], "corrected_meaning": "Total is numeric"}, "independent")
    assert independent["status"] == "active"


def test_review_candidate_does_not_auto_activate_when_overlap_disappears(monkeypatch):
    import services.knowledge_ingestion_service as ingestion
    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", True)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_LEARNING_MODE", "immediate")
    saved = rule("review", status="needs_review", active=False, confirmation_count=5, embedding=[1.0])
    repository = Mock()
    repository.get_by_key.return_value = saved
    repository.add_or_confirm.return_value = ("confirmation_added", saved)
    repository.find_overlapping.return_value = []
    service = ingestion.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: CATALOG), embedding_provider=Mock())
    result = service.ingest({"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "business_rule", "collections": ["orders"], "fields": ["status"],
        "corrected_meaning": saved["text"], "confidence": 0.99}, "repeat")
    assert result["status"] == "needs_review" and result["active"] is False
    repository.set_status.assert_not_called()


def test_preference_is_last_and_database_rule_wins_same_type_conflict(caplog):
    preferences = [{"_id": "user-sort", "collection": "orders", "field": "total", "type": "sorting_preference",
                    "preference": "Sort totals descending", "direction": -1},
                   {"_id": "user-status", "collection": "orders", "field": "status", "type": "filter_preference",
                    "preference": "Include all statuses", "confidence": 1.0}]
    result = generator([rule("database", text="Exclude failed orders")], preferences=preferences)
    assert ids(result) == ["database"]
    assert [item["_id"] for item in result["user_preferences"]] == ["user-sort"]
    assert "knowledge_id=user-status kept_id=database reason=database_scope_precedence" in caplog.text


def test_date_storage_definition_does_not_conflict_with_sorting_preference():
    preference = {"_id": "user-sort", "collection": "orders", "field": "executionDate",
                  "type": "sorting_preference", "preference": "Sort newest dates first", "direction": -1}
    result = generator([rule("date", "field_definition", fields=["executionDate"])], preferences=[preference])
    assert ids(result) == ["date"] and result["user_preferences"] == [preference]
