"""Offline coverage for lexical selection, rank floors and collection handoff."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.knowledge_retriever as retrieval
import services.knowledge_collection_suggestions as suggestions
from services.knowledge_embeddings import definition_text, embedding_fields
from services.knowledge_keywords import KeywordIndex, terms_for_rule

CATALOG = {"collections": {
    "requests": {"fields": {"requestId": ["string"], "status": ["string"]}},
    "request_proc_hist": {"fields": {"requestId": ["string"], "status": ["string"]}},
    "recharge": {"fields": {"status": ["string"]}},
    "archive": {"fields": {"status": ["string"]}},
}}


def rule(key, score=0.1, **changes):
    item = {"_id": key, "type": "business_rule", "text": "Instruction " + key,
            "collections": ["request_proc_hist"], "fields": [], "scope": "database",
            "active": True, "status": "active", "confidence": 0.95, **changes}
    vector = [score, math.sqrt(1 - score ** 2)]
    item.update(embedding_fields(vector, definition_text(item), retrieval.KNOWLEDGE_EMBEDDING_MODEL))
    return item


def fillers():
    return [rule("alpha", text="Billing invoice settlement"), rule("beta", text="Warehouse inventory stock"),
            rule("gamma", text="Customer address region")]


def retriever(records, service=None):
    return retrieval.KnowledgeRetriever(SimpleNamespace(list_active=lambda: deepcopy(records)),
        embedding_service=service or SimpleNamespace(ensure_embeddings=lambda item, **kwargs: deepcopy(item)))


def retrieve(records, question="", **kwargs):
    return retriever(records).rules_for_context([1., 0.], CATALOG, question=question, **kwargs)


def ids(result):
    return [item["knowledge_id"] for item in result["knowledge"]]


def diagnostic(caplog):
    logs = [entry.message for entry in caplog.records if entry.message.startswith("knowledge_context_diagnostics ")]
    assert len(logs) == 1
    return json.loads(logs[0].split(" data=", 1)[1])


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_MIN_SIMILARITY", 0.75)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_TOP_K", 3)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_FLOOR", 0.30)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_RANK_SELECTION_ENABLED", True)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", True)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION", 0.30)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MAX_EXTRA", 2)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(suggestions, "KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED", True)
    monkeypatch.setattr(suggestions, "KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX", 2)


def test_source_at_039_selected_by_rank_floor(caplog):
    result = retrieve([rule("source", 0.39)])
    assert ids(result) == ["source"]
    assert result["knowledge"][0]["selection_method"] == "rank_floor"
    assert diagnostic(caplog)["selected"][0]["selection_method"] == "rank_floor"


@pytest.mark.parametrize("score", [0.29, 0., -0.2])
def test_below_floor_without_keyword_not_selected(score):
    assert ids(retrieve([rule("source", score)])) == []


def test_rank_floor_only_applies_within_top_k():
    records = [rule(str(index), score) for index, score in enumerate([0.9, 0.6, 0.5, 0.39])]
    assert ids(retrieve(records)) == ["0", "1", "2"]


@pytest.mark.parametrize("term,question", [("recharge", "Show Recharge"), ("historique", "Voir HISTORÍQUE"),
                                            ("échoué", "Show echoue")])
def test_keyword_selects_very_low_score_and_logs_reason(term, question, caplog):
    result = retrieve([rule("source", 0.01, text=f'"{term}" means processing evidence'), *fillers()], question)
    assert ids(result) == ["source"]
    item = result["knowledge"][0]
    assert item["selection_method"] == "keyword"
    assert any(reason.startswith("keyword:") for reason in item["inclusion_reasons"])
    assert diagnostic(caplog)["selected"][0]["keyword_terms"]


@pytest.mark.parametrize("question", ["prerecharge", "rechargesuffix", "prehistoriques"])
def test_keyword_word_boundaries(question):
    assert ids(retrieve([rule("source", text='"recharge" means historique'), *fillers()], question)) == []


def test_generic_terms_are_filtered():
    records = [rule(str(index), text=f"status commonword unique{index}") for index in range(4)]
    index = KeywordIndex()
    assert index.match("status commonword", records, ttl=3600, max_fraction=0.30) == {}
    assert list(index.match("unique0", records, ttl=3600, max_fraction=0.30)) == ["0"]


def test_keyword_unknown_collection_still_dropped(caplog):
    result = retrieve([rule("source", text="recharge evidence", collections=["missing"]), *fillers()], "recharge")
    assert ids(result) == []
    assert {item["reason"] for item in diagnostic(caplog)["skipped"]} >= {"unknown_collection"}


@pytest.mark.parametrize("change", ["model", "length", "empty", "refresh"])
def test_bad_vectors_do_not_block_keyword(change):
    record = rule("source", text="recharge evidence")
    if change == "model": record["embedding_model"] = "other"
    if change == "length": record["embedding"] = [1., 0., 0.]
    if change == "empty": record["embedding"] = []
    service = SimpleNamespace(ensure_embeddings=Mock(side_effect=RuntimeError("fake failure"))) if change == "refresh" else None
    result = retriever([record, *fillers()], service).rules_for_context([1., 0.], CATALOG, question="recharge")
    assert ids(result) == ["source"]
    assert result["knowledge"][0]["selection_method"] == "keyword"


def test_keyword_works_when_query_embedding_unavailable():
    result = retriever([rule("source", text="recharge evidence"), *fillers()]).rules_for_context(
        None, CATALOG, question="recharge")
    assert ids(result) == ["source"]


def test_keyword_extras_and_total_cap(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_TOP_K", 1)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION", 1.)
    records = [rule("semantic", 0.95)] + [rule(str(index), text=f"recharge evidence{index}") for index in range(5)]
    result = retrieve(records, "recharge")
    assert len(result["knowledge"]) == 3
    assert ids(result)[0] == "semantic"
    assert sum(item["selection_method"] == "keyword" for item in result["knowledge"]) == 2


def test_keyword_overflow_above_threshold_can_use_extra_slot(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_TOP_K", 1)
    result = retrieve([rule("semantic", 0.99), rule("source", 0.9, text="recharge evidence"), *fillers()], "recharge")
    assert ids(result) == ["semantic", "source"]
    assert result["knowledge"][1]["selection_method"] == "keyword"


def test_flags_off_restore_threshold_only(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_RANK_SELECTION_ENABLED", False)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", False)
    records = [rule("high", 0.9), rule("middle", 0.39), rule("low", text="recharge evidence"), *fillers()]
    assert ids(retrieve(records, "recharge")) == ["high"]


def test_keyword_flag_and_configurable_floor(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", False)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_FLOOR", 0.40)
    assert ids(retrieve([rule("middle", 0.39), rule("source", text="recharge evidence"), *fillers()], "recharge")) == []


def test_keyword_zero_extra_and_zero_suggestion_cap(monkeypatch):
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_TOP_K", 1)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MAX_EXTRA", 0)
    records = [rule("high", 0.9), rule("source", text="recharge evidence"), *fillers()]
    assert ids(retrieve(records, "recharge")) == ["high"]
    monkeypatch.setattr(suggestions, "KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX", 0)
    context = {"collections": ["requests"]}
    assert suggestions.apply_rule_suggestions(context, {"collections": {}}, records, CATALOG)[2] == []


def test_keyword_matches_still_resolve_conflicts_and_do_not_mutate_records():
    records = [rule("older", text="recharge older", fields=["status"], updated_at="2025-01-01"),
               rule("newer", text="recharge newer", fields=["status"], updated_at="2026-01-01"), *fillers()]
    original = deepcopy(records)
    instance = retriever(records)
    # Two of five rules share the word: allow it for this conflict test.
    instance._keyword_index.match("recharge", records, ttl=3600, max_fraction=1.)
    result = instance.rules_for_context([1., 0.], CATALOG, question="recharge older newer")
    # Distinctive words older/newer qualify independently under the 30% filter.
    assert ids(result) == ["newer"]
    assert records == original


def test_sensitive_keyword_is_redacted_before_truncation(monkeypatch, caplog):
    secret = "sensitivecredential" * 8
    monkeypatch.setenv("TEST_API_KEY", secret)
    retrieve([rule("source", text=secret), *fillers()], secret)
    assert secret[:80] not in caplog.text
    assert "[redacted]" in caplog.text


def test_index_reuses_cache_and_rebuilds_on_text_collection_changes_and_ttl(monkeypatch):
    import services.knowledge_keywords as module
    clock = Mock(return_value=0.)
    monkeypatch.setattr(module.time, "monotonic", clock)
    records = [rule("source", text="recharge evidence"), *fillers()]
    index = KeywordIndex()
    assert index.match("recharge", records, ttl=10, max_fraction=0.30)
    cached = index._index
    clock.return_value = 5.
    index.match("recharge", records, ttl=10, max_fraction=0.30)
    assert index._index is cached
    records[0]["text"] = "historique evidence"
    assert index.match("historique", records, ttl=10, max_fraction=0.30)
    assert index._index is not cached
    cached = index._index
    records[0]["collections"] = ["archive"]
    assert index.match("archive", records, ttl=10, max_fraction=0.30)
    assert index._index is not cached
    cached = index._index
    clock.return_value = 16.
    index.match("archive", records, ttl=10, max_fraction=0.30)
    assert index._index is not cached


def test_keyword_sources_and_stopwords():
    terms = terms_for_rule({"text": '"Recharge" means billing; status = "Failed"; historique is Archive.',
                            "collections": ["request_proc_hist"]})
    assert {"recharge", "billing", "failed", "historique", "archive", "request_proc_hist"} <= terms
    assert not {"the", "means", "is", "est", "les"} & terms


def test_suggested_collection_reaches_generator_scope_and_conflict_log(caplog):
    records = [rule("source", 0.39), rule("status", type="coded_value", fields=["status"])]
    instance = retriever(records)
    rules = instance.rules_for_context([1., 0.], CATALOG)["knowledge"]
    context = {"collections": ["requests"], "fields": {"requests": ["requestId"]}, "complexity": "simple", "operation": "find"}
    schema = {"collections": {"requests": deepcopy(CATALOG["collections"]["requests"])}}
    original = deepcopy((context, schema, CATALOG))
    context, schema, added = suggestions.apply_rule_suggestions(context, schema, rules, CATALOG)
    assert added == ["request_proc_hist"]
    assert context["collection_selection_reasons"][added[0]] == "rule_suggested"
    assert context["collections"] == ["requests", "request_proc_hist"]
    assert context["operation"] == "find"  # Suggestions must not trigger another agent call.
    assert "rule_suggested_conflict" in caplog.text
    generator = instance.rules_for_generator("status", context["collections"], context["fields"], full_schema=CATALOG)
    assert set(ids(generator)) == {"source", "status"}
    assert "status" in schema["collections"][added[0]]["fields"]
    assert original[2] == CATALOG
    assert original[0]["collections"] == ["requests"]


def test_suggestion_cap_unknown_existing_and_disabled(monkeypatch):
    context = {"collections": ["requests"], "fields": {"requests": ["status"]}}
    schema = {"collections": {"requests": CATALOG["collections"]["requests"]}}
    knowledge = [rule("source", collections=["missing", "requests", "request_proc_hist", "recharge", "archive"])]
    updated, _, added = suggestions.apply_rule_suggestions(context, schema, knowledge, CATALOG)
    assert added == ["request_proc_hist", "recharge"]
    assert "archive" not in updated["collections"]
    monkeypatch.setattr(suggestions, "KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED", False)
    assert suggestions.apply_rule_suggestions(context, schema, knowledge, CATALOG) == (context, schema, [])


def test_keyword_invalid_scope_and_fields_still_dropped():
    records = [rule("user", text="recharge", scope="user"),
               rule("field", text="recharge", fields=["missing"]), *fillers()]
    assert ids(retrieve(records, "recharge")) == []


def test_command_source_rule_is_excluded_from_request_context_even_at_high_similarity(caplog):
    records = [rule("command", 0.99, text="Status about commands is collected from orders_proc_hist."),
               rule("request", 0.39, text="The latest status of a request should be retrieved from request_proc_hist.")]
    result = retrieve(records, "give me status about this request REQ-220148108")
    assert ids(result) == ["request"]
    assert any(item["rule_id"] == "command" and item["reason"] == "entity_mismatch"
               for item in diagnostic(caplog)["skipped"])


@pytest.mark.parametrize("text,compatible", [
    ("Execution dates are ISO strings", True),
    ("Requests join commands by their identifier", True),
    ("Request status comes from request_proc_hist", True),
    ("Command status comes from orders_proc_hist", False),
])
def test_rule_entity_guard_retains_generic_and_relationship_rules(text, compatible):
    from services.query_entity_guard import rule_entity_mismatch
    assert rule_entity_mismatch({"text": text}, "status REQ-220148108") is not compatible


def test_multi_entity_question_can_keep_separate_rules():
    from services.query_entity_guard import rule_entity_mismatch
    question = "Compare command CMD-123 with request REQ-456"
    assert not rule_entity_mismatch({"text": "Command status comes from orders_proc_hist"}, question)
    assert not rule_entity_mismatch({"text": "Request status comes from request_proc_hist"}, question)


@pytest.mark.parametrize("context_outcome", ["success", "error", "exception"])
def test_orchestrator_reuses_one_question_vector_and_carries_omitted_rule_collection(monkeypatch, caplog, context_outcome):
    import orchestrator
    import services.schema_retriever as schema_module
    from agent.llm import llm_router
    records = [rule("source", 0.39, text="Request status comes from request_proc_hist"),
               rule("status", type="coded_value", fields=["status"]),
               rule("wrong-command", 0.99, text="Command status comes from orders_proc_hist",
                    collections=["orders_proc_hist"])]
    catalog = deepcopy(CATALOG)
    catalog["collections"]["orders_proc_hist"] = {"fields": {"commandId": ["string"], "status": ["string"]}}
    embed = Mock(return_value=[1., 0.])
    monkeypatch.setattr(llm_router, "embed_query", embed)
    schema_retriever = schema_module.SchemaRetriever(catalog, top_k=1)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: deepcopy(catalog))
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", schema_retriever.retrieve)
    app = orchestrator.OrchestratorAgent()
    app.knowledge_retriever = retriever(records)
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.query_context_agent = Mock()
    app.query_context_agent.build.return_value = {"query_context": {"complexity": "simple", "operation": "find",
        "collections": ["requests"], "fields": {"requests": ["requestId"]}},
        "schema_spec": {"collections": {"requests": deepcopy(CATALOG["collections"]["requests"])}}}
    if context_outcome == "error":
        app.query_context_agent.build.return_value = {"error": "fake_error", "detail": "Context unavailable"}
    elif context_outcome == "exception":
        app.query_context_agent.build.side_effect = RuntimeError("fake context failure")
    app.query_generator = Mock()
    app.query_generator.generate.return_value = {"template": {"operation": "find", "collection": "request_proc_hist",
        "filter": {}, "limit": 100}, "parameters": {}}
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    app.summary_agent = Mock()
    app.summary_agent.summarize.return_value = {"summary_text": "No results"}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "fake"}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())
    result = app.run("request status")
    embed.assert_called_once_with("request status")
    data = diagnostic(caplog)
    assert data["question_embedding_reused"] is True
    if context_outcome != "success":
        assert result["execution_success"] is False
        assert data["rule_suggested_collections_added"] == []
        app.query_generator.generate.assert_not_called()
        return
    assert result["execution_success"], result
    assert "request_proc_hist" in result["query_context"]["collections"]
    assert {item["knowledge_id"] for item in result["applied_knowledge"]} == {"source", "status"}
    assert data["rule_suggested_collections_added"] == ["request_proc_hist"]
