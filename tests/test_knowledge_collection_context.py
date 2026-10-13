from copy import deepcopy
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.query_context_agent import QueryContextAgent
from services.context_builder import ContextBuilder


def fixtures():
    full_catalog = {"collections": {
        "orders": {"fields": {"commandId": ["string"], "createdAt": ["date"]},
            "description": "Orders"},
        "orders_proc_hist": {"fields": {"commandId": ["string"], "executionDate": ["string"]}},
    }}
    candidates = {"collections": {"orders_proc_hist": deepcopy(full_catalog["collections"]["orders_proc_hist"])}}
    rules = [{"knowledge_id": "rule_commands", "type": "business_rule",
        "text": "Commands should be fetched from orders, not orders_proc_hist.",
        "collections": ["orders", "orders_proc_hist"], "fields": [], "confidence": 0.98}]
    return full_catalog, candidates, rules


def test_active_collection_correction_reaches_query_context_prompt(monkeypatch):
    import agents.query_context_agent as module
    full_catalog, candidates, rules = fixtures()
    original = deepcopy(candidates)
    context = ContextBuilder().build("commands in December 2025", candidates, rules, full_catalog=full_catalog)
    invoke = Mock(return_value=SimpleNamespace(content='{"complexity":"simple","operation":"find",'
        '"collections":["orders"],"fields":{"orders":["commandId","createdAt"]}}'))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    result = QueryContextAgent().build(context["question"], context["schema"],
        context["user_preferences"], context["database_knowledge"])
    prompt = invoke.call_args.args[1][0].content
    assert '"orders":{"fields":{"commandId":["string"],"createdAt":["date"]}' in prompt
    assert rules[0]["text"] in prompt
    assert result["query_context"]["collections"] == ["orders"]
    assert set(result["schema_spec"]["collections"]["orders"]["fields"]) == {"commandId", "createdAt"}
    assert candidates == original  # No mutation of the retrieval cache.


def test_unverified_rule_collection_is_not_invented():
    full_catalog, candidates, rules = fixtures()
    rules[0]["collections"] = ["missing_orders"]
    with pytest.raises(ValueError, match="unknown collection missing_orders"):
        ContextBuilder().build("commands", candidates, rules, full_catalog=full_catalog)


def test_no_knowledge_keeps_retrieved_catalog():
    full_catalog, candidates, _ = fixtures()
    context = ContextBuilder().build("commands", candidates, [], full_catalog=full_catalog)
    assert context["schema"] == candidates


@pytest.mark.parametrize("source_scope", [["manual_recycle"], ["manual_recycle", "orders_proc_hist"]])
def test_in_progress_source_missing_from_shortlist_reaches_both_prompts(monkeypatch, source_scope):
    import orchestrator
    from agent.llm import llm_router
    import services.knowledge_retriever as retrieval

    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_MIN_SIMILARITY", 0.50)
    full_catalog = {"collections": {
        "orders": {"fields": {"commandId": ["string"]}},
        "orders_proc_hist": {"fields": {"commandId": ["string"]}},
        "manual_recycle": {"fields": {"commandId": ["string"], "statusOrder": ["string"]}},
    }}
    candidates = {"collections": {"orders": deepcopy(full_catalog["collections"]["orders"])}}
    records = [{"_id": "in-progress-source", "type": "business_rule", "source": "user_feedback",
        "text": "Orders in progress must be fetched from the manual_recycle collection.",
        "collections": source_scope, "fields": [], "active": True, "status": "active",
        "embedding_model": retrieval.KNOWLEDGE_EMBEDDING_MODEL,
        "embedding": [0.60, math.sqrt(1 - 0.60 ** 2)]},
        {"_id": "in-progress-status", "type": "coded_value", "source": "user_feedback",
        "text": 'Orders in progress have statusOrder = "In Progress".',
        "collections": ["manual_recycle"], "fields": ["statusOrder"], "active": True, "status": "active",
        "embedding_model": retrieval.KNOWLEDGE_EMBEDDING_MODEL,
        "embedding": [0.55, math.sqrt(1 - 0.55 ** 2)]}]
    originals = deepcopy((full_catalog, candidates, records))
    prompts = {}

    def invoke(role, messages, schema):
        prompts[role] = messages[0].content
        if role == "query_context":
            # The actual prompt must contain the source before the mocked LLM
            # can choose it; the validator then checks the restored catalog.
            assert '"manual_recycle":' in prompts[role]
            output = {"complexity": "simple", "operation": "find", "collections": ["manual_recycle"],
                      "fields": {"manual_recycle": ["commandId"]}}
        else:
            output = {"semantic_intent": "Find orders in progress", "template": {
                "operation": "find", "collection": "manual_recycle",
                "filter": {"statusOrder": {"$param": "status"}}, "limit": 100},
                "parameters": {"status": "In Progress"}}
        return SimpleNamespace(content=json.dumps(output))

    monkeypatch.setattr(llm_router, "invoke", invoke)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "get_catalog", lambda: full_catalog)
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(orchestrator.schema_bootstrap_service, "retrieve", lambda *args: {
        "catalog": candidates, "telemetry": {}, "_query_embedding": [1.0, 0.0]})
    app = orchestrator.OrchestratorAgent()
    app.knowledge_retriever = retrieval.KnowledgeRetriever(SimpleNamespace(list_active=lambda: records))
    app.preference_service = Mock(user_id="test")
    app.preference_service.retrieve_for_query.return_value = []
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    monkeypatch.setattr(orchestrator.QueryCostGuard, "analyze", Mock(return_value={
        "status": "safe", "query_signature": "sig", "estimated_rows": 0}))
    monkeypatch.setattr(orchestrator.PerformanceLogger, "record", Mock())

    result = app.run("give me orders in progress")
    assert result["execution_success"], result
    assert result["query_context"]["collections"] == ["manual_recycle"]
    assert "statusOrder" in result["schema_spec"]["collections"]["manual_recycle"]["fields"]
    assert result["validated_query"]["collection"] == "manual_recycle"
    assert result["validated_query"]["filter"] == {"statusOrder": "In Progress"}
    assert len(result["applied_knowledge"]) == 2
    for role in ("query_context", "query_generator"):
        for record in (records[:1] if role == "query_context" else records):
            # JSON encodes the quotes inside the complete status rule.
            assert json.dumps(record["text"])[1:-1] in prompts[role]
        assert '"knowledge_id"' not in prompts[role]
        assert '"semantic_similarity"' not in prompts[role]
    assert json.dumps(records[1]["text"])[1:-1] not in prompts["query_context"]
    assert (full_catalog, candidates, records) == originals
