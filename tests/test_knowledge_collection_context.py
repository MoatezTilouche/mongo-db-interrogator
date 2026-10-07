from copy import deepcopy
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
