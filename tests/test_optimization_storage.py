"""Optimization against a store enforcing pre-5.0 MongoDB field restrictions."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pymongo.errors import WriteError

from agent.config import QUERY_HISTORY_COLLECTION
from database.storage_codec import decode_record
from services.query_parameter_binder import template_from_concrete, parameterize_filter_values
import services.query_optimization_service as module


def assert_storable(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith("$") or "." in key:
                raise WriteError("Reserved field name in stored document", 52)
            assert_storable(child)
    elif isinstance(value, list):
        for child in value:
            assert_storable(child)


class StrictCollection:
    def __init__(self):
        self.records = {}

    def create_index(self, *args, **kwargs):
        pass

    def insert_one(self, record):
        assert_storable(record)
        self.records[record["_id"]] = deepcopy(record)

    def update_one(self, query, update):
        assert_storable(update.get("$set", {}))
        record = self.records.get(query["_id"])
        if record is not None and ("claim" not in query or record.get("claim") == query["claim"]):
            record.update(deepcopy(update.get("$set", {})))

    def find_one_and_update(self, query, update):
        record = self.records.get(query["_id"])
        if record is None or record["state"] not in query["state"]["$in"]:
            return None
        before = deepcopy(record)
        self.update_one({"_id": query["_id"]}, update)
        return before


def setup_service(monkeypatch, still_heavy=False):
    pending, history = StrictCollection(), StrictCollection()
    database = {"_system_pending_queries": pending, QUERY_HISTORY_COLLECTION: history}
    monkeypatch.setattr(module, "get_database", lambda: database)
    query = {"operation": "find", "collection": "inventory_items",
        "filter": {"attrs.status": {"$regex": "available", "$options": "i"}},
        "limit": 100, "skip": 0, "projection": {}, "sort": [["_id", 1]]}
    generated = parameterize_filter_values(template_from_concrete(query, True))
    result = {"request_id": "req_test", "status": "optimization_required",
        "validated_query": query, "generated_query": generated,
        "query_context": {"collections": ["inventory_items"]},
        "schema_spec": {"collections": {"inventory_items": {"fields": {
            "attrs.status": ["string"], "_id": ["ObjectId"]}}}},
        "cost_analysis": {"estimated_rows": 1000},
        "suggestions": [{"id": choice} for choice in ["latest_100", "paginate_100", "refine", "run_anyway"]]}
    history.insert_one({"_id": "req_test"})
    service = module.QueryOptimizationService()
    service.pause(result, "Show available inventory")
    executor = Mock(return_value={"ok": True, "all_rows": [{"x": 1}, {"x": 2}],
        "row_count": 2, "execution_time_ms": 1})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    monkeypatch.setattr(module.QueryGeneratorAgent, "optimize", Mock(return_value=generated))
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={
        "status": "heavy" if still_heavy else "safe", "query_signature": "sig", "estimated_rows": 1000}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    return service, pending, history, result, executor


def payload(choice, page=1):
    return SimpleNamespace(optimization=choice, value=2, page=page, refinement="Only available inventory")


@pytest.mark.parametrize("choice", ["latest_100", "refine", "run_anyway"])
def test_optimization_completion_and_history_are_safe_and_lossless(monkeypatch, choice):
    service, pending, history, original, executor = setup_service(monkeypatch)
    assert decode_record(pending.records["req_test"])["result"] == original
    result = service.continue_query("req_test", payload(choice))
    assert result["execution_success"]
    stored = decode_record(pending.records["req_test"])
    assert stored["state"] == "completed"
    assert stored["result"]["generated_query"] == result["generated_query"]
    assert decode_record(history.records["req_test"])["validated_query"] == result["validated_query"]
    executor.assert_called_once()


def test_still_heavy_query_can_be_saved_and_resumed(monkeypatch):
    service, pending, history, _, executor = setup_service(monkeypatch, still_heavy=True)
    result = service.continue_query("req_test", payload("refine"))
    assert result["status"] == "optimization_required"
    assert decode_record(pending.records["req_test"])["state"] == "pending"
    assert decode_record(history.records["req_test"])["generated_query"] == result["generated_query"]
    executor.assert_not_called()
    result = service.continue_query("req_test", payload("run_anyway"))
    assert result["execution_success"]


def test_pagination_restores_encoded_base_query_on_next_page(monkeypatch):
    service, pending, history, original, executor = setup_service(monkeypatch)
    first = service.continue_query("req_test", payload("paginate_100"))
    assert first["pagination"]["page"] == 1
    saved = decode_record(pending.records["req_test"])
    assert saved["pagination_base_query"] == original["validated_query"]
    second = service.continue_query("req_test", payload("pagination", page=2))
    assert second["pagination"]["page"] == 2
    assert executor.call_args.args[-1]["skip"] == 2
    assert executor.call_args.args[-1]["filter"] == original["validated_query"]["filter"]
    assert decode_record(history.records["req_test"])["validated_query"] == second["validated_query"]


def test_existing_native_pending_state_remains_readable(monkeypatch):
    service, pending, _, _, _ = setup_service(monkeypatch)
    # Native records from a newer MongoDB deployment need no migration script.
    pending.records["req_test"] = decode_record(pending.records["req_test"])
    assert service.continue_query("req_test", payload("latest_100"))["execution_success"]
    assert_storable(pending.records["req_test"])


def test_aggregation_pause_preserves_stages_and_dotted_schema(monkeypatch):
    service, pending, _, original, _ = setup_service(monkeypatch)
    query = {"operation": "aggregate", "collection": "inventory_items", "pipeline": [
        {"$match": {"attrs.status": {"$regex": "available", "$options": "i"}}},
        {"$group": {"_id": "$attrs.status", "count": {"$sum": 1}}}]}
    result = {**original, "request_id": "req_aggregate", "validated_query": query,
        "generated_query": parameterize_filter_values(template_from_concrete(query))}
    service.pause(result, "Count inventory by status")
    assert decode_record(pending.records["req_aggregate"])["result"] == result
