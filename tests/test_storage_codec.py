from datetime import datetime
from unittest.mock import Mock

from bson import ObjectId

from database.storage_codec import encode_record, decode_record
from database.query_template_repository import QueryTemplateRepository, template_signature
from database.query_history_repository import QueryHistoryRepository
from database.schema_cache_repository import SchemaCacheRepository
from agent.config import QUERY_HISTORY_COLLECTION, SCHEMA_CACHE_COLLECTION


def assert_legacy_safe(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert not key.startswith("$") and "." not in key
            assert_legacy_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_legacy_safe(child)


def test_round_trip_preserves_query_operators_dotted_fields_and_bson():
    record = {
        "status": "active", "success_count": 1,
        "template": {"pipeline": [{"$match": {"customer.id": ObjectId(),
            "created_at": {"$gte": datetime(2026, 1, 1)}}},
            {"$limit": {"$param": "limit"}}]},
        "schema_spec": {"collections": {"v2DB.orders": {"fields": {"customer.id": ["objectId"]}}}},
    }
    stored = encode_record(record)
    assert_legacy_safe(stored)
    assert stored["status"] == "active" and stored["success_count"] == 1
    assert decode_record(stored) == record
    assert decode_record(record) == record


def test_encoding_marker_collision_is_lossless():
    record = {"result": {"__data_pilot_extended_json_v1__": "user data"}}
    assert decode_record(encode_record(record)) == record


def test_literal_extended_json_operators_and_escape_sequences_remain_dicts():
    record = {"query": {"regex": {"$regex": "abc", "$options": "i"},
        "date": {"$date": "2026-01-01T00:00:00Z"}, "id": {"$oid": str(ObjectId())},
        "%24name": 1, "$name": 2, "a.b": 3, "a%2Eb": 4}}
    assert decode_record(encode_record(record)) == record


def test_template_write_and_read_preserve_signature_and_binding():
    collection = Mock()
    repository = QueryTemplateRepository({"_system_query_templates": collection})
    template = {"operation": "find", "collection": "orders",
        "filter": {"customer.id": {"$param": "customer_id"}}, "limit": 10}
    repository.record_success({"template": template, "parameters": {"customer_id": "123"}}, "req_test")
    update = collection.update_one.call_args.args[1]
    assert_legacy_safe(update["$setOnInsert"])
    assert_legacy_safe(update["$set"])
    collection.find_one.return_value = update["$setOnInsert"]
    loaded = repository.get("tpl_test")
    assert loaded["template"] == template
    assert loaded["template_signature"] == template_signature(template)


def test_history_save_update_and_read_handle_operator_payloads():
    collection = Mock()
    repository = QueryHistoryRepository({QUERY_HISTORY_COLLECTION: collection})
    record = {"request_id": "req_test", "created_at": datetime(2026, 1, 1),
        "validated_query": {"filter": {"name": {"$regex": "abc"}}}}
    repository.save(record)
    stored = collection.insert_one.call_args.args[0]
    assert_legacy_safe(stored)
    collection.find_one.return_value = stored
    assert repository.get_by_request_id("req_test")["validated_query"] == record["validated_query"]
    repository.update("req_test", record)
    assert_legacy_safe(collection.update_one.call_args.args[1]["$set"])


def test_schema_cache_round_trip_preserves_large_dotted_field_maps():
    collection = Mock()
    repository = SchemaCacheRepository({SCHEMA_CACHE_COLLECTION: collection})
    fields = {f"nested.field{i}": ["string"] for i in range(250)}
    record = {"database": "test", "collections": {"v2DB.orders": {"fields": fields}}}
    repository.save("test", record)
    stored = collection.replace_one.call_args.args[1]
    assert_legacy_safe(stored)
    collection.find_one.return_value = stored
    assert repository.get("test")["collections"] == record["collections"]
