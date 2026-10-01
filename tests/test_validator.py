import pytest

from agents.validator_agent import QueryValidationError, ValidatorAgent


SCHEMA = {
    "collections": {
        "orders": {
            "fields": {
                "_id": ["ObjectId"],
                "status": ["string"],
                "total": ["float"],
                "customerId": ["ObjectId"],
            }
        },
        "customers": {"fields": {"_id": ["ObjectId"], "country": ["string"]}},
    },
    "available_collections": ["orders", "customers"],
}


def test_accepts_read_only_find():
    result = ValidatorAgent().validate({
        "operation": "find",
        "collection": "orders",
        "filter": {"status": "paid"},
        "sort": [["total", -1]],
        "limit": 10,
    }, SCHEMA)
    assert result["limit"] == 10


def test_rejects_out_stage():
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({
            "operation": "aggregate",
            "collection": "orders",
            "pipeline": [{"$out": "exfil"}],
        }, SCHEMA)


def test_rejects_unknown_collection():
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({
            "operation": "find",
            "collection": "missing",
            "filter": {},
        }, SCHEMA)
