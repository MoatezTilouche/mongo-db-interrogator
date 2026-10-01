from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any


def decode_special_values(value: Any) -> Any:
    """Convert a small safe subset of Mongo Extended JSON wrappers."""
    if isinstance(value, list):
        return [decode_special_values(item) for item in value]
    if not isinstance(value, dict):
        return value

    if set(value) == {"$oid"} and isinstance(value["$oid"], str):
        from bson import ObjectId
        return ObjectId(value["$oid"])
    if set(value) == {"$date"} and isinstance(value["$date"], str):
        text = value["$date"].replace("Z", "+00:00")
        return datetime.fromisoformat(text)

    return {key: decode_special_values(item) for key, item in value.items()}


def to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    try:
        from bson import ObjectId, Decimal128
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, Decimal128):
            return str(value.to_decimal())
    except Exception:
        pass
    return str(value)
