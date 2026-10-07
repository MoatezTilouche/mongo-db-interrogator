"""Store arbitrary query/schema payloads without MongoDB-reserved field names.

Searchable top-level metadata stays native. Only values containing unsafe keys
are encoded as Extended JSON, preserving BSON dates and ObjectIds on reads.
Legacy native documents remain readable.
"""
from bson import json_util

_MARKER = "__data_pilot_extended_json_v1__"


def _transform_keys(value, decode=False):
    if isinstance(value, dict):
        def transform(key):
            if decode:
                return key.replace("%24", "$").replace("%2E", ".").replace("%25", "%")
            return key.replace("%", "%25").replace("$", "%24").replace(".", "%2E")
        return {transform(key): _transform_keys(child, decode) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_transform_keys(child, decode) for child in value]
    return value


def _needs_encoding(value):
    if isinstance(value, dict):
        return _MARKER in value or any(
            key.startswith("$") or "." in key or _needs_encoding(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_needs_encoding(child) for child in value)
    return False


def encode_record(record):
    return {
        # Escape literal operator keys before Extended JSON parsing can mistake
        # query syntax such as $regex/$date for native BSON values.
        key: {_MARKER: json_util.dumps(_transform_keys(value))} if _needs_encoding(value) else value
        for key, value in record.items()
    }


def decode_record(record):
    if record is None:
        return None
    return {
        key: _transform_keys(json_util.loads(value[_MARKER]), decode=True)
        if isinstance(value, dict) and set(value) == {_MARKER} else value
        for key, value in record.items()
    }
