from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent.config import MONGO_MAX_COLLECTIONS, MONGO_SCHEMA_SAMPLE_SIZE
from database.mongo_client import get_database


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _walk(value: Any, prefix: str, out: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out[path].add(_type_name(child))
            _walk(child, path, out)
    elif isinstance(value, list):
        for child in value[:5]:
            if isinstance(child, dict):
                _walk(child, prefix, out)
            elif prefix:
                out[prefix].add(f"array<{_type_name(child)}>")


def inspect_database_schema() -> dict[str, Any]:
    db = get_database()
    collection_names = sorted(db.list_collection_names())[:MONGO_MAX_COLLECTIONS]
    collections: dict[str, Any] = {}

    for name in collection_names:
        fields: dict[str, set[str]] = defaultdict(set)
        docs = list(db[name].find({}, limit=MONGO_SCHEMA_SAMPLE_SIZE))
        for doc in docs:
            _walk(doc, "", fields)
        collections[name] = {
            "sampled_documents": len(docs),
            "fields": {path: sorted(types) for path, types in sorted(fields.items())},
        }

    return {
        "database": db.name,
        "collection_count": len(collections),
        "collections": collections,
    }


def select_schema(full_schema: dict[str, Any], names: list[str]) -> dict[str, Any]:
    all_collections = full_schema.get("collections", {})
    selected = {name: all_collections[name] for name in names if name in all_collections}
    return {
        "database": full_schema.get("database"),
        "collections": selected,
    }


def compact_schema(schema):
    """Only type metadata crosses the LLM boundary."""
    return {name: {"fields": info.get("fields", {})} for name, info in schema.get("collections", {}).items()}
