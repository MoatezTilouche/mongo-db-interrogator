from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import logging
from threading import RLock
import time
from typing import Any

from agent.config import (
    FEEDBACK_COLLECTION,
    MONGO_MAX_COLLECTIONS,
    MONGO_SCHEMA_SAMPLE_SIZE,
    QUERY_HISTORY_COLLECTION,
    SCHEMA_CACHE_ENABLED,
    SCHEMA_CACHE_TTL_SECONDS,
)
from database.mongo_client import get_database

logger = logging.getLogger(__name__)
_schema_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_schema_cache_lock = RLock()


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


def _inspect_database_schema() -> dict[str, Any]:
    db = get_database()
    internal = {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION}
    collection_names = sorted(name for name in db.list_collection_names() if name not in internal)[:MONGO_MAX_COLLECTIONS]
    collections: dict[str, Any] = {}

    for name in collection_names:
        fields: dict[str, set[str]] = defaultdict(set)
        docs = list(db[name].find({}, limit=MONGO_SCHEMA_SAMPLE_SIZE))
        for doc in docs:
            _walk(doc, "", fields)
        collections[name] = {
            "fields": {path: sorted(types) for path, types in sorted(fields.items())},
        }

    return {
        "database": db.name,
        "collection_count": len(collections),
        "collections": collections,
    }


def _with_cache_status(schema: dict[str, Any], **status: bool) -> dict[str, Any]:
    result = deepcopy(schema)
    result["_schema_cache"] = {
        "hit": status.get("hit", False),
        "miss": status.get("miss", False),
        "refresh": status.get("refresh", False),
        "stale_fallback": status.get("stale_fallback", False),
    }
    return result


def inspect_database_schema(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return the cached compact schema, inspecting MongoDB only on a miss/refresh."""
    db = get_database()
    key = db.name
    now = time.monotonic()
    with _schema_cache_lock:
        entry = _schema_cache.get(key)
        stale = entry[1] if entry and SCHEMA_CACHE_ENABLED else None
        if not SCHEMA_CACHE_ENABLED:
            _schema_cache.pop(key, None)
        fresh = bool(entry and now - entry[0] < SCHEMA_CACHE_TTL_SECONDS)

        if SCHEMA_CACHE_ENABLED and fresh and not force_refresh:
            logger.info("schema_cache_hit database=%s", key)
            return _with_cache_status(entry[1], hit=True)

        cache_miss = not force_refresh
        logger.info("schema_cache_miss database=%s forced_refresh=%s enabled=%s", key, force_refresh, SCHEMA_CACHE_ENABLED)
        try:
            inspected = _inspect_database_schema()
        except Exception as exc:
            if stale is not None:
                logger.warning("schema_cache_stale_fallback database=%s error_type=%s", key, type(exc).__name__)
                return _with_cache_status(stale, miss=cache_miss, stale_fallback=True)
            raise

        if SCHEMA_CACHE_ENABLED:
            _schema_cache[key] = (time.monotonic(), deepcopy(inspected))
        else:
            _schema_cache.pop(key, None)
        logger.info("schema_cache_refresh database=%s", key)
        return _with_cache_status(inspected, miss=cache_miss, refresh=True)


def invalidate_cache(database_name: str | None = None) -> None:
    """Invalidate one database's schema, or the whole process-local cache."""
    with _schema_cache_lock:
        if database_name is None:
            _schema_cache.clear()
        else:
            _schema_cache.pop(database_name, None)
    logger.info("schema_cache_invalidated database=%s", database_name or "*")


def refresh_schema() -> dict[str, Any]:
    """Force a fresh MongoDB inspection and replace the cached schema."""
    return inspect_database_schema(force_refresh=True)


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
