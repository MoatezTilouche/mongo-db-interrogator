"""Deterministic validator for LLM-generated MongoDB query specifications."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.config import (DEFAULT_LIMIT, MAX_LIMIT, QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION,
                          USER_PREFERENCES_COLLECTION, SCHEMA_CACHE_COLLECTION,
                          DATABASE_KNOWLEDGE_COLLECTION, QUERY_PERFORMANCE_COLLECTION, QUERY_TEMPLATES_COLLECTION)
from agent.tracing import traceable
from services.aggregation_validation import QueryValidationError, validate_pipeline
from services.mongo_capabilities import get_mongo_capabilities


ALLOWED_OPERATIONS = {"find", "aggregate", "count_documents", "distinct"}
ALLOWED_STAGES = {
    "$match", "$project", "$group", "$sort", "$limit", "$skip", "$unwind",
    "$lookup", "$addFields", "$set", "$unset", "$count", "$facet",
    "$replaceRoot", "$replaceWith", "$sortByCount", "$bucket", "$bucketAuto",
}
BANNED_KEYS = {
    "$where", "$function", "$accumulator", "$out", "$merge",
}


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_no_banned_keys(value: Any) -> None:
    for key, _ in _walk_keys(value):
        if key in BANNED_KEYS:
            raise QueryValidationError(f"Banned MongoDB operator/stage: {key}")


def _known_collections(schema_spec: dict[str, Any]) -> set[str]:
    names = set(schema_spec.get("collections", {}).keys())
    names.update(schema_spec.get("available_collections", []))
    return {str(name) for name in names}


def _known_fields(schema_spec: dict[str, Any], collection: str) -> set[str]:
    fields = schema_spec.get("collections", {}).get(collection, {}).get("fields", {})
    return set(fields.keys())


def _field_is_known(field: str, known_fields: set[str]) -> bool:
    # MongoDB collection documents have _id even when compact retrieval omits it.
    if field == "_id":
        return True
    root = field.split(".", 1)[0]
    return field in known_fields or root in {item.split(".", 1)[0] for item in known_fields}


def _validate_filter_fields(value: Any, known_fields: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key = str(key)
            if not key.startswith("$") and known_fields and not _field_is_known(key, known_fields):
                raise QueryValidationError(f"Unknown field in filter/expression: {key}")
            _validate_filter_fields(child, known_fields)
    elif isinstance(value, list):
        for child in value:
            _validate_filter_fields(child, known_fields)


class ValidatorAgent:
    def __init__(self, capabilities=None):
        self.capabilities = capabilities or get_mongo_capabilities()

    @traceable(name="validator.validate")
    def validate(self, query: dict[str, Any], schema_spec: dict[str, Any], profile_lookup=None) -> dict[str, Any]:
        if not isinstance(query, dict):
            raise QueryValidationError("Query must be a JSON object")
        normalized = deepcopy(query)
        operation = str(normalized.get("operation", "")).strip()
        if operation not in ALLOWED_OPERATIONS:
            raise QueryValidationError(f"Unsupported operation: {operation}")

        collection = str(normalized.get("collection", "")).strip()
        internal = {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION,
                    SCHEMA_CACHE_COLLECTION, DATABASE_KNOWLEDGE_COLLECTION, QUERY_PERFORMANCE_COLLECTION, QUERY_TEMPLATES_COLLECTION}
        if collection in internal or collection.split(".")[-1].startswith("_system_"):
            raise QueryValidationError("Internal learning collections are not queryable")
        known_collections = _known_collections(schema_spec)
        if not collection or collection not in known_collections:
            raise QueryValidationError(f"Unknown or missing collection: {collection}")

        _validate_no_banned_keys(normalized)
        for key, child in _walk_keys(normalized):
            if key in {"$lookup", "$unionWith"}:
                target = child.get("from") if key == "$lookup" and isinstance(child, dict) else (
                    child.get("coll") if isinstance(child, dict) else child
                )
                if isinstance(target, str) and (target in internal or target.split(".")[-1].startswith("_system_")):
                    raise QueryValidationError("Internal learning collections are not queryable")
        known_fields = _known_fields(schema_spec, collection)
        from services.query.profile_config import RuntimeProfileConfig
        from services.query.profile_facts import safe_profile
        from services.query.profile_validation import validate_profile_query
        profile_active = (RuntimeProfileConfig.from_env().validation_mode != "off"
                          and safe_profile(collection, profile_lookup) is not None)
        # The profile layer owns paths when available, including partial-profile
        # warning policy. Preserve the legacy schema checks without a profile.
        if profile_active:
            known_fields = set()

        if operation == "find":
            normalized.setdefault("filter", {})
            normalized.setdefault("projection", None)
            normalized.setdefault("sort", [])
            limit = normalized.get("limit")
            normalized["limit"] = max(1, int(limit)) if limit is not None else None
            normalized["skip"] = max(0, int(normalized.get("skip") or 0))
            _validate_filter_fields(normalized["filter"], known_fields)

            sort = normalized.get("sort", [])
            if not isinstance(sort, list):
                raise QueryValidationError("find.sort must be a list of [field, direction] pairs")
            for item in sort:
                if not isinstance(item, list) or len(item) != 2 or int(item[1]) not in (-1, 1):
                    raise QueryValidationError("Invalid sort item")
                if known_fields and not _field_is_known(str(item[0]), known_fields):
                    raise QueryValidationError(f"Unknown sort field: {item[0]}")

        elif operation == "aggregate":
            pipeline = normalized.get("pipeline")
            stages = ALLOWED_STAGES | {"$unionWith"} if profile_active else ALLOWED_STAGES
            validate_pipeline(pipeline, stages, known_collections, self.capabilities)

        elif operation == "count_documents":
            normalized.setdefault("filter", {})
            _validate_filter_fields(normalized["filter"], known_fields)

        elif operation == "distinct":
            field = str(normalized.get("field", "")).strip()
            if not field:
                raise QueryValidationError("distinct.field is required")
            if known_fields and not _field_is_known(field, known_fields):
                raise QueryValidationError(f"Unknown distinct field: {field}")
            normalized.setdefault("filter", {})
            _validate_filter_fields(normalized["filter"], known_fields)

        validated = validate_profile_query(normalized, profile_lookup)
        from services.query.prompt_selection.pruning import log_fields_outside_prompt
        return log_fields_outside_prompt(validated, schema_spec)
