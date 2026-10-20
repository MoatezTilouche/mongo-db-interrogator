"""Copies for prompt serialization and request-local visibility diagnostics."""
from contextvars import ContextVar
from copy import deepcopy
import logging
from database.schema_inspector import compact_schema

logger = logging.getLogger(__name__)
_view = ContextVar("generator_prompt_view", default=None)


def prune_schema_for_prompt(selected_schema, relevance, config, *, required_fields=()):
    result = deepcopy(selected_schema)
    if not config.enabled:
        return result
    for name, collection in result.get("collections", {}).items():
        fields = collection.get("fields", {})
        if len(fields) <= config.max_fields or name not in getattr(relevance, "profiled", set(relevance)):
            continue
        signal_count = sum(not all(reason.startswith("default_") for reason in row["reasons"]) for row in relevance[name])
        if signal_count < config.min_signals:
            logger.info("schema_prune_low_signal collection=%s signals=%d minimum=%d fill=structural_defaults",
                        name, signal_count, config.min_signals)
        # Preserve catalog-backed rule, preference and join dependencies even
        # when they exceed the optional field budget.
        kept = [path for path in fields if (name, path) in required_fields]
        kept.extend(row["field"] for row in relevance[name] if row["field"] in fields and row["field"] not in kept
                    and set(row["reasons"]) & {"relationship", "preference", "field_leaf", "field_token", "value", "time"})
        eligible = {row["field"] for row in relevance[name]}
        for row in relevance[name]:
            path = row["field"]
            if path not in fields:
                continue
            parents = [".".join(path.split(".")[:index]) for index in range(1, len(path.split(".")))] if config.include_parents else []
            additions = [candidate for candidate in [*parents, path] if candidate in fields and candidate in eligible and candidate not in kept]
            if len(additions) > config.max_fields:
                additions = [path]
            if len(kept) + len(additions) <= config.max_fields:
                kept.extend(additions)
        if not kept:
            logger.info("schema_prune_no_safe_fields collection=%s omitted_count=%d", name, len(fields))
        collection["fields"] = {path: deepcopy(fields[path]) for path in kept}
        omitted = len(fields) - len(kept)
        if omitted:
            collection["_prompt_note"] = f"(+{omitted} more fields not shown; use listed fields only, never guess omitted names)"
    return result


def compact_prompt_schema(schema):
    result = compact_schema(schema)
    for name, entry in schema.get("collections", {}).items():
        if entry.get("_prompt_note"):
            result[name]["omitted_fields"] = entry["_prompt_note"]
    return result


def remember_prompt_view(full_schema, prompt_schema):
    _view.set((full_schema, {name: set(entry.get("fields", {})) for name, entry in prompt_schema.get("collections", {}).items()}))


def query_field_references(query, full_schema):
    """Known source fields and owners, including nested join/union inputs."""
    seen = {}
    def owner(collection):
        if isinstance(collection, str) and collection in full_schema.get("collections", {}):
            seen.setdefault(collection, set())
    def path(collection, field):
        owner(collection)
        if isinstance(collection, str) and isinstance(field, str) and field in full_schema.get("collections", {}).get(collection, {}).get("fields", {}):
            seen[collection].add(field)
    def visit(value, collection):
        if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            path(collection, value[1:])
        elif isinstance(value, dict):
            for key, child in value.items():
                if key == "$lookup" and isinstance(child, dict):
                    path(collection, child.get("localField"))
                    path(child.get("from"), child.get("foreignField"))
                    visit(child.get("pipeline", []), child.get("from"))
                    visit(child.get("let", {}), collection)
                elif key == "$unionWith" and isinstance(child, dict):
                    owner(child.get("coll"))
                    visit(child.get("pipeline", []), child.get("coll"))
                elif key == "$unionWith" and isinstance(child, str):
                    owner(child)
                elif key not in {"$literal", "$const"}:
                    path(collection, key)
                    visit(child, collection)
        elif isinstance(value, list):
            for child in value:
                visit(child, collection)
    collection = query.get("collection")
    owner(collection)
    for key in ("filter", "projection", "pipeline"):
        visit(query.get(key, {}), collection)
    for item in query.get("sort") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            path(collection, item[0])
    path(collection, query.get("field"))
    return seen


def log_fields_outside_prompt(query, full_schema):
    view = _view.get()
    if view is None or view[0] is not full_schema:
        return query
    for collection, fields in query_field_references(query, full_schema).items():
        for field in sorted(fields - view[1].get(collection, set())):
            logger.info("field_outside_pruned_view collection=%s field=%s", collection, field)
    return query


def validate_prompt_collection_scope(query, full_schema):
    """Generation cannot use a source that was removed from its prompt view."""
    from services.aggregation_validation import QueryValidationError
    view = _view.get()
    if view is None or view[0] is not full_schema:
        return
    allowed = set(view[1])
    def check(name):
        if name not in allowed:
            raise QueryValidationError("Query uses a collection outside the selected prompt schema",
                                       code="query_collection_scope")
    check(query.get("collection"))
    def stages(pipeline):
        for stage in pipeline or []:
            if "$lookup" in stage:
                check(stage["$lookup"]["from"])
                stages(stage["$lookup"].get("pipeline"))
            if "$unionWith" in stage:
                union = stage["$unionWith"]
                check(union if isinstance(union, str) else union["coll"])
                if isinstance(union, dict):
                    stages(union.get("pipeline"))
            for child in stage.get("$facet", {}).values():
                stages(child)
    stages(query.get("pipeline"))
