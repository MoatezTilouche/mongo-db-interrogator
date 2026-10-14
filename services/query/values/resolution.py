"""Resolve categorical filter spellings from stored data, without domain terms."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.filter_value_resolution"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "filter_value_resolution", _sys.modules[__spec__.name])

from copy import deepcopy
import difflib
import json
import logging
import re
import unicodedata

from agent.config import (
    KNOWLEDGE_VALUE_RESOLUTION_ENABLED, KNOWLEDGE_CASE_VARIANTS_POLICY,
    KNOWLEDGE_CASE_SENSITIVE_FIELDS, KNOWLEDGE_VALUE_SUGGESTION_CUTOFF,
)
from database.record_safety import safe_record
from services.value_catalog import get_value_catalog, get_value_catalog_status

logger = logging.getLogger("services.filter_value_resolution")
logger.setLevel(logging.INFO)


def _normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in unicodedata.normalize("NFKD", value)
                   if not unicodedata.combining(char))


def _distinguished(variants, collection, field, rules):
    mentioned = set()
    for rule in rules or []:
        if (rule.get("active") is not True or rule.get("status") != "active"
                or (rule.get("knowledge_type") or rule.get("type")) not in {"coded_value", "field_definition"}
                or field not in rule.get("fields", [])
                or rule.get("collections") and collection not in rule["collections"]):
            continue
        # Folding case/accents here would turn one mention into every variant.
        # Canonicalize compatibility characters while preserving distinctions.
        text = unicodedata.normalize("NFKC", rule.get("text") or "")
        for value in variants:
            spelling = unicodedata.normalize("NFKC", value)
            if spelling and re.search(r"(?<!\w)" + re.escape(spelling) + r"(?!\w)", text):
                mentioned.add(spelling)
    return len(mentioned) >= 2


def _log(collection, field, original, resolved, reason, closest=None):
    contents = {"original_value": original, "resolved_values": resolved}
    if closest is not None:
        contents["closest_stored_value"] = closest
    contents = safe_record({field: contents})[field]
    if not isinstance(contents, dict):
        contents = {"original_value": contents, "resolved_values": contents}
    payload = {"collection": safe_record(collection), "field": safe_record(field), **contents, "reason": reason}
    logger.info("%s data=%s", "value_not_in_catalog" if reason == "not_in_catalog" else "query_value_resolution",
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")))


def resolve_filter_value(collection, field, user_value, rules):
    if not KNOWLEDGE_VALUE_RESOLUTION_ENABLED or not isinstance(user_value, str):
        return [user_value], "unchanged"
    if collection + "." + field in KNOWLEDGE_CASE_SENSITIVE_FIELDS:
        return [user_value], "case_sensitive"
    values = get_value_catalog(collection, field)
    if values is None:
        reason = get_value_catalog_status(collection, field)
        if reason == "catalog_unavailable":
            _log(collection, field, user_value, [user_value], reason)
        return [user_value], reason
    variants = sorted(value for value in values if _normalize(value) == _normalize(user_value))
    if not variants:
        normalized = {_normalize(value): value for value in reversed(sorted(values))}
        matches = difflib.get_close_matches(_normalize(user_value), sorted(normalized), n=1,
                                          cutoff=KNOWLEDGE_VALUE_SUGGESTION_CUTOFF)
        _log(collection, field, user_value, [user_value], "not_in_catalog", normalized[matches[0]] if matches else None)
        return [user_value], "not_in_catalog"
    if _distinguished(variants, collection, field, rules):
        _log(collection, field, user_value, [user_value], "rule_guard")
        return [user_value], "rule_guard"
    if KNOWLEDGE_CASE_VARIANTS_POLICY == "exact" and user_value in variants:
        resolved, reason = [user_value], "exact"
    elif len(variants) > 1:
        resolved, reason = variants, "merged_variants"
    elif variants[0] == user_value:
        resolved, reason = variants, "exact"
    elif unicodedata.normalize("NFKC", variants[0]).casefold() == unicodedata.normalize("NFKC", user_value).casefold():
        resolved, reason = variants, "case_variant"
    else:
        resolved, reason = variants, "accent_variant"
    if resolved != [user_value]:
        _log(collection, field, user_value, resolved, reason)
    return resolved, reason


def _unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _contains_regex(value):
    return isinstance(value, dict) and ("$regex" in value or any(_contains_regex(child) for child in value.values()))


def apply_value_resolution(query, selected_schema, rules):
    if not KNOWLEDGE_VALUE_RESOLUTION_ENABLED:
        return query
    result = deepcopy(query)
    catalog = selected_schema.get("collections", selected_schema)

    def origin(field, collection, aliases, blocked):
        if any(field == path or field.startswith(path + ".") for path in blocked):
            return None
        head, separator, tail = field.partition(".")
        if separator and head in aliases:
            return aliases[head], tail
        return (collection, field) if collection in catalog else None

    def condition(value, collection, field):
        if isinstance(value, str):
            values, _ = resolve_filter_value(collection, field, value, rules)
            return (values[0] if len(values) == 1 else {"$in": values}), []
        if not isinstance(value, dict) or _contains_regex(value) or any(key in value for key in ("$oid", "$date", "$param")):
            return value, []
        changed = deepcopy(value)
        extra = []
        for operator, child in list(value.items()):
            if operator in {"$eq", "$ne"} and isinstance(child, str):
                values, _ = resolve_filter_value(collection, field, child, rules)
                if len(values) == 1:
                    changed[operator] = values[0]
                else:
                    del changed[operator]
                    target = "$in" if operator == "$eq" else "$nin"
                    if target in value:
                        extra.append({field: {target: values}})
                    else:
                        changed[target] = values
            elif operator in {"$in", "$nin", "$all"} and isinstance(child, list):
                expanded, groups = [], []
                for item in child:
                    values = resolve_filter_value(collection, field, item, rules)[0] if isinstance(item, str) else [item]
                    expanded.extend(values)
                    if len(values) > 1:
                        groups.append({field: {"$in": values}})
                    elif operator == "$all":
                        groups.append({field: {"$all": values}})
                if operator == "$all" and any("$in" in group[field] for group in groups):
                    del changed[operator]
                    extra.extend(groups)
                else:
                    changed[operator] = _unique(expanded)
            elif operator == "$not" and isinstance(child, dict):
                nested, constraints = condition(child, collection, field)
                # Keep complex negations intact when expanding an all-of clause.
                if not constraints:
                    changed[operator] = nested
            elif operator == "$elemMatch" and isinstance(child, dict) and any(key.startswith("$") for key in child):
                nested, constraints = condition(child, collection, field)
                if not constraints:
                    changed[operator] = nested
        return changed, extra

    def predicate(node, collection, aliases, blocked, prefix=""):
        if not isinstance(node, dict):
            return node
        output, conjunctions = {}, []
        for field, value in node.items():
            if field in {"$and", "$or", "$nor"} and isinstance(value, list):
                output[field] = [predicate(item, collection, aliases, blocked, prefix) for item in value]
            elif field.startswith("$"):
                output[field] = value
            else:
                resolved = origin(prefix + field, collection, aliases, blocked)
                if resolved is None:
                    output[field] = value
                    continue
                owner, path = resolved
                if isinstance(value, dict) and "$elemMatch" in value and isinstance(value["$elemMatch"], dict) and not any(key.startswith("$") for key in value["$elemMatch"]):
                    updated = deepcopy(value)
                    updated["$elemMatch"] = predicate(value["$elemMatch"], collection, aliases, blocked, prefix + field + ".")
                    output[field] = updated
                    continue
                updated, extra = condition(value, owner, path)
                if updated or not isinstance(updated, dict):
                    output[field] = updated
                # Conditions use the query's alias path, not the source path.
                conjunctions.extend({field: constraint[path]} for constraint in extra)
        if conjunctions:
            if output:
                conjunctions.insert(0, output)
            return {"$and": conjunctions}
        return output

    def pipeline(stages, collection, aliases=None, blocked=None):
        if not isinstance(stages, list):
            return
        aliases, blocked = dict(aliases or {}), set(blocked or ())
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            if isinstance(stage.get("$match"), dict):
                stage["$match"] = predicate(stage["$match"], collection, aliases, blocked)
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict):
                pipeline(lookup.get("pipeline"), lookup.get("from"))
                if isinstance(lookup.get("as"), str) and isinstance(lookup.get("from"), str):
                    aliases[lookup["as"]] = lookup["from"]
            if isinstance(stage.get("$facet"), dict):
                for child in stage["$facet"].values():
                    pipeline(child, collection, aliases, blocked)
            for operator in ("$set", "$addFields", "$project"):
                if isinstance(stage.get(operator), dict):
                    blocked.update(field for field, value in stage[operator].items() if value not in (0, 1, True, False))
            if any(operator in stage for operator in ("$group", "$replaceRoot", "$replaceWith")):
                collection, aliases = None, {}

    if isinstance(result.get("filter"), dict):
        result["filter"] = predicate(result["filter"], result.get("collection"), {}, set())
    pipeline(result.get("pipeline"), result.get("collection"))
    return result


def parameterize_resolved_query(generated, query):
    """Rebuild filter parameters using the existing rules; preserve limit slots."""
    from services.query_parameter_binder import (
        parameterize_filter_values, reconcile_template, visit_filter_values,
    )
    if not KNOWLEDGE_VALUE_RESOLUTION_ENABLED:
        return parameterize_filter_values(reconcile_template(generated, query))

    def concrete(value, path):
        if isinstance(value, dict) and set(value) == {"$param"}:
            return deepcopy(generated["parameters"][value["$param"]])
        return value

    skeleton = visit_filter_values(generated["template"], concrete)
    base = {"template": skeleton, "parameters": {}}
    if "limit" in generated["parameters"]:
        base["parameters"]["limit"] = generated["parameters"]["limit"]
    if generated.get("semantic_intent"):
        base["semantic_intent"] = generated["semantic_intent"]
    return parameterize_filter_values(reconcile_template(base, query))


def active_resolution_rules():
    if not KNOWLEDGE_VALUE_RESOLUTION_ENABLED:
        return []
    try:
        from database.knowledge_repository import KnowledgeRepository
        return KnowledgeRepository().list_active()
    except Exception:
        return []
