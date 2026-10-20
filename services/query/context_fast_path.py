"""Conservative, fully consumed explicit syntax; never generate or execute a query."""
from copy import deepcopy
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

from agent.output_schemas import QueryContext
from services.query.profile_facts import safe_profile, is_date_field, is_safe_path

_FIELD = r"\w[\w.]*"
_VALUE = r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[+-]?\d+(?:\.\d+)?|true|false|null|[\w:@./-]+'''


def mode_from_env():
    mode = os.getenv("QUERY_CONTEXT_FAST_PATH_MODE", "log").strip().lower()
    if mode not in {"off", "log", "on"}:
        raise ValueError("QUERY_CONTEXT_FAST_PATH_MODE must be off, log or on")
    return mode


def _alternatives(values):
    return "(?:" + "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True)) + ")"


@lru_cache(maxsize=1)
def _syntax():
    vocabulary = json.loads((Path(__file__).resolve().parents[2] / "config/context_fast_path.json").read_text(encoding="utf-8"))
    actions = [word for words in vocabulary["actions"].values() for word in words]
    fields = _FIELD + r"(?:\s*,\s*" + _FIELD + r")*"
    pattern = (
        r"\s*(?P<action>" + _alternatives(actions) + r")\s+"
        r"(?:(?P<leading_limit>\d+)\s+)?"
        r"(?:(?P<fields>" + _alternatives(vocabulary["all"]) + "|" + fields + r")\s+)?"
        + _alternatives(vocabulary["from"]) + r"\s+(?P<collection>[\w.-]+)"
        r"(?:\s+" + _alternatives(vocabulary["where"]) + r"\s+(?P<predicate>.+?))?"
        r"(?:\s+" + _alternatives(vocabulary["sort"]) + r"\s+(?P<sort>" + _FIELD +
        r")(?:\s+(?P<direction>" + _alternatives(vocabulary["directions"]) + r"))?)?"
        r"(?:\s+" + _alternatives(vocabulary["limit"]) + r"\s+(?P<limit>\d+))?\s*"
    )
    predicate = re.compile(r"\s*(?P<field>" + _FIELD + r")\s*(?P<op>>=|<=|!=|=|>|<)\s*(?P<value>" + _VALUE + r")(?=\s|$)", re.I)
    connector = re.compile(r"\s+" + _alternatives(vocabulary["and"]) + r"\s+", re.I)
    return vocabulary, re.compile(pattern, re.I), predicate, connector


def _literal(raw):
    if raw.startswith('"'):
        return json.loads(raw)
    if raw.startswith("'"):
        # Only plain single-quoted strings are accepted; no Python evaluation.
        if "\\" in raw:
            raise ValueError
        return raw[1:-1]
    if raw.lower() in {"true", "false", "null"}:
        return json.loads(raw.lower())
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
        value = float(raw) if "." in raw else int(raw)
        if not math.isfinite(value):
            raise ValueError
        return value
    return raw


def _compatible(value, types):
    if isinstance(types, dict):
        types = types.get("types", types.get("type", []))
    kinds = set([types] if isinstance(types, str) else types or []) - {"null"}
    if value is None or not kinds or kinds & {"array", "object", "date", "datetime", "timestamp"}:
        return False
    if type(value) is bool:
        return kinds <= {"bool", "boolean"}
    if type(value) in {int, float}:
        return kinds <= {"int", "long", "double", "float", "decimal", "number", "integer"}
    return kinds <= {"string"}


def assess_context(question, catalog, knowledge=(), preferences=(), *, full_catalog=None, profile_lookup=None, mode=None):
    """Return eligibility and a context hint, leaving all query work downstream.

    Unrecognized text, implicit collection aliases, dates, joins and business
    rules always use the existing context agent. Log mode never skips that agent.
    """
    mode = mode or mode_from_env()
    result = {"mode": mode, "eligible": False, "reason": "disabled" if mode == "off" else "unsupported_syntax"}
    if mode == "off":
        return result
    try:
        vocab, syntax, predicate, connector = _syntax()
        matched = syntax.fullmatch(question)
    except (ValueError, OSError, KeyError, TypeError, re.error):
        return {**result, "reason": "configuration_unavailable"}
    if not matched:
        return result
    parts = matched.groupdict()
    name = parts["collection"]
    available = (full_catalog or catalog).get("collections", {})
    if name not in available or name not in catalog.get("collections", {}):
        return {**result, "reason": "source_not_in_selected_catalog"}
    if knowledge:
        return {**result, "reason": "business_rules_require_context"}
    if any(item.get("collection") != name or item.get("source_collection", name) != name for item in preferences):
        return {**result, "reason": "preferences_require_context"}
    operation = next(key for key, words in vocab["actions"].items() if parts["action"].casefold() in {word.casefold() for word in words})
    fields = available[name].get("fields", {})
    if not fields:
        return {**result, "reason": "empty_schema"}
    projection = parts["fields"]
    all_fields = projection and projection.casefold() in {word.casefold() for word in vocab["all"]}
    if all_fields and projection in fields and projection != "*":
        return {**result, "reason": "ambiguous_projection"}
    if operation == "count" and (projection and not all_fields or any(parts[key] for key in ("sort", "leading_limit", "limit"))):
        return {**result, "reason": "count_with_extra_shape"}
    if parts["leading_limit"] and parts["limit"] or any(parts[key] and int(parts[key]) < 1 for key in ("leading_limit", "limit")):
        return {**result, "reason": "ambiguous_limit"}
    requested = set()
    if projection and not all_fields:
        requested.update(field.strip() for field in projection.split(","))
    if parts["sort"]:
        requested.add(parts["sort"])
    profile = safe_profile(name, profile_lookup)
    if parts["sort"] and (is_date_field((profile or {}).get("fields", {}).get(parts["sort"], {}))
                          or "." in parts["sort"]):
        return {**result, "reason": "sort_requires_context"}
    expression = parts["predicate"] or ""
    cursor = 0
    seen = set()
    while cursor < len(expression):
        term = predicate.match(expression, cursor)
        if not term:
            return {**result, "reason": "unsupported_filter"}
        path = term["field"]
        if path in seen or path not in fields:
            return {**result, "reason": "ambiguous_or_unknown_filter_field"}
        seen.add(path)
        requested.add(path)
        spec = (profile or {}).get("fields", {}).get(path, {})
        if is_date_field(spec) or not is_safe_path(profile, path) or "." in path:
            return {**result, "reason": "filter_requires_context"}
        try:
            value = _literal(term["value"])
        except (ValueError, OverflowError):
            return {**result, "reason": "unsupported_filter_literal"}
        if not _compatible(value, fields[path]):
            return {**result, "reason": "filter_type_requires_context"}
        cursor = term.end()
        if cursor == len(expression):
            break
        join = connector.match(expression, cursor)
        if not join or join.end() == len(expression):
            return {**result, "reason": "unsupported_filter"}
        cursor = join.end()
    if not requested <= set(fields) or any(not is_safe_path(profile, path) for path in requested):
        return {**result, "reason": "unknown_or_excluded_field"}
    context = QueryContext(complexity="simple", operation=operation, collections=[name], fields={name: list(fields)})
    return {**result, "eligible": True, "reason": "explicit_single_collection", "result": {
        "query_context": context.model_dump(), "schema_spec": {"collections": {name: {"fields": deepcopy(fields)}}},
        "user_preferences": deepcopy(list(preferences)),
    }}
