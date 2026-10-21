"""Repair intent against visible names and published observations before retrying."""
from copy import deepcopy
from difflib import SequenceMatcher
import logging
import math
import re

from services.query.values.matching import normalize_for_match
from services.query.profile_facts import safe_profile, is_date_field
from .config import mentions, requested_numbers, vocabulary_terms
from .models import QueryPlan
from .time_intent import recency_only

logger = logging.getLogger(__name__)


class PlanError(ValueError):
    def __init__(self, *errors):
        self.errors = list(errors)
        super().__init__("; ".join(errors))


def repair_log(action, count=1):
    from agent.telemetry import current_request_id
    logger.info("%s request_id=%s count=%d", action, current_request_id(), count)


def resolve_name(name, allowed, cutoff):
    if name in allowed:
        return name
    if not isinstance(name, str):
        raise PlanError("source_name_required")
    normalized = normalize_for_match(name)[0]
    exact = [candidate for candidate in allowed if normalize_for_match(candidate)[0] == normalized]
    candidates = exact or [candidate for candidate in allowed if
        SequenceMatcher(None, normalized, normalize_for_match(candidate)[0]).ratio() >= cutoff]
    if len(candidates) != 1:
        raise PlanError("source_name_ambiguous" if candidates else "source_name_unresolved")
    repair_log("plan_name_repaired")
    return candidates[0]


def coerce_literal(value, spec, op):
    if isinstance(value, (dict, list)):
        raise PlanError("filter_scalar_literal_required")
    if type(value) is float and not math.isfinite(value):
        raise PlanError("finite_literal_required")
    if op == "exists":
        if type(value) is not bool:
            raise PlanError("exists_boolean_required")
        return value
    types = set(spec.get("types", {})) - {"null"}
    if value is None:
        return value
    numeric = {"int", "long", "double", "decimal", "number"}
    if types and types <= numeric and isinstance(value, str):
        try:
            value = int(value) if types <= {"int", "long"} else float(value)
            if not math.isfinite(value):
                raise ValueError
            repair_log("plan_number_coerced")
        except (ValueError, OverflowError):
            raise PlanError("numeric_literal_required") from None
    if op == "contains" and (types != {"string"} or not isinstance(value, str)):
        raise PlanError("contains_string_required")
    if types and types <= numeric and (type(value) not in {int, float} or not math.isfinite(value)):
        raise PlanError("numeric_literal_required")
    if types <= {"int", "long"} and types and type(value) is float:
        raise PlanError("integer_literal_required")
    if types == {"string"} and not isinstance(value, str):
        raise PlanError("string_literal_required")
    if types in ({"bool"}, {"boolean"}) and type(value) is not bool:
        raise PlanError("boolean_literal_required")
    if op in {"gt", "gte", "lt", "lte"} and (type(value) is bool or types & {"object", "array", "bool", "boolean"}):
        raise PlanError("comparison_type_invalid")
    return value


def resolve_plan(plan, selected_schema, question, config, directives, preferences=(), profile_lookup=None):
    directives = [item for item in directives if item.get("status") == "approved"]
    raw = plan.model_dump()
    collections = selected_schema.get("collections", {})
    mappings = {item["collection"] for item in directives if item["type"] == "source_mapping"}
    if len(mappings) > 1:
        raise PlanError("source_directives_conflict")
    if mappings:
        raw["collection"] = next(iter(mappings))
    raw["collection"] = resolve_name(raw["collection"], collections, config.field_match_cutoff)
    collection = raw["collection"]
    profile = safe_profile(collection, profile_lookup)
    if not profile:
        raise PlanError("stored_profile_required")
    visible = {path: spec for path, spec in profile["fields"].items() if path in collections[collection].get("fields", {})}
    errors = []

    def field(value, location):
        try:
            return resolve_name(value, visible, config.field_match_cutoff)
        except PlanError as exc:
            errors.append(location + ": " + str(exc))
            return value

    raw["select"] = [field(path, f"select[{index}]") for index, path in enumerate(raw["select"])] if raw["select"] is not None else None
    raw["group_by"] = [field(path, f"group_by[{index}]") for index, path in enumerate(raw["group_by"])]
    for key in ("distinct_field", "time_field"):
        if raw[key] is not None:
            raw[key] = field(raw[key], key)
    if raw["time_field"] in visible and not is_date_field(visible[raw["time_field"]]):
        errors.append("time_field: date_field_required")
    local_directives = [item for item in directives if item["collection"] == collection]
    latest = [item for item in local_directives if item["type"] == "latest_per_key"]
    if len({(item["key_field"], item["order_field"]) for item in latest}) > 1:
        errors.append("latest_directives_conflict")
    # Losing eligibility for the directive also prevents an unapproved model task.
    if raw["task"] == "latest_per_key" and not latest:
        errors.append("latest_per_key_requires_applicable_approved_directive")
    for item in local_directives:
        if item["type"] == "latest_per_key" and any(item[key] not in visible for key in ("key_field", "order_field")):
            errors.append("directive_fields_outside_visible_schema")
    vocabulary = config.vocabulary()
    prefs = [item for item in preferences if item.get("collection") == collection]
    limits = [item["limit"] for item in prefs if item.get("type") == "result_limit_preference" and type(item.get("limit")) is int]
    if raw["limit"] is not None and raw["limit"] not in requested_numbers(question, vocabulary) | set(limits):
        raw["limit"] = None
        repair_log("plan_limit_removed")
    if raw["order"]:
        pref_order = any(item.get("field") == raw["order"]["field"] and
                         item.get("direction") in {raw["order"]["dir"], "asc" if raw["order"]["dir"] == 1 else "desc"}
                         for item in prefs if item.get("type") == "sorting_preference")
        if not mentions(question, vocabulary_terms(vocabulary, "ordering")) and not pref_order:
            raw["order"] = None
            repair_log("plan_order_removed")
        else:
            raw["order"]["field"] = field(raw["order"]["field"], "order.field")
    # Reuse the structured preference applier on a source-shaped query so it
    # also works when this plan later compiles to an aggregation.
    from services.preferences.application import apply_preferences_to_query
    preference_query = {"operation": "find", "collection": collection, "limit": raw["limit"],
                        "sort": [[raw["order"]["field"], raw["order"]["dir"]]] if raw["order"] else []}
    apply_preferences_to_query(preference_query, prefs, question=question)
    raw["limit"] = preference_query.get("limit")
    if preference_query.get("sort"):
        path, direction = preference_query["sort"][0]
        raw["order"] = {"field": field(path, "preference.order"), "dir": direction}
    if raw["join"]:
        raw["join"] = resolve_name(raw["join"], collections, config.field_match_cutoff)
    for directive in local_directives:
        if directive["type"] == "default_filter":
            path = directive["field"]
            if path not in visible:
                errors.append("default_filter_outside_visible_schema")
            elif not any(item["field"] == path for item in raw["filters"]):
                values = directive["values"]
                raw["filters"].append({"field": path, "op": directive["op"],
                                       "value": deepcopy(values if directive["op"] in {"in", "nin"} else values[0])})
    filters = []
    model_date_filter = False
    for index, item in enumerate(raw["filters"]):
        path = field(item["field"], f"filters[{index}].field")
        if path not in visible:
            continue
        item["field"] = path
        spec = visible[path]
        if is_date_field(spec) and item["op"] != "exists":
            model_date_filter = True
            raw["uses_time"] = True
            if raw["time_field"] is None:
                raw["time_field"] = path
            repair_log("plan_model_date_ignored")
            continue
        values = item["value"] if item["op"] in {"in", "nin"} else [item["value"]]
        if not isinstance(values, list):
            errors.append("filters.value: membership_array_required")
            continue
        try:
            if spec.get("kind") == "identifier" and item["op"] != "exists":
                pattern = spec.get("pattern")
                if any(str(value) not in question and not (pattern and re.fullmatch(pattern, str(value))) for value in values):
                    repair_log("plan_identifier_literal_removed")
                    continue
            values = [coerce_literal(value, spec, item["op"]) for value in values]
        except (PlanError, re.error) as exc:
            errors.append("filters.value: " + (str(exc) if isinstance(exc, PlanError) else "identifier_pattern_invalid"))
            continue
        item["value"] = values if item["op"] in {"in", "nin"} else values[0]
        filters.append(item)
    raw["filters"] = filters
    if raw["uses_time"] and not model_date_filter and recency_only(raw, profile, question, vocabulary):
        raw["uses_time"] = False
        raw["time_field"] = None
        repair_log("plan_recency_time_flag_cleared")
    if raw["task"] in {"group_count", "top_n"} and not raw["group_by"]:
        errors.append("group_by_required")
    if raw["task"] == "distinct" and not raw["distinct_field"]:
        errors.append("distinct_field_required")
    if errors:
        raise PlanError(*dict.fromkeys(errors))
    return QueryPlan.model_validate(raw), profile, local_directives
