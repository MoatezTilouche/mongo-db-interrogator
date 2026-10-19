"""Compile repaired intent into the existing concrete query dictionary."""
from datetime import datetime, timezone
import re

from services.query.values.resolution import apply_value_resolution
from services.query.profile_facts import is_date_field, safe_profile
from services.query.dates.normalization import DateNormalizer, representation
from services.query.dates.resolver import resolve_question_dates
from services.query.profile_config import RuntimeProfileConfig
from .resolution import PlanError, repair_log


def helper_name(fields, suffix):
    result = "__plan_" + suffix
    while any(path == result or path.startswith(result + ".") for path in fields):
        result += "_"
    return result


def compile_filter(plan, profile, selected_schema, rules, directives, question, now=None):
    conditions = []
    insensitive = {path for item in directives if item["type"] == "case_insensitive_fields" for path in item["fields"]}
    for item in plan.filters:
        field, op, value = item.field, item.op, item.value
        if op == "contains":
            condition = {"$regex": re.escape(value)}
            if field in insensitive:
                condition["$options"] = "i"
        elif field in insensitive and profile["fields"][field].get("kind") != "categorical" and op in {"eq", "ne", "in", "nin"}:
            values = value if op in {"in", "nin"} else [value]
            if not all(isinstance(child, str) for child in values):
                raise PlanError("case_insensitive_string_required")
            regex = {"$regex": "^(?:" + "|".join(re.escape(child) for child in values) + ")$", "$options": "i"}
            condition = {"$not": regex} if op in {"ne", "nin"} else regex
        else:
            condition = value if op == "eq" else {"$" + op: value}
        conditions.append({field: condition})
    predicate = conditions[0] if len(conditions) == 1 else {"$and": conditions} if conditions else {}
    # Resolve while ownership is explicit, before a group/replaceRoot can hide it.
    predicate = apply_value_resolution({"operation": "find", "collection": plan.collection, "filter": predicate},
                                      selected_schema, rules,
                                      profile_lookup=lambda name: profile if name == plan.collection else None)["filter"]
    runtime = RuntimeProfileConfig.from_env()
    constraints = resolve_question_dates(question, now or datetime.now(timezone.utc), runtime.zone(), runtime.day_first)
    if constraints:
        candidates = [path for path, spec in profile["fields"].items() if is_date_field(spec)
                      and path in selected_schema["collections"][plan.collection].get("fields", {})]
        field = plan.time_field
        if field is None:
            if len(candidates) != 1:
                raise PlanError("time_field_ambiguous")
            field = candidates[0]
        if field not in candidates:
            raise PlanError("time_field_invalid")
        normalizer = DateNormalizer(lambda name: profile if name == plan.collection else None, question, now)
        date_filter = {field: normalizer.question_condition(profile["fields"][field], field, False)}
        predicate = {"$and": [predicate, date_filter]} if predicate else date_filter
        repair_log("plan_question_time_resolved")
    elif plan.uses_time:
        raise PlanError("question_time_unresolved")
    return predicate


def latest_stages(profile, directive):
    fields = profile["fields"]
    order_field = directive["order_field"]
    spec = fields[order_field]
    stored, fmt = representation(spec)
    stages, helpers = [], []
    root = helper_name(fields, "root")
    if stored == "string":
        parsed = helper_name(fields, "date")
        stages.append({"$addFields": {parsed: {"$dateFromString": {
            "dateString": "$" + order_field, "format": fmt, "onError": None, "onNull": None}}}})
        order_field = parsed
        helpers.append(parsed)
    elif stored != "native":
        raise PlanError("latest_date_representation_unsupported")
    stages.extend([
        {"$sort": {order_field: -1, "_id": -1}},
        {"$group": {"_id": "$" + directive["key_field"], root: {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$" + root}},
    ])
    return stages, helpers


def join_stage(source, target, selected_schema, profile_lookup=None):
    """Endpoints must be visible and verified by both stored profiles."""
    profile = safe_profile(source, profile_lookup)
    foreign = safe_profile(target, profile_lookup)
    if not profile or not foreign or source == target:
        raise PlanError("join_profiles_required")
    relationships = [row for row in profile.get("relationships", [])
        if row.get("target_collection") == target
        and row.get("field") in profile["fields"]
        and row.get("target_field") in foreign["fields"]
        and row["field"] in selected_schema["collections"][source].get("fields", {})
        and row["target_field"] in selected_schema["collections"][target].get("fields", {})]
    endpoints = {(row["field"], row["target_field"]) for row in relationships}
    if len(endpoints) != 1:
        raise PlanError("join_relationship_unverified_or_ambiguous")
    local, remote = next(iter(endpoints))
    alias = helper_name(profile["fields"], "join")
    return {"$lookup": {"from": target, "localField": local, "foreignField": remote, "as": alias}}


def compile_plan(plan, profile, selected_schema, question, rules=(), directives=(), profile_lookup=None, now=None):
    from .directives import validate_directive
    lookup = profile_lookup or (lambda name: profile if name == plan.collection else safe_profile(name))
    directives = [validate_directive(item, lookup) for item in directives
                  if item.get("status") == "approved" and item.get("collection") == plan.collection]
    predicate = compile_filter(plan, profile, selected_schema, rules, directives, question, now)
    query = {"collection": plan.collection}
    latest = next((item for item in directives if item["type"] == "latest_per_key"), None)
    is_aggregate = latest is not None or plan.task in {"group_count", "top_n"} or plan.join is not None
    projection = {field: 1 for field in plan.select} if plan.select else None
    order = [[plan.order.field, plan.order.dir]] if plan.order else []
    if not is_aggregate:
        if plan.task == "list":
            query.update(operation="find", filter=predicate, projection=projection, sort=order, limit=plan.limit, skip=0)
        elif plan.task == "count":
            query.update(operation="count_documents", filter=predicate)
        elif plan.task == "distinct":
            query.update(operation="distinct", filter=predicate, field=plan.distinct_field)
        else:
            raise PlanError("task_unsupported_without_directive")
    else:
        stages, helpers = latest_stages(profile, latest) if latest else ([], [])
        if predicate:
            stages.append({"$match": predicate})
        if plan.join:
            stages.append(join_stage(plan.collection, plan.join, selected_schema, profile_lookup))
        if plan.task in {"group_count", "top_n"}:
            metric = helper_name(profile["fields"], "metric")
            key = {field.replace(".", "_"): "$" + field for field in plan.group_by}
            if len(key) != len(plan.group_by):
                raise PlanError("group_output_names_ambiguous")
            stages.append({"$group": {"_id": key, metric: {"$sum": 1}}})
            stages.append({"$sort": {metric: -1, "_id": 1}})
            helpers = []
        elif plan.task == "count":
            stages.append({"$count": helper_name(profile["fields"], "metric")})
            helpers = []
        elif plan.task == "distinct":
            stages.append({"$group": {"_id": "$" + plan.distinct_field}})
            helpers = []
        elif plan.task in {"list", "latest_per_key"}:
            if order:
                stages.append({"$sort": dict(order)})
        else:
            raise PlanError("task_unsupported")
        if plan.limit is not None and plan.task != "count":
            stages.append({"$limit": plan.limit})
        if helpers:
            stages.append({"$unset": helpers})
        if projection and plan.task in {"list", "latest_per_key"}:
            stages.append({"$project": projection})
        query.update(operation="aggregate", pipeline=stages)
    # All tokens describe structure; filter values never enter cached intent.
    concepts = sorted(set([*plan.group_by, *(plan.select or []),
                          *[item.field for item in plan.filters],
                          *([plan.distinct_field] if plan.distinct_field else []),
                          *([plan.time_field] if plan.time_field else [])]))
    intent = " ".join([plan.task, plan.collection, *concepts])[:600]
    return query, intent
