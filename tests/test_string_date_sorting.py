"""Chronological sorting rewrites preserve query meaning without live services."""
from copy import deepcopy
from datetime import datetime, timezone
from functools import cmp_to_key
import json

import pytest

from agents.validator_agent import ValidatorAgent
from services.aggregation_validation import QueryValidationError
from services.query.dates.normalization import apply_date_normalization

C, D, T, S = "synthetic_sort_rows", "synthetic_sort_other", "momentX", "stateX"


@pytest.fixture
def facts(monkeypatch):
    for key, value in {"PLAN_MODE": "off", "DATE_NORMALIZATION_ENABLED": "true",
                       "DATE_ENFORCE_QUESTION_RANGE": "true", "DATE_TIMEZONE": "UTC",
                       "DATE_PLAUSIBILITY_MODE": "off", "PROFILE_VALIDATION_MODE": "strict"}.items():
        monkeypatch.setenv(key, value)
    fields = {T: {"kind": "date", "types": {"string": 1},
                  "date_format": "%d/%m/%Y %H:%M:%S", "sortable_as_string": False},
              S: {"kind": "text", "types": {"string": 1}}}
    return {name: {"collection": name, "fields": deepcopy(fields)} for name in (C, D)}


def normalize(facts, query):
    return apply_date_normalization(query, facts.get, "", datetime(2026, 10, 21, tzinfo=timezone.utc))


def schema(facts):
    return {"collections": {name: {"fields": {field: list(spec["types"]) for field, spec in row["fields"].items()}}
                            for name, row in facts.items()}}


def find(**changes):
    return {"operation": "find", "collection": C, "filter": {S: "Value_A"},
            "projection": None, "sort": [[T, -1], ["_id", 1]], "skip": 0, "limit": None, **changes}


def path_value(row, path):
    for part in path.split("."):
        row = row.get(part) if isinstance(row, dict) else None
    return row


def evaluate(value, row, variables=None):
    variables = variables or {}
    if value == "$$ROOT":
        return deepcopy(row)
    if isinstance(value, str) and value.startswith("$$"):
        return variables.get(value[2:])
    if isinstance(value, str) and value.startswith("$"):
        return path_value(row, value[1:])
    if not isinstance(value, dict):
        return value
    if "$dateFromString" in value:
        options = value["$dateFromString"]
        try:
            return datetime.strptime(evaluate(options["dateString"], row, variables), options["format"]).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return options["onError"]
    if "$isArray" in value:
        return isinstance(evaluate(value["$isArray"], row, variables), list)
    if "$cond" in value:
        predicate, yes, no = value["$cond"]
        return evaluate(yes if evaluate(predicate, row, variables) else no, row, variables)
    if "$map" in value:
        options = value["$map"]
        return [evaluate(options["in"], row, {**variables, options["as"]: item})
                for item in evaluate(options["input"], row, variables)]
    return {key: evaluate(child, row, variables) for key, child in value.items()}


def execute(stages, rows):
    """Evaluate only the emitted stages, including complete-document restore."""
    rows = deepcopy(rows)
    for stage in stages:
        operator, value = next(iter(stage.items()))
        if operator == "$match":
            rows = [row for row in rows if all(row.get(key) == item for key, item in value.items())]
        elif operator == "$replaceRoot":
            rows = [evaluate(value["newRoot"], row) for row in rows]
        elif operator == "$sort":
            def compare(left, right):
                for path, direction in value.items():
                    a, b = path_value(left, path), path_value(right, path)
                    if isinstance(a, list):
                        a = (min if direction == 1 else max)(a, default=None)
                    if isinstance(b, list):
                        b = (min if direction == 1 else max)(b, default=None)
                    if a == b:
                        continue
                    return direction * (-1 if a is None else 1 if b is None else (a > b) - (a < b))
                return 0
            rows.sort(key=cmp_to_key(compare))
        elif operator == "$skip":
            rows = rows[value:]
        elif operator == "$limit":
            rows = rows[:value]
        elif operator == "$project":
            if any(enabled for field, enabled in value.items() if field != "_id"):
                rows = [{key: item for key, item in row.items() if value.get(key) or key == "_id" and value.get("_id", 1)}
                        for row in rows]
            else:
                rows = [{key: item for key, item in row.items() if value.get(key, 1)} for row in rows]
        else:
            raise AssertionError(operator)
    return rows


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_synthetic_chronology_predicate_skip_limit_projection_and_collisions(facts, aggregate, direction):
    rows = [{"_id": index, T: text + " 00:00:00", S: "Value_A",
             "__date_sort_document": {"original": index}, "__date_sort_0": "original_value"}
            for index, text in enumerate(["31/08/2026", "01/09/2026", "15/09/2026", "30/09/2026", "01/10/2026"])]
    rows.append({"_id": 9, T: "31/12/2026 00:00:00", S: "Value_B"})
    candidate = find(sort=[[T, direction], ["_id", 1]], skip=1, limit=3,
                     projection={"_id": 0, T: 1, "__date_sort_document": 1, "__date_sort_0": 1})
    facts[C]["fields"].update({"__date_sort_document": {"types": {"object": 1}},
                                "__date_sort_0": {"types": {"string": 1}}})
    if aggregate:
        candidate = {"operation": "aggregate", "collection": C, "pipeline": [
            {"$match": candidate["filter"]}, {"$sort": dict(candidate["sort"])},
            {"$skip": 1}, {"$limit": 3}, {"$project": candidate["projection"]}]}
    original = deepcopy(candidate)
    fixed = normalize(facts, candidate)
    assert candidate == original
    assert fixed["operation"] == "aggregate"
    ValidatorAgent().validate(fixed, schema(facts), facts.get)
    expected = rows[1:4] if direction == 1 else list(reversed(rows[:5]))[1:4]
    expected = [{key: value for key, value in row.items() if key not in {"_id", S}} for row in expected]
    assert execute(fixed["pipeline"], rows) == expected
    assert normalize(facts, fixed) == fixed


@pytest.mark.parametrize("direction", [-1, 1])
def test_invalid_null_missing_records_are_not_dropped_or_mutated(facts, direction):
    rows = [{"_id": 0, S: "Value_A"}, {"_id": 1, T: None, S: "Value_A"},
            {"_id": 2, T: "bad", S: "Value_A"}, {"_id": 3, T: "01/09/2026 00:00:00", S: "Value_A"}]
    fixed = normalize(facts, find(sort=[[T, direction], ["_id", 1]]))
    output = execute(fixed["pipeline"], rows)
    assert sorted(output, key=lambda row: row["_id"]) == rows
    assert [row["_id"] for row in output] == ([0, 1, 2, 3] if direction == 1 else [3, 0, 1, 2])


@pytest.mark.parametrize("fmt", [None, "", "ambiguous", "ISO-8601", "%d/%m/%Y %H:%M:%S.%f"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_missing_or_unsupported_sort_format_needs_metadata_review_not_llm(facts, fmt, aggregate):
    facts[C]["fields"][T]["date_format"] = fmt
    candidate = {"operation": "aggregate", "collection": C, "pipeline": [{"$sort": {T: -1}}]} if aggregate else find()
    with pytest.raises(QueryValidationError) as error:
        normalize(facts, candidate)
    assert error.value.details["code"] in {"date_format_unknown", "date_format_unsupported"}
    assert not error.value.repairable


@pytest.mark.parametrize("kind", ["native", "sortable", "unprofiled"])
def test_already_safe_sort_is_unchanged(facts, kind):
    if kind == "native":
        facts[C]["fields"][T].update(types={"date": 1}, date_format="BSON Date")
    elif kind == "sortable":
        facts[C]["fields"][T].update(date_format="%Y-%m-%dT%H:%M:%SZ", sortable_as_string=True)
    else:
        facts.clear()
    candidate = find()
    assert normalize(facts, candidate) == candidate


@pytest.mark.parametrize("shape", ["alias", "lookup", "facet", "union"])
def test_scoped_sorts_rewrite_and_validate_idempotently(facts, shape):
    stages = {
        "alias": [{"$set": {"aliasX": "$" + T}}, {"$sort": {"aliasX": -1}}, {"$match": {S: "Value_A"}}],
        "lookup": [{"$lookup": {"from": D, "as": "joinedX", "pipeline": [{"$sort": {T: -1}}, {"$limit": 2}]}}],
        "facet": [{"$facet": {"partX": [{"$sort": {T: -1}}, {"$limit": 2}]}}],
        "union": [{"$unionWith": {"coll": D, "pipeline": [{"$sort": {T: -1}}, {"$limit": 2}]}}],
    }[shape]
    fixed = normalize(facts, {"operation": "aggregate", "collection": C, "pipeline": stages})
    assert "$dateFromString" in json.dumps(fixed)
    ValidatorAgent().validate(fixed, schema(facts), facts.get)
    assert normalize(facts, fixed) == fixed


@pytest.mark.parametrize("fmt", ["%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"])
def test_explicit_timezone_applies_only_to_timezone_free_storage(facts, monkeypatch, fmt):
    monkeypatch.setenv("DATE_TIMEZONE", "Europe/Paris")
    facts[C]["fields"][T]["date_format"] = fmt
    fixed = normalize(facts, find())
    parser = fixed["pipeline"][1]["$replaceRoot"]["newRoot"]["__date_sort_0"]["$cond"][2]["$dateFromString"]
    assert parser.get("timezone") == (None if "%z" in fmt or fmt.endswith("Z") else "Europe/Paris")


def test_multiple_date_keys_and_nondate_sort_order_are_preserved(facts):
    facts[C]["fields"]["secondX"] = deepcopy(facts[C]["fields"][T])
    fixed = normalize(facts, find(sort=[[S, 1], [T, -1], ["secondX", 1], ["_id", -1]]))
    ordering = fixed["pipeline"][2]["$sort"]
    assert list(ordering.items()) == [("__date_sort_document." + S, 1), ("__date_sort_0", -1),
                                      ("__date_sort_1", 1), ("__date_sort_document._id", -1)]


def test_range_rewrite_survives_find_to_aggregation_conversion(facts):
    fixed = normalize(facts, find(filter={S: "Value_A", T: {"$gte": "2026-09-01T00:00:00Z"}}))
    predicate = fixed["pipeline"][0]["$match"]
    assert predicate["$and"][0] == {S: "Value_A"}
    assert "$dateFromString" in json.dumps(predicate)
    assert normalize(facts, fixed) == fixed
    ValidatorAgent().validate(fixed, schema(facts), facts.get)


def test_find_only_projection_never_silently_changes_meaning(facts):
    with pytest.raises(QueryValidationError, match="find-only projection"):
        normalize(facts, find(projection={"itemsX": {"$elemMatch": {"valueX": 1}}}))


@pytest.mark.parametrize("direction,expected", [(1, [1, 0]), (-1, [0, 1])])
def test_array_values_keep_mongodb_minimum_or_maximum_sort_semantics(facts, direction, expected):
    rows = [{"_id": 0, T: ["30/09/2026 00:00:00", "01/10/2026 00:00:00"], S: "Value_A"},
            {"_id": 1, T: ["01/09/2026 00:00:00", "15/09/2026 00:00:00"], S: "Value_A"}]
    fixed = normalize(facts, find(sort=[[T, direction]]))
    output = execute(fixed["pipeline"], rows)
    assert [row["_id"] for row in output] == expected
    assert sorted(output, key=lambda row: row["_id"]) == rows


@pytest.mark.parametrize("sort", [{}, [], T, {T: 0}, {T: True}, {T: {"wrong": "metadata"}}])
def test_malformed_aggregation_sort_is_a_clear_repairable_error(facts, sort):
    with pytest.raises(QueryValidationError, match="sort") as error:
        normalize(facts, {"operation": "aggregate", "collection": C, "pipeline": [{"$sort": sort}]})
    assert error.value.repairable
