"""Date range safety on synthetic records; no MongoDB or model services."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from agents.validator_agent import ValidatorAgent
from database.value_codec import decode_special_values
from services.aggregation_validation import QueryValidationError
from services.query.dates.normalization import apply_date_normalization


C, D, T, S = "synthetic_rows", "synthetic_other", "momentX", "stateX"
NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)
BOUNDS = {"$gte": "01/09/2026 00:00:00", "$lt": "01/10/2026 00:00:00"}


@pytest.fixture
def facts(monkeypatch):
    for name, value in {"DATE_NORMALIZATION_ENABLED": "true", "DATE_ENFORCE_QUESTION_RANGE": "true",
                        "DATE_TIMEZONE": "UTC", "DATE_PLAUSIBILITY_MODE": "off",
                        "PROFILE_VALIDATION_MODE": "strict", "PLAN_MODE": "off"}.items():
        monkeypatch.setenv(name, value)
    profile = {"fields": {T: {"kind": "date", "types": {"string": 1},
                              "date_format": "%d/%m/%Y %H:%M:%S", "sortable_as_string": False},
                          S: {"kind": "text", "types": {"string": 1}}}}
    return {name: {**deepcopy(profile), "collection": name} for name in (C, D)}


def schema(facts):
    return {"collections": {name: {"fields": {path: list(spec["types"]) for path, spec in profile["fields"].items()}}
                            for name, profile in facts.items()}}


def query(predicate=None, pipeline=None):
    if pipeline is not None:
        return {"operation": "aggregate", "collection": C, "pipeline": pipeline}
    return {"operation": "find", "collection": C, "filter": predicate if predicate is not None else {T: deepcopy(BOUNDS)},
            "projection": None, "sort": [], "limit": None, "skip": 0}


def normalize(facts, candidate, question=""):
    return apply_date_normalization(candidate, facts.get, question, NOW)


def expression(value, row):
    """Evaluate only the emitted MongoDB expression subset on fake records."""
    if isinstance(value, str) and value.startswith("$"):
        return row.get(value[1:])
    if not isinstance(value, dict):
        return value
    operator, operands = next(iter(value.items()))
    if operator == "$dateFromString":
        try:
            return datetime.strptime(expression(operands["dateString"], row), operands["format"]).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return operands["onError"]
    if operator == "$and":
        return all(expression(child, row) for child in operands)
    left, right = [expression(child, row) for child in operands]
    if operator == "$ne":
        return left != right
    if operator == "$eq":
        return left == right
    # MongoDB orders null before dates. Model it to expose missing null guards.
    if left is None:
        return operator in {"$lt", "$lte"}
    return {"$gt": lambda: left > right, "$gte": lambda: left >= right,
            "$lt": lambda: left < right, "$lte": lambda: left <= right}[operator]()


def matches(predicate, row):
    for key, value in predicate.items():
        if key == "$and":
            if not all(matches(child, row) for child in value):
                return False
        elif key == "$or":
            if not any(matches(child, row) for child in value):
                return False
        elif key == "$expr":
            if not expression(value, row):
                return False
        elif isinstance(value, dict):
            if not all(expression({op: ["$" + key, boundary]}, row) for op, boundary in value.items()):
                return False
        elif row.get(key) != value:
            return False
    return True


def decoded(predicate):
    # The real execution path decodes Extended JSON before sending to MongoDB.
    return decode_special_values(predicate)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("suffix", ["", " 00:00:00"])
def test_september_regression_on_five_records(facts, aggregate, suffix):
    if not suffix:
        facts[C]["fields"][T]["date_format"] = "%d/%m/%Y"
    rows = [{T: value + suffix, S: "Value_A"} for value in
            ["31/08/2026", "01/09/2026", "15/09/2026", "30/09/2026", "01/10/2026"]]
    unsafe = {S: "Value_A", T: {"$gte": "01/09/2026" + suffix, "$lt": "01/10/2026" + suffix}}
    assert [row[T] for row in rows if matches(unsafe, row)] == ["01/09/2026" + suffix]
    candidate = query(pipeline=[{"$match": unsafe}]) if aggregate else query(unsafe)
    original = deepcopy(candidate)
    fixed = normalize(facts, candidate)
    assert candidate == original
    ValidatorAgent().validate(fixed, schema(facts), facts.get)
    predicate = fixed["pipeline"][0]["$match"] if aggregate else fixed["filter"]
    assert [row[T] for row in rows if matches(decoded(predicate), row)] == [
        value + suffix for value in ("01/09/2026", "15/09/2026", "30/09/2026")]
    assert normalize(facts, fixed) == fixed


@pytest.mark.parametrize("operator", ["$gt", "$gte", "$lt", "$lte"])
def test_each_range_and_invalid_or_missing_dates(facts, operator):
    fixed = normalize(facts, query({T: {operator: BOUNDS["$gte"]}}))
    predicate = decoded(fixed["filter"])
    for row in [{}, {T: None}, {T: "garbage"}, {T: "31/02/2026 00:00:00"}]:
        assert not matches(predicate, row)
    expected = {"$gt": False, "$gte": True, "$lt": False, "$lte": True}
    assert matches(predicate, {T: BOUNDS["$gte"]}) == expected[operator]


@pytest.mark.parametrize("mode", ["strict", "warn", "off"])
@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("kind", ["find", "match", "expr", "not", "lookup", "alias"])
def test_validator_always_rejects_raw_ranges(facts, monkeypatch, mode, enabled, kind):
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", mode)
    monkeypatch.setenv("DATE_NORMALIZATION_ENABLED", enabled)
    predicate = {T: deepcopy(BOUNDS)}
    candidate = {
        "find": query(predicate),
        "match": query(pipeline=[{"$match": predicate}]),
        "expr": query({"$expr": {"$lt": ["$" + T, "$" + S]}}),
        "not": query({T: {"$not": deepcopy(BOUNDS)}}),
        "lookup": query(pipeline=[{"$lookup": {"from": D, "as": "joinedX", "pipeline": [{"$match": predicate}]}}]),
        "alias": query(pipeline=[{"$set": {"aliasX": "$" + T}}, {"$match": {"aliasX": deepcopy(BOUNDS)}}]),
    }[kind]
    with pytest.raises(QueryValidationError, match="sortable_as_string=false") as error:
        ValidatorAgent().validate(candidate, schema(facts), facts.get)
    assert error.value.to_dict()["code"] == "date_string_ordering"


@pytest.mark.parametrize("fmt", [None, "", "ambiguous", "ISO-8601", "%d/%m/%Y %H:%M:%S.%f"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_unknown_or_unsupported_format_requires_metadata_review(facts, fmt, aggregate):
    facts[C]["fields"][T]["date_format"] = fmt
    predicate = {T: {"$gte": "2026-09-01T00:00:00Z", "$lt": "2026-10-01T00:00:00Z"}}
    candidate = query(pipeline=[{"$match": predicate}]) if aggregate else query(predicate)
    with pytest.raises(QueryValidationError, match="format") as error:
        normalize(facts, candidate)
    assert not error.value.repairable
    assert error.value.to_dict()["code"] in {"date_format_unknown", "date_format_unsupported"}


@pytest.mark.parametrize("predicate", [{}, {T: "2026-09-01"}])
def test_unknown_format_question_cannot_be_silently_ignored(facts, predicate):
    facts[C]["fields"][T]["date_format"] = None
    with pytest.raises(QueryValidationError, match="date_format_unknown"):
        normalize(facts, query(predicate), "last month")


@pytest.mark.parametrize("shape", ["or", "expr", "reversed_expr", "lookup", "facet", "alias", "question"])
def test_rewrite_nested_scopes_and_question_ranges(facts, shape):
    predicate = {T: deepcopy(BOUNDS)}
    candidate = {
        "or": query({"$or": [predicate, {S: "Value_A"}]}),
        "expr": query({"$expr": {"$gte": ["$" + T, BOUNDS["$gte"]]}}),
        "reversed_expr": query({"$expr": {"$gt": [BOUNDS["$lt"], "$" + T]}}),
        "lookup": query(pipeline=[{"$lookup": {"from": D, "as": "joinedX", "pipeline": [{"$match": predicate}]}}]),
        "facet": query(pipeline=[{"$facet": {"partX": [{"$match": predicate}]}}]),
        "alias": query(pipeline=[{"$set": {"aliasX": "$" + T}}, {"$match": {"aliasX": deepcopy(BOUNDS)}}]),
        "question": query({}),
    }[shape]
    fixed = normalize(facts, candidate, "last month")
    assert "$dateFromString" in json.dumps(fixed)
    ValidatorAgent().validate(fixed, schema(facts), facts.get)
    assert normalize(facts, fixed, "last month") == fixed


def test_calendar_equality_expands_using_parsed_date(facts):
    predicate = decoded(normalize(facts, query({T: "2026-09-15"}))["filter"])
    assert matches(predicate, {T: "15/09/2026 15:30:00"})
    assert not matches(predicate, {T: "16/09/2026 00:00:00"})


def test_no_format_is_invented_for_bson_or_sortable_strings(facts):
    facts[C]["fields"][T].update(date_format="%Y-%m-%d", sortable_as_string=True)
    assert normalize(facts, query({T: {"$gte": "2026-09-01"}}))["filter"] == {T: {"$gte": "2026-09-01"}}
    facts[C]["fields"][T].update(types={"date": 1}, date_format="BSON Date", sortable_as_string=False)
    fixed = normalize(facts, query({T: {"$gte": "2026-09-01"}}))
    assert fixed["filter"] == {T: {"$gte": {"$date": "2026-09-01T00:00:00Z"}}}


def test_normalization_cannot_hide_banned_operator(facts):
    with pytest.raises(QueryValidationError, match="Banned"):
        normalize(facts, query({T: {**BOUNDS, "$where": "privateX"}}), "last month")


@pytest.mark.parametrize("mode", ["strict", "warn", "off"])
def test_normalized_find_valid_with_each_profile_policy(facts, monkeypatch, mode):
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", mode)
    fixed = normalize(facts, query())
    assert ValidatorAgent().validate(fixed, schema(facts), facts.get) == fixed


@pytest.mark.parametrize("reference", ["$$ROOT.", "$$CURRENT."])
def test_raw_root_reference_range_rejected(facts, reference):
    with pytest.raises(QueryValidationError, match="sortable_as_string=false"):
        ValidatorAgent().validate(query({"$expr": {"$lt": [reference + T, BOUNDS["$lt"]]}}), schema(facts), facts.get)


def test_explicit_timezone_used_for_parsing_and_bounds(facts, monkeypatch):
    monkeypatch.setenv("DATE_TIMEZONE", "Europe/Paris")
    predicate = normalize(facts, query())["filter"]["$expr"]["$and"]
    assert predicate[0]["$ne"][0]["$dateFromString"]["timezone"] == "Europe/Paris"
    assert predicate[1]["$gte"][1] == {"$date": "2026-08-31T22:00:00Z"}
    assert predicate[2]["$lt"][1] == {"$date": "2026-09-30T22:00:00Z"}


def test_unspecified_timezone_preserves_wall_clock_bounds_on_repeat(facts, monkeypatch):
    from zoneinfo import ZoneInfo
    from services.query.profile_config import RuntimeProfileConfig
    monkeypatch.setenv("DATE_TIMEZONE", "")
    # Emulate a non-UTC local OS zone without changing the machine's settings.
    monkeypatch.setattr(RuntimeProfileConfig, "zone", lambda self: ZoneInfo("Europe/Paris"))
    fixed = normalize(facts, query(), "last month")
    terms = fixed["filter"]["$expr"]["$and"]
    assert "timezone" not in terms[0]["$ne"][0]["$dateFromString"]
    assert terms[1]["$gte"][1] == {"$date": "2026-09-01T00:00:00Z"}
    assert normalize(facts, fixed, "last month") == fixed


def test_unsupported_element_scope_reports_repair(facts):
    facts[C]["fields"]["arrayX." + T] = facts[C]["fields"].pop(T)
    with pytest.raises(QueryValidationError, match="inside \\$elemMatch"):
        normalize(facts, query({"arrayX": {"$elemMatch": {T: BOUNDS}}}))


@pytest.mark.parametrize("stage", [{"$lookup": {}}, {"$lookup": []}, {"$facet": []}, {"$match": []}])
def test_malformed_stage_remains_a_repairable_validation_error(facts, stage):
    with pytest.raises(QueryValidationError) as error:
        normalize(facts, query(pipeline=[stage]))
    assert error.value.repairable


@pytest.mark.parametrize("sort", [T, [T]])
def test_malformed_sort_remains_a_repairable_validation_error(facts, sort):
    candidate = query()
    candidate["sort"] = sort
    with pytest.raises(QueryValidationError, match="find.sort"):
        normalize(facts, candidate)
