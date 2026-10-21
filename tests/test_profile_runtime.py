"""Runtime profile consumers use synthetic facts and no external services."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from agents.validator_agent import ValidatorAgent, QueryValidationError
from services.query.dates.resolver import resolve_question_dates
from services.query.dates.normalization import apply_date_normalization, parse_date_literal
from services.query.profile_view import profile_to_generator_view

C, D = "qzv_collection_701", "qzv_collection_702"
F, T, I, N, P, E, M = ("qzv_state_713", "qzv_moment_719", "qzv_link_727", "qzv_nested_733",
                        "qzv_private_739", "qzv_excluded_743", "qzv_mixed_751")
VALUES = ["qzv_OpÉn_761", "QZV_OPÉN_761", "qzv_closéd_769"]
SECRET = "qzv_sensitive_773"
NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def date_spec(fmt="%Y-%m-%dT%H:%M:%SZ", types=None, sortable=True):
    return {"kind": "date", "types": types or {"string": 1}, "date_format": fmt,
            "sortable_as_string": sortable, "date_min": "2026-01-01", "date_max": "2026-10-01",
            "date_range_source": "full"}


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    import services.profiling.accessors as accessors
    monkeypatch.setattr(accessors, "_profiles", {})
    for key, value in {"DATE_NORMALIZATION_ENABLED": "true", "DATE_ENFORCE_QUESTION_RANGE": "true",
                       "DATE_PLAUSIBILITY_MODE": "error", "DATE_PLAUSIBILITY_MARGIN_DAYS": "365",
                       "DATE_DAY_FIRST": "true", "DATE_TIMEZONE": "UTC", "DATE_EXPRESSIONS_FILE": "",
                       "GENERATOR_PROFILE_FACTS_ENABLED": "true", "GENERATOR_FACTS_MAX_VALUES_PER_FIELD": "15",
                       "GENERATOR_FACTS_MAX_CHARS": "2400", "PROFILE_VALIDATION_MODE": "strict",
                       "RELATIONSHIP_ENFORCEMENT": "warn"}.items():
        monkeypatch.setenv(key, value)
    from agent.llm import llm_router
    monkeypatch.setattr(llm_router, "invoke", Mock(side_effect=AssertionError("Supply a canned model")))


@pytest.fixture
def profiles():
    profile = {"collection": C, "fields": {
        F: {"kind": "categorical", "types": {"string": 1}, "values": [{"value": v, "count": 42} for v in VALUES]},
        T: date_spec(), I: {"kind": "identifier", "types": {"string": 1}, "values": [{"value": SECRET}]},
        N + "." + F: {"kind": "text", "types": {"string": 1}},
        P: {"kind": "categorical", "types": {"string": 1}, "pii_suspected": True, "values": [{"value": SECRET}]},
        E: {"kind": "categorical", "types": {"string": 1}, "values": [{"value": SECRET}]},
        M: {"kind": "unknown", "types": {"string": .5, "int": .5}}},
        "config": {"exclude_fields": [E]}, "relationships": [
            {"field": I, "target_collection": D, "target_field": I}]}
    return {C: profile, D: {"collection": D, "fields": {I: {"kind": "identifier", "types": {"string": 1}},
                                                           T: date_spec()}, "relationships": []}}


def query(predicate=None, pipeline=None):
    return {"operation": "find", "collection": C, "filter": predicate or {}} if pipeline is None else {
        "operation": "aggregate", "collection": C, "pipeline": pipeline}


def schema(profiles):
    return {"collections": {name: {"fields": {path: list(spec["types"]) for path, spec in profile["fields"].items()}}
                            for name, profile in profiles.items()}}


def normalize(profiles, q, question="", now=NOW):
    return apply_date_normalization(q, profiles.get, question, now)


def validate(profiles, q):
    return ValidatorAgent().validate(q, schema(profiles), profiles.get)


@pytest.mark.parametrize("phrase,start,end", [
    ("today", "2026-10-15", "2026-10-16"), ("aujourd'hui", "2026-10-15", "2026-10-16"),
    ("yesterday", "2026-10-14", "2026-10-15"), ("hier", "2026-10-14", "2026-10-15"),
    ("last 4 days", "2026-10-11", "2026-10-15"), ("4 derniers jours", "2026-10-11", "2026-10-15"),
    ("this month", "2026-10-01", "2026-11-01"), ("ce mois", "2026-10-01", "2026-11-01"),
    ("last week", "2026-10-08", "2026-10-15"), ("la semaine dernière", "2026-10-08", "2026-10-15"),
    ("last month", "2026-09-01", "2026-10-01"), ("le mois dernier", "2026-09-01", "2026-10-01"),
    ("this week", "2026-10-12", "2026-10-19"), ("cette semaine", "2026-10-12", "2026-10-19"),
    ("since 2026-10-01", "2026-10-01", None), ("depuis 01/10/2026", "2026-10-01", None),
    ("before 2026-10-01", None, "2026-10-01"), ("avant 01/10/2026", None, "2026-10-01"),
    ("until 2026-10-01", None, "2026-10-02"), ("jusqu'au 01/10/2026", None, "2026-10-02"),
    ("between 2026-10-01 and 2026-10-03", "2026-10-01", "2026-10-04"),
    ("entre 01/10/2026 et 03/10/2026", "2026-10-01", "2026-10-04"),
    ("October 3, 2026", "2026-10-03", "2026-10-04"), ("3 octobre 2026", "2026-10-03", "2026-10-04")])
def test_locale_constraints(phrase, start, end):
    constraints = resolve_question_dates(phrase, NOW, ZoneInfo("UTC"), True)
    assert len(constraints) == 1
    item = constraints[0]
    assert (item["start"].date().isoformat() if item["start"] else None) == start
    assert (item["end_exclusive"].date().isoformat() if item["end_exclusive"] else None) == end
    assert item["text"] and item["kind"] in {"relative", "explicit", "range", "open_start", "open_end"}


@pytest.mark.parametrize("day_first,expected", [(True, "2026-04-03"), (False, "2026-03-04")])
def test_explicit_policy(day_first, expected):
    item, = resolve_question_dates("03/04/2026", NOW, "UTC", day_first)
    assert item["ambiguous"] and item["start"].date().isoformat() == expected


def test_invalid_date(caplog):
    assert resolve_question_dates("31/02/2026", NOW, "UTC", True) == []
    assert "date_expression_invalid" in caplog.text


def test_calendar_dst_and_local_boundary(profiles, monkeypatch):
    monkeypatch.setenv("DATE_TIMEZONE", "Europe/Paris")
    clock = datetime(2026, 3, 29, 12, tzinfo=timezone.utc)
    item, = resolve_question_dates("today", clock, "Europe/Paris", True)
    assert (item["end_exclusive"].timestamp() - item["start"].timestamp()) == 23 * 3600
    profiles[C]["fields"][T] = date_spec("native", {"date": 1})
    result = normalize(profiles, query(), "today", clock)["filter"][T]
    assert result["$gte"] == {"$date": "2026-03-28T23:00:00Z"}
    assert result["$lt"] == {"$date": "2026-03-29T22:00:00Z"}
    # Clock is still the previous UTC day, already the next local day.
    assert resolve_question_dates("today", datetime(2026, 10, 14, 23, 30, tzinfo=timezone.utc),
                                  "Europe/Paris", True)[0]["start"].day == 15


@pytest.mark.parametrize("fmt,types,literal,expected", [
    ("%d/%m/%Y", {"string": 1}, {"$date": "2026-10-15T00:00:00Z"}, "15/10/2026"),
    ("native", {"date": 1}, "2026-10-15T01:02:03Z", {"$date": "2026-10-15T01:02:03Z"}),
    ("epoch_seconds", {"long": 1}, "2026-10-15T00:00:00Z", 1792022400),
    ("epoch_milliseconds", {"long": 1}, {"$date": "2026-10-15T00:00:00Z"}, 1792022400000)])
def test_stored_representation(profiles, fmt, types, literal, expected):
    profiles[C]["fields"][T] = date_spec(fmt, types)
    result = normalize(profiles, query({T: literal}))
    assert result["filter"][T] == expected
    assert normalize(profiles, result) == result


@pytest.mark.parametrize("operator", ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"])
def test_comparisons(profiles, operator):
    value = ["2026-10-15T01:00:00Z"] if operator in {"$in", "$nin"} else "2026-10-15T01:00:00Z"
    assert normalize(profiles, query({T: {operator: value}}))["filter"][T] == {operator: value}


def test_day_equality_and_question_correction(profiles):
    expected = {"$gte": "2026-10-15T00:00:00Z", "$lt": "2026-10-16T00:00:00Z"}
    assert normalize(profiles, query({T: "2026-10-15"}))["filter"][T] == expected
    assert normalize(profiles, query({T: "2026-10-01"}), "today")["filter"][T] == expected
    assert normalize(profiles, query(), "today")["filter"][T] == expected


def test_ambiguous_date_target_repairable(profiles):
    profiles[C]["fields"][T + "_other"] = date_spec()
    with pytest.raises(QueryValidationError, match="date fields available") as error:
        normalize(profiles, query(), "today")
    assert error.value.repairable


@pytest.mark.parametrize("q", [query({T: {"$gte": "2026-10-15"}}),
                              {**query(), "sort": [[T, 1]]},
                              query(pipeline=[{"$match": {T: {"$lt": "2026-10-15"}}}])])
def test_nonsortable_dates(profiles, q):
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    if q.get("sort"):
        fixed = normalize(profiles, q)
        assert fixed["operation"] == "aggregate"
        assert "$dateFromString" in json.dumps(fixed)
        validate(profiles, fixed)
    else:
        with pytest.raises(QueryValidationError, match="cannot be ordered/compared as text"):
            validate(profiles, q)
        fixed = normalize(profiles, q)
        assert "$dateFromString" in json.dumps(fixed)
        validate(profiles, fixed)
    assert normalize(profiles, query({T: {"$date": "2026-10-15T00:00:00Z"}}), "today")["filter"][T] == "15/10/2026"
    assert "$dateFromString" in json.dumps(normalize(profiles, query(), "last week"))


def test_plausibility_and_explicit_override(profiles, monkeypatch, caplog):
    q = query({T: "2099-12-31"})
    with pytest.raises(QueryValidationError, match="data range"):
        normalize(profiles, q)
    assert normalize(profiles, q, "on 2099-12-31")["filter"][T]["$gte"].startswith("2099")
    monkeypatch.setenv("DATE_PLAUSIBILITY_MODE", "warn")
    normalize(profiles, q)
    assert "date_plausibility_warning" in caplog.text
    monkeypatch.setenv("DATE_PLAUSIBILITY_MODE", "off")
    assert normalize(profiles, q)


def test_sample_lower_margin(profiles):
    q = query({T: "2024-06-01T00:00:00Z"})
    with pytest.raises(QueryValidationError):
        normalize(profiles, q)
    profiles[C]["fields"][T]["date_range_source"] = "sample"
    assert normalize(profiles, q)


@pytest.mark.parametrize("logic", ["$and", "$or"])
def test_logical_idempotence(profiles, logic):
    q = query({logic: [{T: "2026-10-01"}, {F: VALUES[0]}]})
    result = normalize(profiles, q, "today")
    assert normalize(profiles, result, "today") == result
    if logic == "$or":
        assert result["filter"]["$and"][1][T]["$gte"].startswith("2026-10-15")


def test_nested_pipelines_and_alias_dates(profiles):
    q = query(pipeline=[{"$lookup": {"from": D, "as": "childQ", "pipeline": [
        {"$match": {T: {"$date": "2026-10-15T01:00:00Z"}}}, {"$project": {"aliasQ": "$" + T}}]}},
        {"$facet": {"branchQ": [{"$match": {T: {"$date": "2026-10-15T02:00:00Z"}}}]} }])
    result = normalize(profiles, q)
    assert result["pipeline"][0]["$lookup"]["pipeline"][0]["$match"][T] == "2026-10-15T01:00:00Z"
    assert result["pipeline"][1]["$facet"]["branchQ"][0]["$match"][T] == "2026-10-15T02:00:00Z"
    alias = query(pipeline=[{"$project": {"aliasQ": "$" + T}}, {"$match": {"aliasQ": "2026-10-15"}}])
    assert "$gte" in normalize(profiles, alias)["pipeline"][1]["$match"]["aliasQ"]


def test_flags_and_absent_profile(profiles, monkeypatch):
    q = query({T: {"$date": "2026-10-15T01:00:00Z"}})
    assert apply_date_normalization(q, lambda _: None, "today", NOW) is q
    assert profile_to_generator_view(schema(profiles), "today", lambda _: None, NOW) == ""
    monkeypatch.setenv("DATE_NORMALIZATION_ENABLED", "false")
    assert normalize(profiles, q, "today") is q
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "false")
    assert profile_to_generator_view(schema(profiles), "today", profiles.get, NOW) == ""
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", "off")
    assert validate(profiles, query(pipeline=[{"$match": {"ghostQ": 1}}]))


def test_unknown_format(profiles, caplog):
    profiles[C]["fields"][T]["date_format"] = "ambiguous"
    q = query({T: "opaque"})
    assert normalize(profiles, q) == q
    assert "date_format_unknown" in caplog.text


def test_generic_iso_label_cannot_invent_stored_spelling(profiles, caplog):
    profiles[C]["fields"][T]["date_format"] = "ISO-8601"
    q = query({T: {"$date": "2026-10-15T01:00:00Z"}})
    assert normalize(profiles, q) == q
    assert "date_format_unknown" in caplog.text
    with pytest.raises(QueryValidationError, match="Implausible date"):
        normalize(profiles, query({T: "2099-12-31"}))


def test_editable_week_semantics(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "config/date_expressions.json").read_text(encoding="utf-8"))
    rule = next(row for row in data["languages"]["en"]["relative"] if row["offset"] == -7)
    rule.update(unit="week", offset=-1, length=1, calendar=True)
    path = tmp_path / "locale.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("DATE_EXPRESSIONS_FILE", str(path))
    item, = resolve_question_dates("last week", NOW, "UTC", True)
    assert item["start"].date().isoformat() == "2026-10-05"
    assert item["end_exclusive"].date().isoformat() == "2026-10-12"


def test_question_enforcement_switch_and_naive_stored_time(profiles, monkeypatch):
    monkeypatch.setenv("DATE_ENFORCE_QUESTION_RANGE", "false")
    q = query({T: {"$date": "2026-10-01T01:00:00Z"}})
    assert normalize(profiles, q, "today")["filter"][T] == "2026-10-01T01:00:00Z"
    assert normalize(profiles, query(), "today")["filter"] == {}
    monkeypatch.setenv("DATE_ENFORCE_QUESTION_RANGE", "true")
    monkeypatch.setenv("DATE_TIMEZONE", "Europe/Paris")
    profiles[C]["fields"][T] = date_spec("%Y-%m-%d %H:%M:%S")
    result = normalize(profiles, query({T: {"$date": "2026-10-14T22:00:00Z"}}), "today")["filter"][T]
    assert result == {"$gte": "2026-10-15 00:00:00", "$lt": "2026-10-16 00:00:00"}


@pytest.mark.parametrize("literal_first", [False, True])
def test_date_expr_comparisons(profiles, literal_first):
    arguments = ["$" + T, {"$date": "2026-10-15T01:00:00Z"}]
    if literal_first:
        arguments.reverse()
    q = query(pipeline=[{"$match": {"$expr": {"$gte": arguments}}}])
    result = normalize(profiles, q)
    expected_operator = "$lte" if literal_first else "$gte"
    assert result["pipeline"][0]["$match"]["$expr"] == {expected_operator: ["$" + T, "2026-10-15T01:00:00Z"]}
    assert normalize(profiles, result) == result
    enforced = normalize(profiles, q, "today")
    assert enforced["pipeline"][0]["$match"]["$expr"] == {"$and": [
        {"$gte": ["$" + T, "2026-10-15T00:00:00Z"]}, {"$lt": ["$" + T, "2026-10-16T00:00:00Z"]}]}
    assert normalize(profiles, enforced, "today") == enforced


def test_view_safety_and_caps(profiles, monkeypatch):
    view = profile_to_generator_view(schema(profiles), "today " + F, profiles.get, NOW)
    assert all(value in view for value in VALUES)
    assert all(key in view for key in ("date_format", "sortable_as_string", "date_min", "date_max", "never_case_fold", "time expression"))
    assert f"{C}.{I} -> {D}.{I}" in view
    assert all(value not in view for value in (P, E, SECRET, "count=", '"count"'))
    monkeypatch.setenv("GENERATOR_FACTS_MAX_VALUES_PER_FIELD", "1")
    view = profile_to_generator_view(schema(profiles), "today " + F, profiles.get, NOW)
    assert sum(value in view for value in VALUES) == 1
    monkeypatch.setenv("GENERATOR_FACTS_MAX_CHARS", "250")
    assert len(profile_to_generator_view(schema(profiles), "today", profiles.get, NOW)) <= 250
    only = {"collections": {C: schema(profiles)["collections"][C]}}
    assert f"{D}.{T}" not in profile_to_generator_view(only, "today", profiles.get, NOW)


def test_refinement_facts_preserve_prior_dates(profiles):
    view = profile_to_generator_view(schema(profiles), "limit 5", profiles.get, NOW, fallback_question="today")
    assert 'time expression "today"' in view
    view = profile_to_generator_view(schema(profiles), "yesterday", profiles.get, NOW, fallback_question="today")
    assert 'time expression "yesterday"' in view and 'time expression "today"' not in view


@pytest.mark.parametrize("path", ["ghostQ", N + ".ghostQ"])
@pytest.mark.parametrize("aggregate", [True, False])
def test_full_paths(profiles, path, aggregate):
    q = query(pipeline=[{"$match": {path: 1}}]) if aggregate else query({path: 1})
    with pytest.raises(QueryValidationError, match="closest valid paths") as error:
        validate(profiles, q)
    assert path in str(error.value) and SECRET not in str(error.value)


@pytest.mark.parametrize("path", [N, N + ".0." + F, N + ".$[]." + F])
def test_parent_and_array_paths(profiles, path):
    assert validate(profiles, query({path: VALUES[0]}))


@pytest.mark.parametrize("stages", [
    [{"$group": {"_id": "$" + F, "aliasQ": {"$sum": 1}}}, {"$sort": {"aliasQ": -1}}],
    [{"$project": {"aliasQ": "$" + F}}, {"$match": {"aliasQ": VALUES[0]}}],
    [{"$project": {"aliasQ": "$" + N}}, {"$match": {"aliasQ." + F: VALUES[0]}}],
    [{"$group": {"_id": {"aliasQ": "$" + F}}}, {"$match": {"_id.aliasQ": VALUES[0]}}],
    [{"$addFields": {"aliasQ": "$" + F}}, {"$match": {"aliasQ": VALUES[0]}}],
    [{"$lookup": {"from": D, "localField": I, "foreignField": I, "as": "childQ"}},
     {"$match": {"childQ." + T: "2026-10-15"}}],
    [{"$lookup": {"from": D, "as": "childQ", "pipeline": [{"$project": {"aliasQ": "$" + T}}]}},
     {"$sort": {"childQ.aliasQ": 1}}],
    [{"$count": "aliasQ"}, {"$sort": {"aliasQ": -1}}],
    [{"$unwind": {"path": "$" + N, "includeArrayIndex": "aliasQ"}}, {"$sort": {"aliasQ": 1}}]])
def test_output_scope(profiles, stages):
    assert validate(profiles, query(pipeline=stages))


def test_scope_removed_and_unknown(profiles, caplog):
    with pytest.raises(QueryValidationError):
        validate(profiles, query(pipeline=[{"$project": {F: 1}}, {"$match": {T: "2026-10-15"}}]))
    # Known nested document shapes remain checkable after root replacement.
    with pytest.raises(QueryValidationError, match="Unknown path"):
        validate(profiles, query(pipeline=[{"$replaceRoot": {"newRoot": "$" + N}}, {"$match": {"ghostQ": 1}}]))
    assert validate(profiles, query(pipeline=[{"$replaceRoot": {"newRoot": {
        "$mergeObjects": ["$" + N, {"ghostQ": 1}]}}}, {"$match": {"ghostQ": 1}}]))
    assert "pipeline_scope_unknown" in caplog.text


def test_unknown_foreign_output_scope_preserves_known_root_checks(profiles):
    stages = [{"$lookup": {"from": D, "as": "childQ", "pipeline": [
        {"$replaceWith": "$" + I}]}}, {"$match": {"childQ.ghostQ": 1}}]
    assert validate(profiles, query(pipeline=stages))
    with pytest.raises(QueryValidationError, match="Unknown path"):
        validate(profiles, query(pipeline=stages + [{"$match": {"ghostQ": 1}}]))
    selected = schema(profiles)
    profiles.pop(D)
    assert ValidatorAgent().validate(query(pipeline=[{"$lookup": {"from": D, "localField": I,
        "foreignField": "ghostQ", "as": "childQ"}}, {"$match": {"childQ.ghostQ": 1}}]), selected, profiles.get)


@pytest.mark.parametrize("stage", [
    {"$lookup": {"from": D, "localField": "ghostQ", "foreignField": I, "as": "childQ"}},
    {"$lookup": {"from": D, "localField": I, "foreignField": "ghostQ", "as": "childQ"}},
    {"$lookup": {"from": D, "as": "childQ", "let": {"refQ": "$ghostQ"}, "pipeline": []}},
    {"$lookup": {"from": D, "as": "childQ", "pipeline": [{"$match": {"$expr": {"$eq": ["$ghostQ", "$$refQ"]}}}]}},
    {"$facet": {"branchQ": [{"$match": {"ghostQ": 1}}]}},
    {"$unionWith": {"coll": D, "pipeline": [{"$match": {"ghostQ": 1}}]}}])
def test_nested_invalid_paths(profiles, stage):
    with pytest.raises(QueryValidationError, match="Unknown path"):
        validate(profiles, query(pipeline=[stage]))


def test_relationship_policy(profiles, monkeypatch, caplog):
    q = query(pipeline=[{"$lookup": {"from": D, "localField": F, "foreignField": I, "as": "childQ"}}])
    assert validate(profiles, q)
    assert f"{C}.{I} -> {D}.{I}" in caplog.text
    monkeypatch.setenv("RELATIONSHIP_ENFORCEMENT", "strict")
    with pytest.raises(QueryValidationError, match="profiled relationship"):
        validate(profiles, q)


def test_types_regex_mixed_and_dates(profiles):
    with pytest.raises(QueryValidationError, match="Incompatible literal type"):
        validate(profiles, query({F: 3}))
    with pytest.raises(QueryValidationError, match="Incompatible literal type"):
        validate(profiles, query({"$expr": {"$eq": ["$" + F, 3]}}))
    profiles[C]["fields"][I]["types"] = {"long": 1}
    with pytest.raises(QueryValidationError, match="Regex requires a string"):
        validate(profiles, query({I: {"$regex": "^x"}}))
    assert validate(profiles, query({M: 3, T: {"$date": "2026-10-15"}}))


@pytest.mark.parametrize("partial", [{"partial": True}, {"warnings": ["path cap truncated"]}, {"warnings": ["time budget exhausted"]}])
def test_partial_profile(profiles, partial, caplog):
    profiles[C].update(partial)
    assert validate(profiles, query({"ghostQ": 1}))
    assert "profile_validation_warning" in caplog.text


def test_warn_and_no_profile_legacy(profiles, monkeypatch):
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", "warn")
    assert validate(profiles, query({"ghostQ": 1}))
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", "strict")
    # Legacy root fallback still permits this sibling when no facts exist.
    assert ValidatorAgent().validate(query({N + ".ghostQ": 1}), schema(profiles), lambda _: None)


def test_new_modules_contain_no_fixture_names_or_values():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "services/query/dates").glob("*.py"))
    paths += list((root / "services/query").glob("profile_*.py"))
    forbidden = {C, D, F, T, I, N, P, E, M, SECRET, *VALUES}
    for path in paths:
        strings = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not strings & forbidden, path


def publish(profiles):
    from agent.config import MONGODB_DATABASE
    from services.profiling.accessors import publish_profiles
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in profiles.items()})


def runtime_spies(monkeypatch, module, profiles):
    """Exercise the real validators/normalizer/binder at runtime boundaries."""
    publish(profiles)
    order = []
    real_validate = ValidatorAgent.validate
    def checked(self, candidate, selected):
        order.append("validate")
        return real_validate(self, candidate, selected)
    def dates(candidate, **kwargs):
        order.append("dates")
        return apply_date_normalization(candidate, now=NOW, **kwargs)
    def values(candidate, *args):
        order.append("values")
        assert candidate["filter"][T] == "15/10/2026"
        return candidate
    from services.filter_value_resolution import parameterize_resolved_query
    def parameters(generated, candidate):
        order.append("parameters")
        assert candidate["filter"][T] == "15/10/2026"
        return parameterize_resolved_query(generated, candidate)
    monkeypatch.setattr(ValidatorAgent, "validate", checked)
    monkeypatch.setattr(module, "apply_date_normalization", dates)
    monkeypatch.setattr(module, "apply_value_resolution", values)
    monkeypatch.setattr(module, "parameterize_resolved_query", parameters)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "qzv_signature_797"}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    if hasattr(module.PerformanceLogger, "record_optimization"):
        monkeypatch.setattr(module.PerformanceLogger, "record_optimization", Mock())
    return order


def fresh_runtime(monkeypatch, profiles, initial, repaired=None):
    import orchestrator as module
    from services.query_parameter_binder import template_from_concrete
    selected = schema(profiles)
    monkeypatch.setattr(module.schema_bootstrap_service, "get_catalog", lambda: selected)
    monkeypatch.setattr(module.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module.schema_bootstrap_service, "retrieve", lambda *args: {"catalog": selected, "telemetry": {}})
    app = module.OrchestratorAgent()
    app.context_builder = SimpleNamespace(build=lambda question, catalog, knowledge, preferences, **kwargs: {
        "question": question, "schema": catalog, "database_knowledge": knowledge, "user_preferences": preferences})
    app.preference_service = Mock(user_id="qzv_user_809")
    app.preference_service.retrieve_for_query.return_value = []
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [], "telemetry": {}, "_active_rules": []}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [], "user_preferences": []}
    app.query_context_agent = Mock()
    app.query_context_agent.build.return_value = {"schema_spec": selected, "query_context": {
        "complexity": "simple", "operation": "find", "collections": [C]}}
    app.query_generator = Mock()
    app.query_generator.generate.return_value = template_from_concrete(initial)
    app.query_generator.repair.return_value = template_from_concrete(repaired or initial)
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    return app, module


@pytest.mark.parametrize("failure", [None, "path", "plausibility", "missing_target"])
def test_main_order_and_existing_repair_limit(profiles, monkeypatch, failure):
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    good = query({T: {"$date": "2026-10-15T00:00:00Z"}})
    initial = query({"ghostQ": 1}) if failure == "path" else query({T: "31/12/2099"}) if failure == "plausibility" else good
    if failure == "missing_target":
        profiles[C]["fields"][T + "_other"] = date_spec()
        initial = query()
    app, module = fresh_runtime(monkeypatch, profiles, initial, good)
    order = runtime_spies(monkeypatch, module, profiles)
    result = app._run("today" if failure != "plausibility" else "Find dates")
    assert result["execution_success"], result
    assert order[-4:] == ["dates", "validate", "values", "parameters"]
    assert app.query_generator.repair.call_count == int(failure is not None)
    if failure:
        error = app.query_generator.repair.call_args.args[3]
        assert any(term in error for term in ("Unknown path", "Implausible date", "date fields available"))
    from services.query_parameter_binder import bind_query_parameters
    generated = result["generated_query"]
    assert bind_query_parameters(generated["template"], generated["parameters"]) == result["validated_query"]


def test_failed_date_repair_is_bounded(profiles, monkeypatch):
    bad = query({T: "2099-12-31"})
    app, module = fresh_runtime(monkeypatch, profiles, bad)
    publish(profiles)
    monkeypatch.setattr(module, "apply_date_normalization", lambda candidate, **kwargs: apply_date_normalization(candidate, now=NOW, **kwargs))
    result = app._run("Find dates")
    assert not result["execution_success"] and len(result["validation_errors"]) == 2
    assert result["execution_error_code"] == "date_implausible"
    app.query_generator.repair.assert_called_once()
    app.executor.execute.assert_not_called()


def test_execution_repair_also_normalizes_dates(profiles, monkeypatch):
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    app, module = fresh_runtime(monkeypatch, profiles, query({T: "15/10/2026"}), query({T: {"$date": "2026-10-15T00:00:00Z"}}))
    order = runtime_spies(monkeypatch, module, profiles)
    app.executor.execute.side_effect = [{"ok": False, "repairable": True, "error": "Structural failure"},
                                        {"ok": True, "row_count": 0, "all_rows": []}]
    assert app._run("today")["execution_success"]
    assert order == ["dates", "validate", "values", "parameters"] * 2
    app.query_generator.repair.assert_called_once()


def test_main_learned_optimization_normalizes_before_values(profiles, monkeypatch):
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    app, module = fresh_runtime(monkeypatch, profiles, query({T: "15/10/2026"}))
    order = runtime_spies(monkeypatch, module, profiles)
    def learned(self, candidate, *args):
        candidate = deepcopy(candidate)
        candidate["filter"][T] = {"$date": "2026-10-15T00:00:00Z"}
        return candidate, {"selected": "qzv_reorder_821"}
    monkeypatch.setattr(module.LearnedOptimizationService, "apply", learned)
    result = app._run("today")
    assert result["execution_success"] and result["learned_optimization"]
    assert order == ["dates", "validate", "values", "parameters", "dates", "values"]


def test_invalid_learned_dates_keep_original(profiles, monkeypatch):
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    app, module = fresh_runtime(monkeypatch, profiles, query({T: "15/10/2026"}))
    runtime_spies(monkeypatch, module, profiles)
    monkeypatch.setattr(module.LearnedOptimizationService, "apply", lambda *args: (
        query({T: "31/12/2099"}), {"selected": "qzv_reorder_821"}))
    result = app._run("Find dates")
    assert result["execution_success"] and not result["learned_optimization"]
    assert result["validated_query"]["filter"][T] == "15/10/2026"


def template_runtime(monkeypatch, profiles, initial, question="today"):
    import services.template_decision_service as module
    from services.query_parameter_binder import template_from_concrete
    from services.query_history_service import QueryHistoryService
    from services.user_preference_service import UserPreferenceService
    from services.schema_bootstrap_service import schema_bootstrap_service
    generated = template_from_concrete(initial)
    document = {"template": generated["template"], "schema_spec": schema(profiles), "semantic_text": "qzv_intent_811"}
    pending = Mock()
    pending.find_one_and_update.return_value = {"question": question, "cache": {}}
    service = module.TemplateDecisionService()
    monkeypatch.setattr(service, "collection", lambda: pending)
    monkeypatch.setattr(module.QueryTemplateRepository, "get", lambda *args: deepcopy(document))
    failure = Mock()
    monkeypatch.setattr(module.QueryTemplateRepository, "record_failure", failure)
    db = Mock()
    db.list_collection_names.return_value = list(profiles)
    monkeypatch.setattr(module, "get_database", lambda: db)
    monkeypatch.setattr(module.SemanticTemplateCache, "compatible", lambda *args: True)
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module, "extract_parameters", lambda *args: (deepcopy(generated["parameters"]), "deterministic"))
    monkeypatch.setattr(module, "active_resolution_rules", lambda: [])
    monkeypatch.setattr(UserPreferenceService, "retrieve_for_query", lambda *args, **kwargs: [])
    monkeypatch.setattr(QueryHistoryService, "save_result", Mock())
    execute = Mock(return_value={"ok": True, "row_count": 0, "all_rows": [], "execution_time_ms": 0})
    monkeypatch.setattr(module.ExecutorAgent, "execute", execute)
    return module, service, document, failure, execute


@pytest.mark.parametrize("learned", [False, True])
def test_template_reuse_and_learned_order(profiles, monkeypatch, learned):
    import services.learned_optimization_service as learned_module
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    module, service, document, failure, execute = template_runtime(monkeypatch, profiles,
        query({T: {"$date": "2026-10-15T00:00:00Z"}}))
    original = deepcopy(document)
    order = runtime_spies(monkeypatch, module, profiles)
    if learned:
        def apply(self, candidate, *args):
            candidate = deepcopy(candidate)
            candidate["filter"][T] = {"$date": "2026-10-15T00:00:00Z"}
            return candidate, {"selected": "qzv_reorder_821"}
        monkeypatch.setattr(learned_module.LearnedOptimizationService, "apply", apply)
    result = service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    assert result["execution_success"] and result["query_template_id"] is None
    assert order[:4] == ["dates", "validate", "values", "parameters"]
    if learned:
        assert order[4:] == ["dates", "values"]
    assert document == original
    failure.assert_not_called()
    from services.query_parameter_binder import bind_query_parameters
    assert bind_query_parameters(result["generated_query"]["template"], result["query_parameters"]) == result["validated_query"]


def test_template_date_failure_uses_existing_regenerate_response(profiles, monkeypatch):
    from fastapi import HTTPException
    module, service, _, failure, execute = template_runtime(monkeypatch, profiles, query({T: "2099-12-31"}), "Find dates")
    publish(profiles)
    with pytest.raises(HTTPException, match="Choose Regenerate") as error:
        service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    assert error.value.status_code == 422
    execute.assert_not_called()
    failure.assert_called_once()


@pytest.mark.parametrize("invalid_preference", [False, True])
def test_template_preferences_do_not_reject_valid_saved_query(profiles, monkeypatch, invalid_preference, caplog):
    from services.user_preference_service import UserPreferenceService
    from services.learned_optimization_service import LearnedOptimizationService
    from services.query_parameter_binder import bind_query_parameters
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    publish(profiles)
    initial = query({I: "qzv_identifier_829"})
    module, service, document, failure, execute = template_runtime(monkeypatch, profiles, initial, "Find records")
    original = deepcopy(document)
    preference = {"type": "sorting_preference", "collection": C, "field": T if invalid_preference else F, "direction": -1}
    monkeypatch.setattr(UserPreferenceService, "retrieve_for_query", lambda *args, **kwargs: [preference])
    monkeypatch.setattr(module.QueryCostGuard, "analyze", lambda *args: {"status": "safe", "query_signature": "qzv_signature_797"})
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    monkeypatch.setattr(LearnedOptimizationService, "apply", lambda self, candidate, *args: (candidate, None))
    with caplog.at_level("INFO"):
        result = service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    assert result["execution_success"]
    assert document == original
    assert bind_query_parameters(result["generated_query"]["template"], result["query_parameters"]) == result["validated_query"]
    failure.assert_not_called()
    execute.assert_called_once()
    if invalid_preference:
        assert result["validated_query"]["operation"] == "aggregate"
        assert "$dateFromString" in json.dumps(result["validated_query"])
        assert result["applied_preferences"]
        assert not result["semantic_template_cache"].get("template_reuse_preferences_skipped")
        assert result["query_template_id"] is None
    else:
        assert result["validated_query"]["sort"] == [[F, -1]]
        assert result["applied_preferences"]
        assert result["query_template_id"] is None


@pytest.mark.parametrize("phase,code", [("validator", "profile_validation"), ("date_normalization", "date_string_ordering")])
def test_template_validation_failure_has_safe_reason_and_stage(profiles, monkeypatch, caplog, phase, code):
    from fastapi import HTTPException
    module, service, _, failure, execute = template_runtime(monkeypatch, profiles, query(), "Find records")
    error = QueryValidationError("qzv_private_exception_839", code=code)
    if phase == "validator":
        monkeypatch.setattr(module.ValidatorAgent, "validate", Mock(side_effect=error))
    else:
        monkeypatch.setattr(module, "apply_date_normalization", Mock(side_effect=error))
    history = Mock()
    module.get_database().__getitem__ = Mock(return_value=history)
    with caplog.at_level("WARNING"), pytest.raises(HTTPException) as rejected:
        service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    assert rejected.value.status_code == 422
    assert code in rejected.value.detail and phase in rejected.value.detail
    assert "reason=" + code in caplog.text and "stage=" + phase in caplog.text
    assert "qzv_private_exception_839" not in caplog.text + rejected.value.detail
    cache = history.update_one.call_args.args[1]["$set"]["semantic_template_cache"]
    assert cache["template_reuse_validation_failure_reason"] == code
    assert cache["template_reuse_validation_failure_stage"] == phase
    failure.assert_called_once()
    execute.assert_not_called()


def test_invalid_template_still_rejected_when_preferences_are_skipped(profiles, monkeypatch):
    from fastapi import HTTPException
    from services.user_preference_service import UserPreferenceService
    publish(profiles)
    module, service, _, failure, execute = template_runtime(monkeypatch, profiles, query({"qzv_missing_853": 1}), "Find records")
    monkeypatch.setattr(UserPreferenceService, "retrieve_for_query", lambda *args, **kwargs: [{
        "type": "sorting_preference", "collection": C, "field": T, "direction": -1}])
    with pytest.raises(HTTPException) as rejected:
        service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    assert rejected.value.status_code == 422
    assert "profile_validation" in rejected.value.detail
    failure.assert_called_once()
    execute.assert_not_called()


@pytest.mark.parametrize("repair", [False, True])
def test_optimization_and_repair_order(profiles, monkeypatch, repair):
    import services.query_optimization_service as module
    from services.query_parameter_binder import template_from_concrete
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y", sortable=False)
    good = query({T: {"$date": "2026-10-15T00:00:00Z"}})
    initial = query({"ghostQ": 1}) if repair else good
    generated = template_from_concrete(initial)
    pending = Mock()
    pending.find_one_and_update.return_value = {"state": "pending", "question": "today", "result": {
        "validated_query": initial, "generated_query": generated, "schema_spec": schema(profiles),
        "cost_analysis": {}, "suggestions": [{"id": "run_anyway"}]}}
    service = module.QueryOptimizationService()
    monkeypatch.setattr(service, "collection", lambda: pending)
    monkeypatch.setattr(service, "save_history", Mock())
    monkeypatch.setattr(module, "active_resolution_rules", lambda: [])
    repaired = Mock(return_value=template_from_concrete(good))
    monkeypatch.setattr(module.QueryGeneratorAgent, "repair", repaired)
    order = runtime_spies(monkeypatch, module, profiles)
    monkeypatch.setattr(module.ExecutorAgent, "execute", Mock(return_value={
        "ok": True, "all_rows": [], "row_count": 0, "execution_time_ms": 0}))
    result = service.continue_query("qzv_request_823", SimpleNamespace(
        optimization="run_anyway", refinement=None, value=None, page=1))
    assert result["execution_success"], result
    assert order[-4:] == ["dates", "validate", "values", "parameters"]
    assert repaired.call_count == int(repair)


@pytest.mark.parametrize("route", ["fresh", "reuse", "optimization"])
def test_raw_string_date_range_rewritten_before_execution(profiles, monkeypatch, route):
    from services.query_parameter_binder import bind_query_parameters, template_from_concrete
    from services.learned_optimization_service import LearnedOptimizationService
    profiles[C]["fields"][T] = date_spec("%d/%m/%Y %H:%M:%S", sortable=False)
    publish(profiles)
    initial = query({T: {"$gte": "01/09/2026 00:00:00", "$lt": "01/10/2026 00:00:00"}})
    if route == "fresh":
        app, module = fresh_runtime(monkeypatch, profiles, initial)
        execute = app.executor.execute
        run = lambda: app._run("last month")
    elif route == "reuse":
        module, service, _, failure, execute = template_runtime(monkeypatch, profiles, initial, "last month")
        run = lambda: service.decide("qzv_request_823", SimpleNamespace(action="reuse", template_id="qzv_template_827"))
    else:
        import services.query_optimization_service as module
        pending = Mock()
        pending.find_one_and_update.return_value = {"state": "pending", "question": "last month", "result": {
            "validated_query": initial, "generated_query": template_from_concrete(initial),
            "schema_spec": schema(profiles), "cost_analysis": {}, "suggestions": [{"id": "run_anyway"}]}}
        service = module.QueryOptimizationService()
        monkeypatch.setattr(service, "collection", lambda: pending)
        monkeypatch.setattr(service, "save_history", Mock())
        monkeypatch.setattr(module, "active_resolution_rules", lambda: [])
        execute = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0, "execution_time_ms": 0})
        monkeypatch.setattr(module.ExecutorAgent, "execute", execute)
        run = lambda: service.continue_query("qzv_request_823", SimpleNamespace(
            optimization="run_anyway", refinement=None, value=None, page=1))
    monkeypatch.setattr(module, "apply_date_normalization", lambda candidate, **kwargs: apply_date_normalization(candidate, now=NOW, **kwargs))
    monkeypatch.setattr(module.QueryCostGuard, "analyze", lambda *args: {"status": "safe", "query_signature": "qzv_signature_797"})
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    monkeypatch.setattr(LearnedOptimizationService, "apply", lambda self, candidate, *args: (candidate, None))
    result = run()
    assert result["execution_success"], result
    executed = execute.call_args.args[0]
    assert "$dateFromString" in json.dumps(executed)
    assert executed == result["validated_query"]
    assert bind_query_parameters(result["generated_query"]["template"], result["query_parameters"]) == executed
    if route == "fresh":
        app.query_generator.repair.assert_not_called()
    elif route == "reuse":
        failure.assert_not_called()


@pytest.mark.parametrize("method", ["generate", "repair", "optimize"])
def test_prompts_append_only(profiles, monkeypatch, method):
    from agents.query_generator_agent import QueryGeneratorAgent
    publish(profiles)
    agent = QueryGeneratorAgent()
    invoke = Mock(return_value={})
    monkeypatch.setattr(agent, "_invoke_json", invoke)
    def run():
        if method == "generate":
            agent.generate("today " + F, schema(profiles))
        elif method == "repair":
            agent.repair("today " + F, schema(profiles), query(), "Synthetic error")
        else:
            agent.optimize("today " + F, {}, schema(profiles), query(), "today " + F)
        return invoke.call_args.args[0]
    enabled = run()
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "false")
    disabled = run()
    assert enabled.startswith(disabled + "\nProfile facts:\n")
    assert F in enabled[len(disabled):] and P not in enabled[len(disabled):]
