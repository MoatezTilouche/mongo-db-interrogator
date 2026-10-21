"""Stored date metadata gaps must not spend additional chat calls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.aggregation_validation import QueryValidationError
from services.query.plans.config import PlanConfig
from services.query.plans.compiler import compile_plan
from services.query.plans.generation import generate_with_plan
from services.query.plans.models import QueryPlan
from services.query_parameter_binder import bind_query_parameters


CASE = json.loads((Path(__file__).parent / "fixtures/plan_recency_case.json").read_text(encoding="utf-8"))


@pytest.fixture
def runtime(monkeypatch):
    profile = deepcopy(CASE["profile"])
    plan = deepcopy(CASE["plan"])
    plan.update(filters=[], uses_time=False, time_field=None)
    collection, field = plan["collection"], plan["order"]["field"]
    profiles = {collection: profile}
    schema = {"collections": {collection: {"fields": {
        path: list(spec["types"]) for path, spec in profile["fields"].items()}}}}
    monkeypatch.setattr("services.query.plans.generation.build_prompt_inputs", lambda *args, **kwargs: {
        "selected_schema": schema, "schema": schema, "rules": list(args[2]), "preferences": [], "facts": ""})
    monkeypatch.setenv("DATE_TIMEZONE", "UTC")
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", "strict")
    monkeypatch.setenv("DATE_PLAUSIBILITY_MODE", "off")

    def generate(invoke, fallback, rules=(), question=None, **kwargs):
        return generate_with_plan(
            question or CASE["recency_questions"][0], schema, [], rules, PlanConfig(enabled=True, max_retries=1),
            invoke, fallback, profile_lookup=profiles.get, now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc), **kwargs)

    return SimpleNamespace(profile=profile, plan=plan, field=field, collection=collection, schema=schema, generate=generate)


@pytest.mark.parametrize("date_format,code", [
    (None, "date_format_unknown"),
    ("ISO-8601", "date_format_unknown"),
    ("%Y-%m-%d %H:%M:%S.%f", "date_format_unsupported"),
])
def test_missing_or_unsupported_sort_metadata_stops_after_first_plan(runtime, date_format, code, caplog):
    runtime.profile["fields"][runtime.field] = {
        "kind": "date", "types": {"string": 5}, "date_format": date_format, "sortable_as_string": False}
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(runtime.plan)))
    fallback = Mock(side_effect=AssertionError("Metadata cannot be fixed by generating another query"))
    with caplog.at_level("INFO"), pytest.raises(QueryValidationError) as raised:
        runtime.generate(invoke, fallback)
    error = raised.value
    assert error.to_dict()["code"] == code
    assert error.to_dict()["path"] == f"{runtime.collection}.{runtime.field}"
    assert error.repairable is False and error.to_dict()["repairable"] is False
    assert runtime.field in str(error) and "stored profile" in str(error)
    assert invoke.call_count == 1
    fallback.assert_not_called()
    assert "path=blocked" in caplog.text
    assert "plan_retry" not in caplog.text and "plan_fallback" not in caplog.text


@pytest.mark.parametrize("code", ["date_format_unknown", "date_format_unsupported"])
def test_normalizer_metadata_error_keeps_details_and_skips_retry(runtime, monkeypatch, code):
    error = QueryValidationError("Stored date metadata requires review", code=code,
                                 path=runtime.field, stage_index=0)
    monkeypatch.setattr("services.query.plans.generation.apply_date_normalization", Mock(side_effect=error))
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(runtime.plan)))
    fallback = Mock()
    with pytest.raises(QueryValidationError) as raised:
        runtime.generate(invoke, fallback)
    assert raised.value is error
    assert error.to_dict()["path"] == runtime.field and error.to_dict()["stage_index"] == 0
    assert error.repairable is False
    assert invoke.call_count == 1
    fallback.assert_not_called()


def test_metadata_error_before_model_call_is_not_fallen_back(runtime, monkeypatch):
    error = QueryValidationError("Date format unavailable", code="date_format_unknown")
    monkeypatch.setattr("services.query.plans.generation.build_prompt_inputs", Mock(side_effect=error))
    invoke, fallback = Mock(), Mock()
    with pytest.raises(QueryValidationError) as raised:
        runtime.generate(invoke, fallback)
    assert raised.value is error
    invoke.assert_not_called()
    fallback.assert_not_called()


def test_recoverable_model_shape_keeps_one_retry(runtime):
    invalid = {**runtime.plan, "task": "unsupported"}
    invoke = Mock(side_effect=[SimpleNamespace(content=json.dumps(invalid)),
                               SimpleNamespace(content=json.dumps(runtime.plan))])
    fallback = Mock(side_effect=AssertionError("A valid second plan should succeed"))
    result = runtime.generate(invoke, fallback)
    assert result["template"]["operation"] == "find"
    assert invoke.call_count == 2
    fallback.assert_not_called()


def test_unrelated_validation_error_still_retries_then_falls_back(runtime):
    validator = Mock()
    validator.validate.side_effect = QueryValidationError("Invalid model-generated structure", code="invalid_query")
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(runtime.plan)))
    fallback = Mock(return_value={"fallback": True})
    assert runtime.generate(invoke, fallback, validator=validator) == {"fallback": True}
    assert invoke.call_count == 2 and fallback.call_count == 1


def test_missing_format_does_not_block_queries_without_date_operations(runtime):
    runtime.profile["fields"][runtime.field] = {
        "kind": "date", "types": {"string": 5}, "sortable_as_string": False}
    runtime.plan["order"] = None
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(runtime.plan)))
    fallback = Mock(side_effect=AssertionError("No date operation needs a format"))
    assert runtime.generate(invoke, fallback)["template"]["operation"] == "find"
    assert invoke.call_count == 1
    fallback.assert_not_called()


@pytest.mark.parametrize("date_format,code", [
    (None, "date_format_unknown"),
    ("ISO-8601", "date_format_unknown"),
    ("%Y-%m-%d %H:%M:%S.%f", "date_format_unsupported"),
])
def test_latest_directive_metadata_gap_stops_before_any_model_call(runtime, date_format, code):
    runtime.profile["fields"][runtime.field] = {
        "kind": "date", "types": {"string": 5}, "date_format": date_format, "sortable_as_string": False}
    key = next(path for path, spec in runtime.profile["fields"].items() if spec.get("kind") == "categorical")
    rules = [{"text": "latest state", "mandatory": True, "collections": [runtime.collection], "directives": [{
        "type": "latest_per_key", "status": "approved", "collection": runtime.collection,
        "key_field": key, "order_field": runtime.field}]}]
    invoke, fallback = Mock(), Mock()
    with pytest.raises(QueryValidationError) as raised:
        runtime.generate(invoke, fallback, rules=rules)
    assert raised.value.to_dict()["code"] == code
    assert raised.value.to_dict()["path"] == f"{runtime.collection}.{runtime.field}"
    assert raised.value.repairable is False
    invoke.assert_not_called()
    fallback.assert_not_called()


def test_latest_directive_unknown_field_remains_recoverable(runtime):
    rules = [{"text": "latest state", "mandatory": True, "collections": [runtime.collection], "directives": [{
        "type": "latest_per_key", "status": "approved", "collection": runtime.collection,
        "key_field": "unknown_field", "order_field": runtime.field}]}]
    invoke, fallback = Mock(), Mock(return_value={"fallback": True})
    assert runtime.generate(invoke, fallback, rules=rules) == {"fallback": True}
    invoke.assert_not_called()
    fallback.assert_called_once()


@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize("date_format,has_timezone", [
    ("%d/%m/%Y %H:%M:%S", True), ("%Y-%m-%dT%H:%M:%SZ", False), ("%Y-%m-%dT%H:%M:%S%z", False)])
def test_plan_sort_parser_uses_same_timezone_rules_as_date_filters(runtime, monkeypatch, latest, date_format, has_timezone):
    monkeypatch.setenv("DATE_TIMEZONE", "America/New_York")
    runtime.profile["fields"][runtime.field] = {
        "kind": "date", "types": {"string": 5}, "date_format": date_format, "sortable_as_string": False}
    schema = {"collections": {runtime.collection: {"fields": {
        path: list(spec["types"]) for path, spec in runtime.profile["fields"].items()}}}}
    key = next(path for path, spec in runtime.profile["fields"].items() if spec.get("kind") == "categorical")
    directives = [{"type": "latest_per_key", "status": "approved", "collection": runtime.collection,
                   "key_field": key, "order_field": runtime.field}] if latest else []
    query, _ = compile_plan(QueryPlan.model_validate(runtime.plan), runtime.profile, schema, "", directives=directives,
                            profile_lookup=lambda name: runtime.profile if name == runtime.collection else None)
    parsers = [expression["$dateFromString"] for stage in query["pipeline"] if "$addFields" in stage
               for expression in stage["$addFields"].values()]
    assert parsers
    assert all(parser["format"] == date_format for parser in parsers)
    if has_timezone:
        assert all(parser.get("timezone") == "America/New_York" for parser in parsers)
    else:
        assert all("timezone" not in parser for parser in parsers)


def test_last_day_plan_keeps_status_and_uses_bounded_parsed_dates_without_retry(runtime, monkeypatch):
    runtime.profile["fields"][runtime.field] = {
        "kind": "date", "types": {"string": 5}, "date_format": "%d/%m/%Y %H:%M:%S", "sortable_as_string": False}
    runtime.schema["collections"][runtime.collection]["fields"][runtime.field] = ["string"]
    state = next(path for path, spec in runtime.profile["fields"].items() if spec.get("kind") == "categorical")
    value = runtime.profile["fields"][state]["values"][0]["value"]
    monkeypatch.setattr("services.query.values.resolution.get_value_catalog", lambda *args: {value})
    monkeypatch.setattr("services.query.values.resolution.get_value_catalog_status", lambda *args: "not_categorical")
    runtime.plan.update(filters=[{"field": state, "op": "eq", "value": value},
                                 {"field": runtime.field, "op": "gte", "value": "2026-10-20T00:00:00"}],
                        uses_time=True, time_field=runtime.field, limit=5)
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(runtime.plan)))
    fallback = Mock(side_effect=AssertionError("Valid date plans should not enter free-form generation"))
    generated = runtime.generate(invoke, fallback, question=f"give records with {state} {value} in the last day")
    query = bind_query_parameters(generated["template"], generated["parameters"])
    assert query["operation"] in {"find", "aggregate"}
    if query["operation"] == "find":
        assert query.get("limit") is None
        predicate = query["filter"]
    else:
        assert not any("$limit" in stage for stage in query["pipeline"])
        assert not any(runtime.field in stage["$sort"] for stage in query["pipeline"] if "$sort" in stage)
        predicate = next(stage["$match"] for stage in query["pipeline"] if "$match" in stage)
    assert predicate["$and"][0] == {state: value}
    comparisons = predicate["$and"][1]["$expr"]["$and"]
    lower = next(item["$gte"] for item in comparisons if "$gte" in item)
    upper = next(item["$lt"] for item in comparisons if "$lt" in item)
    assert lower[0]["$dateFromString"]["dateString"] == "$" + runtime.field
    assert lower[0]["$dateFromString"]["format"] == "%d/%m/%Y %H:%M:%S"
    assert lower[1] == {"$date": "2026-10-20T12:00:00Z"}
    assert upper[1] == {"$date": "2026-10-21T12:00:00Z"}
    assert invoke.call_count == 1
    fallback.assert_not_called()
