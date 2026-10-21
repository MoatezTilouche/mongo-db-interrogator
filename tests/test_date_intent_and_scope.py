"""Elapsed date windows and document aliases remain safe without model calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agents.validator_agent import ValidatorAgent
from services.aggregation_validation import QueryValidationError
from services.query.dates.resolver import resolve_question_dates
from services.query.profile_scope import Scope, advance_scope


C, T, S = "synthetic_events", "recordedAt", "state"


@pytest.fixture
def profiles(monkeypatch):
    for key, value in {"PLAN_MODE": "off", "DATE_NORMALIZATION_ENABLED": "false",
                       "PROFILE_VALIDATION_MODE": "strict", "DATE_EXPRESSIONS_FILE": ""}.items():
        monkeypatch.setenv(key, value)
    return {C: {"collection": C, "fields": {
        T: {"kind": "date", "types": {"string": 1},
            "date_format": "%d/%m/%Y %H:%M:%S", "sortable_as_string": False},
        S: {"kind": "text", "types": {"string": 1}}}}}


@pytest.mark.parametrize("phrase,hours", [("last day", 24), ("past day", 24),
                                          ("last 24 hours", 24), ("past 6 hours", 6)])
def test_elapsed_windows_are_bounded_by_request_time(phrase, hours, monkeypatch):
    monkeypatch.setenv("DATE_EXPRESSIONS_FILE", "")
    now = datetime(2026, 10, 21, 15, 20, 30, tzinfo=timezone.utc)
    [constraint] = resolve_question_dates("find events in the " + phrase, now, timezone.utc)
    assert constraint["start"] == now - timedelta(hours=hours)
    assert constraint["end_exclusive"] == now


@pytest.mark.parametrize("day", [datetime(2026, 3, 29, 12, tzinfo=timezone.utc),
                                 datetime(2026, 10, 25, 12, tzinfo=timezone.utc)])
def test_elapsed_day_is_24_hours_across_dst(day, monkeypatch):
    monkeypatch.setenv("DATE_EXPRESSIONS_FILE", "")
    [constraint] = resolve_question_dates("last day", day, ZoneInfo("Europe/Paris"))
    assert constraint["end_exclusive"].astimezone(timezone.utc) == day
    assert constraint["start"].astimezone(timezone.utc) == day - timedelta(hours=24)


def test_yesterday_remains_a_calendar_day(monkeypatch):
    monkeypatch.setenv("DATE_EXPRESSIONS_FILE", "")
    now = datetime(2026, 10, 21, 15, 20, tzinfo=timezone.utc)
    [constraint] = resolve_question_dates("yesterday", now, timezone.utc)
    assert constraint["start"] == datetime(2026, 10, 20, tzinfo=timezone.utc)
    assert constraint["end_exclusive"] == datetime(2026, 10, 21, tzinfo=timezone.utc)


@pytest.mark.parametrize("operation", ["find", "aggregate"])
@pytest.mark.parametrize("mode", ["off", "warn", "strict"])
def test_direct_validator_rejects_raw_date_sort_even_with_policies_disabled(profiles, monkeypatch, operation, mode):
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", mode)
    query = {"collection": C, "operation": operation}
    query.update({"pipeline": [{"$sort": {T: -1}}]} if operation == "aggregate"
                 else {"filter": {}, "sort": [[T, -1]]})
    schema = {"collections": {C: {"fields": {T: ["string"], S: ["string"]}}}}
    with pytest.raises(QueryValidationError) as exc:
        ValidatorAgent().validate(query, schema, profiles.get)
    assert exc.value.to_dict()["code"] == "date_string_ordering"


@pytest.mark.parametrize("operator", ["$replaceRoot", "$replaceWith"])
def test_document_wrap_and_restore_preserve_profile_origins(profiles, operator):
    scope = Scope(C, profiles.get)
    wrap = {"document": "$$ROOT", "computed": {"$literal": 1}}
    advance_scope(scope, operator, {"newRoot": wrap} if operator == "$replaceRoot" else wrap)
    assert scope.spec("document." + T)["sortable_as_string"] is False
    assert scope.knows("document._id")
    advance_scope(scope, operator, {"newRoot": "$document"} if operator == "$replaceRoot" else "$document")
    assert scope.spec(T) == profiles[C]["fields"][T]
    assert scope.origins[T] == (C, T)
    assert not scope.knows("computed")


def test_alias_sort_cannot_hide_raw_date_ordering(profiles):
    query = {"collection": C, "operation": "aggregate", "pipeline": [
        {"$replaceRoot": {"newRoot": {"document": "$$ROOT"}}},
        {"$sort": {"document." + T: -1}},
        {"$replaceRoot": {"newRoot": "$document"}}]}
    schema = {"collections": {C: {"fields": {T: ["string"], S: ["string"]}}}}
    with pytest.raises(QueryValidationError, match="sortable_as_string=false"):
        ValidatorAgent().validate(query, schema, profiles.get)


def test_unknown_field_is_still_rejected_after_restore(profiles):
    scope = Scope(C, profiles.get)
    before = deepcopy(scope.fields)
    advance_scope(scope, "$replaceWith", {"row": "$$ROOT"})
    advance_scope(scope, "$replaceWith", "$row")
    assert {key: value for key, value in scope.fields.items() if key != "_id"} == before
    assert not scope.knows("inventedField")
