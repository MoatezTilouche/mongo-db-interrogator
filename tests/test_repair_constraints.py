"""Offline regression tests for repair outputs that silently drop constraints."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.query_generator_agent import QueryGeneratorAgent
from services.aggregation_validation import QueryValidationError
from services.query.repair_constraints import assert_repair_preserves_constraints
from services.query_parameter_binder import bind_query_parameters


def query(predicate=None, **extra):
    return {"operation": "find", "collection": "records", "filter": predicate or {}, **extra}


def aggregation(*stages):
    return {"operation": "aggregate", "collection": "records", "pipeline": list(stages)}


def refuses(previous, repaired):
    before = deepcopy((previous, repaired))
    with pytest.raises(QueryValidationError) as caught:
        assert_repair_preserves_constraints(previous, repaired)
    assert caught.value.to_dict()["code"] == "repair_changed_constraints"
    assert not caught.value.repairable
    assert (previous, repaired) == before


@pytest.mark.parametrize("repaired", [{}, {"state": "ready"}, {"state": {"$ne": "blocked"}},
                                      {"$or": [{"state": "blocked"}, {"other": 3}]}])
def test_dropped_or_changed_equality_is_rejected(repaired):
    refuses(query({"state": "blocked"}), query(repaired))


@pytest.mark.parametrize("predicate", [
    {"state": "blocked", "other": 4},
    {"$and": [{"state": {"$eq": "blocked"}}, {"other": 4}]},
    {"$or": [{"state": "blocked", "other": 4}, {"state": "blocked", "other": 5}]},
    {"$expr": {"$eq": ["$state", "blocked"]}},
    {"$expr": {"$eq": ["blocked", "$state"]}},
])
def test_conjunction_and_equivalent_expression_rewrites_preserve_equality(predicate):
    assert_repair_preserves_constraints(query({"state": "blocked"}), query(predicate))


def test_or_cannot_escape_old_required_filter():
    old = query({"state": "blocked", "$or": [{"region": "north"}, {"region": "south"}]})
    assert_repair_preserves_constraints(old, query({"state": "blocked", "region": "north"}))
    refuses(old, query({"$or": [{"state": "blocked", "region": "north"}, {"region": "south"}]}))


@pytest.mark.parametrize("old,new,accepted", [
    ({"$gte": 10}, {"$gt": 10}, True), ({"$gt": 10}, {"$gte": 10}, False),
    ({"$gte": 10}, {"$gte": 11}, True), ({"$gte": 10}, {"$gte": 9}, False),
    ({"$lt": 20}, {"$lte": 20}, False), ({"$lte": 20}, {"$lt": 20}, True),
    ({"$gte": 10, "$lt": 20}, 15, True), ({"$gte": 10, "$lt": 20}, 20, False),
    ({"$in": ["a", "b"]}, {"$in": ["b"]}, True),
    ({"$in": ["a", "b"]}, {"$in": ["b", "c"]}, False),
    ({"$eq": None}, {"$exists": False}, False), ({"$eq": 1}, True, False),
])
def test_range_and_membership_proofs_are_conservative(old, new, accepted):
    if accepted:
        assert_repair_preserves_constraints(query({"value": old}), query({"value": new}))
    else:
        refuses(query({"value": old}), query({"value": new}))


def test_reported_repair_that_deleted_status_and_date_filters_is_refused():
    previous = aggregation({"$match": {"state": "blocked", "created": {"$gte": "2026-10-20T00:00:00Z"}}},
                           {"$sort": {"created": -1}}, {"$limit": 5})
    refuses(previous, aggregation({"$sort": {"created": -1}}, {"$limit": 5}))


def test_date_parser_rewrite_preserves_same_iso_bound_and_status():
    previous = query({"state": "blocked", "created": {"$gte": "2026-10-20T00:00:00Z",
                                                            "$lt": "2026-10-21T00:00:00Z"}})
    parser = {"$dateFromString": {"dateString": "$created", "format": "%d/%m/%Y %H:%M:%S", "onError": None}}
    predicate = {"state": "blocked", "$expr": {"$and": [
        {"$gte": [parser, datetime(2026, 10, 20, tzinfo=timezone.utc)]},
        {"$lt": [parser, datetime(2026, 10, 21, tzinfo=timezone.utc)]}]}}
    assert_repair_preserves_constraints(previous, aggregation({"$match": predicate}))
    del predicate["$expr"]["$and"][1]
    refuses(previous, aggregation({"$match": predicate}))


def test_unzoned_boundary_cannot_prove_equivalent_to_utc():
    parser = {"$dateFromString": {"dateString": "$created"}}
    refuses(query({"created": {"$gte": "2026-10-20T00:00:00"}}),
            aggregation({"$match": {"$expr": {"$gte": [parser, {"$date": "2026-10-20T00:00:00Z"}]}}}))


def test_native_date_literal_can_be_rendered_as_equivalent_iso_json():
    previous = query({"created": {"$gte": datetime(2026, 10, 20, tzinfo=timezone.utc)}})
    assert_repair_preserves_constraints(previous, query({"created": {"$gte": "2026-10-20T00:00:00Z"}}))


@pytest.mark.parametrize("change", [{"format": "%m/%d/%Y"}, {"timezone": "Europe/Paris"},
                                    {"onError": {"$date": "2026-10-21T00:00:00Z"}}])
def test_existing_date_parser_format_timezone_and_null_behavior_cannot_change(change):
    parser = {"dateString": "$created", "format": "%d/%m/%Y", "timezone": "UTC", "onError": None, "onNull": None}
    predicate = {"$expr": {"$gte": [{"$dateFromString": parser}, {"$date": "2026-10-20T00:00:00Z"}]}}
    changed = deepcopy(predicate)
    changed["$expr"]["$gte"][0]["$dateFromString"].update(change)
    refuses(query(predicate), query(changed))


def test_alias_cannot_retroactively_reinterpret_an_earlier_filter():
    helper = {"$addFields": {"helper": {"$dateFromString": {"dateString": "$created"}}}}
    bound = {"$gte": "2026-10-20T00:00:00Z"}
    refuses(query({"created": bound}), aggregation({"$match": {"helper": bound}}, helper, {"$match": {"other": 1}}))


def test_equal_python_boolean_and_number_are_not_equal_query_literals():
    refuses(query({"value": 1}), query({"value": True}))


def test_limit_and_group_are_predicate_scope_barriers():
    previous = aggregation({"$match": {"state": "blocked"}}, {"$limit": 5})
    refuses(previous, aggregation({"$limit": 5}, {"$match": {"state": "blocked"}}))
    group = {"$group": {"_id": "$state", "total": {"$sum": 1}}}
    refuses(aggregation({"$match": {"total": {"$gt": 2}}}, group),
            aggregation(group, {"$match": {"total": {"$gt": 2}}}))


def test_foreign_and_facet_filters_cannot_disappear():
    lookup = {"$lookup": {"from": "related", "as": "joined", "pipeline": [{"$match": {"active": True}}]}}
    refuses(aggregation(lookup), aggregation({"$lookup": {"from": "related", "as": "joined", "pipeline": []}}))
    refuses(aggregation({"$facet": {"left": [{"$match": {"active": True}}]}}),
            aggregation({"$facet": {"left": []}}))


def test_union_cannot_add_unfiltered_records_or_drop_its_filters():
    refuses(aggregation({"$match": {"state": "blocked"}}),
            aggregation({"$match": {"state": "blocked"}}, {"$unionWith": "related"}))
    refuses(aggregation({"$unionWith": {"coll": "related", "pipeline": [{"$match": {"active": True}}]}}),
            aggregation({"$unionWith": "related"}))


@pytest.mark.parametrize("old,new", [(5, None), (5, 10)])
def test_result_limit_cannot_disappear_or_increase(old, new):
    refuses(query(limit=old), query(limit=new))
    refuses(aggregation({"$limit": old}), aggregation(*([{"$limit": new}] if new else [])))


def test_unknown_expression_and_negation_are_preserved_as_opaque_conditions():
    predicate = {"$nor": [{"state": "blocked"}], "$expr": {"$regexMatch": {"input": "$name", "regex": "x"}}}
    assert_repair_preserves_constraints(query(predicate), query({"$and": [predicate, {"other": 1}]}))
    refuses(query(predicate), query({"$nor": [{"state": "blocked"}]}))


def test_actual_agent_binds_parameter_values_before_guard_and_calls_model_once(monkeypatch):
    output = {"template": {"operation": "find", "collection": "records", "filter": {"state": {"$param": "state"}}},
              "parameters": {"state": "ready"}}
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(output)))
    monkeypatch.setattr("agents.query_generator_agent.llm_router.invoke", invoke)
    schema = {"collections": {"records": {"fields": {"state": ["string"]}}}}
    with pytest.raises(QueryValidationError) as caught:
        QueryGeneratorAgent().repair("Show blocked records", schema, query({"state": "blocked"}), "Bad projection")
    assert caught.value.to_dict()["code"] == "repair_changed_constraints"
    assert not caught.value.repairable
    invoke.assert_called_once()
    output["parameters"]["state"] = "blocked"
    invoke.return_value.content = json.dumps(output)
    result = QueryGeneratorAgent().repair("Show blocked records", schema, query({"state": "blocked"}), "Bad projection")
    assert bind_query_parameters(result["template"], result["parameters"])["filter"] == {"state": "blocked"}


def test_collection_and_operation_changes_cannot_hide_constraint_loss():
    refuses(query(), {**query(), "collection": "unrelated"})
    refuses(query(), {**query(), "operation": "count_documents"})
