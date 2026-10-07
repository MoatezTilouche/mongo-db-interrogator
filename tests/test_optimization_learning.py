"""Learning stays tied to source patterns and never replays another request's values."""
from copy import deepcopy
from unittest.mock import Mock
import pytest

from agents.validator_agent import ValidatorAgent
from services.optimization_advisor import OptimizationAdvisor
from services.query_performance_memory import QueryPerformanceMemory, optimization_evidence
from services.query_signature import query_signature
from services.query_parameter_binder import template_from_concrete, bind_query_parameters
from services.optimization_intent import effective_question

SCHEMA = {"collections": {"records": {"fields": {"created": ["datetime"], "state": ["string"]}},
                          "details": {"fields": {}}}}


def original_query(limit=10):
    return {"operation": "aggregate", "collection": "records", "pipeline": [
        {"$match": {"state": "current"}},
        {"$lookup": {"from": "details", "localField": "key", "foreignField": "key", "as": "details"}},
        {"$sort": {"score": -1}}, {"$limit": limit}]}


def optimized_query(limit=10):
    return OptimizationAdvisor().apply(original_query(limit), "optimize_aggregation")


def evidence(**changes):
    return {"id": "optimize_aggregation", "attempts": 2, "successes": 2, "failures": 0,
            "success_rate": 1, "safe_successes": 2, "avg_time_reduction": None, **changes}


def test_learning_records_original_to_optimized_signature_and_measured_comparison(monkeypatch):
    collection = Mock()
    memory = QueryPerformanceMemory()
    monkeypatch.setattr(memory, "collection", lambda: collection)
    original, optimized = original_query(), optimized_query()
    before = deepcopy(original)
    signature, _ = query_signature(original)
    memory.record_optimization(original, optimized,
        {"matched_historical_signature": signature, "historical_avg_execution_ms": 100},
        {"ok": True, "execution_time_ms": 40}, {"selected": "optimize_aggregation"})
    query, update = collection.update_one.call_args.args
    assert query == {"signature": signature}
    assert update["$set"]["optimization_outcomes.optimize_aggregation.last_optimized_signature"] == query_signature(optimized)[0]
    assert update["$inc"]["optimization_outcomes.optimize_aggregation.safe_successes"] == 1
    assert update["$inc"]["optimization_outcomes.optimize_aggregation.total_time_reduction"] == .6
    assert original == before
    assert "current" not in str(update)  # No literal filter values or result documents.


@pytest.mark.parametrize("success", [True, False])
def test_learning_without_baseline_does_not_claim_measured_improvement(monkeypatch, success):
    memory = QueryPerformanceMemory()
    collection = Mock()
    monkeypatch.setattr(memory, "collection", lambda: collection)
    memory.record_optimization(original_query(), optimized_query(), {},
                              {"ok": success, "execution_time_ms": 5}, {"selected": "optimize_aggregation"})
    increments = collection.update_one.call_args.args[1]["$inc"]
    assert increments["optimization_outcomes.optimize_aggregation.successes"] == int(success)
    assert increments["optimization_outcomes.optimize_aggregation.failures"] == int(not success)
    assert not any("time_reduction" in key for key in increments)


def test_failed_execution_does_not_increase_successful_choices_or_averages(monkeypatch):
    collection = Mock()
    memory = QueryPerformanceMemory()
    monkeypatch.setattr(memory, "collection", lambda: collection)
    signature, structure = query_signature(original_query())
    memory.update(signature, structure, {"execution_success": False, "was_heavy": True,
        "optimization_suggested": True, "optimization_selected": "optimize_aggregation",
        "execution_time_ms": 40, "returned_rows": 0})
    sets = collection.update_one.call_args.args[1][0]["$set"]
    assert "choices.optimize_aggregation" not in sets
    assert "total_execution_ms" not in sets
    assert sets["failed_execution_count"]["$add"][1] == 1


def test_evidence_ranks_success_over_popularity_and_reports_unknown_measurement():
    profile = {"optimization_outcomes": {
        "limit": {"attempts": 100, "successes": 20, "failures": 80},
        "optimize_aggregation": {"attempts": 2, "successes": 2, "safe_successes": 2},
        "unexpected.key": {"attempts": 1, "successes": 1}}}
    ranked = optimization_evidence(profile)
    assert ranked[0]["id"] == "optimize_aggregation"
    assert ranked[0]["avg_time_reduction"] is None
    assert len(ranked) == 2


def auto_apply(monkeypatch, query=None, items=None, schema=SCHEMA):
    import services.learned_optimization_service as module
    monkeypatch.setattr(module, "QUERY_LEARNED_OPTIMIZATION_ENABLED", True)
    monkeypatch.setattr(module, "QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES", 2)
    validator = Mock(wraps=ValidatorAgent())
    result = module.LearnedOptimizationService().apply(query or original_query(),
        {"optimization_evidence": items if items is not None else [evidence()]}, schema, validator)
    return result, validator


def test_learned_rule_recomputed_with_current_values_and_revalidated(monkeypatch):
    query = original_query(3)
    query["pipeline"][0]["$match"]["state"] = "new_request"
    (updated, metadata), validator = auto_apply(monkeypatch, query)
    assert updated["pipeline"][0] == query["pipeline"][0]
    assert updated["pipeline"][2] == {"$limit": 3}
    assert updated["pipeline"][3] == query["pipeline"][1]
    assert metadata["automatic"] and metadata["scope_preserved"]
    validator.validate.assert_called_once()
    template = template_from_concrete(updated, True)
    assert template["parameters"]["limit"] == 3
    assert bind_query_parameters(template["template"], template["parameters"]) == updated


@pytest.mark.parametrize("item", [
    evidence(successes=1, safe_successes=1), evidence(failures=1),
    evidence(safe_successes=0), evidence(avg_time_reduction=-.1),
    evidence(id="limit"), evidence(id="narrow_date"), evidence(id="summary"), evidence(id="refine"),
])
def test_unproven_or_scope_changing_rules_are_never_automatic(monkeypatch, item):
    (updated, metadata), validator = auto_apply(monkeypatch, items=[item])
    assert updated == original_query() and metadata is None
    validator.validate.assert_not_called()


@pytest.mark.parametrize("stage", [
    {"$unwind": "$details"}, {"$match": {"details.kind": "x"}},
    {"$set": {"snapshot": "$$ROOT"}}, {"$sort": {"details.rank": 1}},
])
def test_automatic_join_deferral_rejects_current_dependency_or_cardinality_changes(monkeypatch, stage):
    query = original_query()
    query["pipeline"].insert(2, stage)
    (updated, metadata), _ = auto_apply(monkeypatch, query)
    assert updated == query and metadata is None


def test_automatic_rule_failure_validation_keeps_original_query(monkeypatch):
    (updated, metadata), validator = auto_apply(monkeypatch, schema={"collections": {"records": {"fields": {}}}})
    assert updated == original_query() and metadata is None
    validator.validate.assert_called_once()


@pytest.mark.parametrize("correlated", [False, True])
def test_join_deferral_does_not_cross_join_that_changes_input_or_captures_root(monkeypatch, correlated):
    query = original_query()
    other = {"from": "details", "localField": "other", "foreignField": "key", "as": "key"}
    if correlated:
        other = {"from": "details", "let": {"document": "$$ROOT"}, "pipeline": [], "as": "key"}
    query["pipeline"].insert(2, {"$lookup": other})
    query["pipeline"][3] = {"$sort": {"key.rank": 1}}
    (updated, metadata), _ = auto_apply(monkeypatch, query)
    assert updated == query and metadata is None


@pytest.mark.parametrize("limit", [False, -1, {}, None])
def test_automatic_rule_requires_a_valid_existing_limit(monkeypatch, limit):
    query = original_query(limit)
    (updated, metadata), validator = auto_apply(monkeypatch, query)
    assert updated == query and metadata is None
    validator.validate.assert_not_called()


def test_query_specific_suggestions_explain_scope_and_use_schema_dates():
    query = original_query()
    result = OptimizationAdvisor().suggest(query, {"signals": [], "plan_evidence": {"collection_scan": True}}, SCHEMA)
    choices = {item["id"]: item for item in result["suggestions"]}
    assert choices["optimize_aggregation"]["changes_result_scope"] is False
    assert choices["narrow_date"]["fields"] == ["created"]
    assert choices["summary"]["changes_result_scope"] is True
    assert "1 $lookup" in result["message"] and "collection scan" in result["message"]
    assert all(item["expected_impact"] and item["reason"] for item in choices.values())
    without_dates = OptimizationAdvisor().suggest(query, {}, {})
    assert "narrow_date" not in {item["id"] for item in without_dates["suggestions"]}


def test_suggestions_rank_measured_success_not_most_chosen_failure():
    cost = {"common_optimization": "refine", "optimization_evidence": [
        evidence(), {"id": "refine", "successes": 30, "success_rate": .3, "avg_time_reduction": -.1}]}
    suggestions = OptimizationAdvisor().suggest(original_query(), cost, SCHEMA)["suggestions"]
    assert suggestions[0]["id"] == "optimize_aggregation"
    assert suggestions[-1]["id"] == "refine"


def test_bounded_find_does_not_offer_redundant_limits():
    query = {"operation": "find", "collection": "records", "limit": 1}
    suggestions = OptimizationAdvisor().suggest(query, {}, SCHEMA)["suggestions"]
    assert not {"limit", "latest_100"} & {item["id"] for item in suggestions}


@pytest.mark.parametrize("choice", ["optimize_aggregation", "run_anyway"])
def test_scope_preserving_choice_keeps_original_intent(choice):
    assert effective_question("Show records", choice, optimized_query()) == "Show records"


def test_refined_intent_accumulates_prior_constraints():
    first = effective_question("Show records", "refine", {}, "Only active records")
    second = effective_question(first, "narrow_date", {}, "During March")
    assert "Only active records" in second and "During March" in second


def test_aggregate_explain_uses_only_winning_plan_and_never_executes(monkeypatch):
    import services.query_cost_guard as module
    monkeypatch.setattr(module, "QUERY_COST_GUARD_ENABLED", True)
    monkeypatch.setattr(module.QueryPerformanceMemory, "match", lambda *args: ({}, 0))
    db = Mock()
    db.command.return_value = {"stages": [{"$cursor": {"queryPlanner": {
        "winningPlan": {"stage": "IXSCAN", "indexName": "state_1"},
        "rejectedPlans": [{"stage": "COLLSCAN"}]}}}]}
    monkeypatch.setattr(module, "get_database", lambda: db)
    result = module.QueryCostGuard().analyze(original_query())
    command = db.command.call_args.args[0]
    assert command["verbosity"] == "queryPlanner"
    assert command["explain"]["aggregate"] == "records"
    assert command["maxTimeMS"] <= 1000
    assert result["plan_evidence"] == {"available": True, "collection_scan": False, "index_names": ["state_1"]}
    db.aggregate.assert_not_called()


def test_generated_query_applies_learned_rule_then_rechecks_cost_before_execution(monkeypatch):
    import orchestrator as module
    import services.learned_optimization_service as learned_module
    from test_lookup_correctness import fresh_pipeline
    monkeypatch.setattr(learned_module, "QUERY_LEARNED_OPTIMIZATION_ENABLED", True)
    monkeypatch.setattr(learned_module, "QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES", 2)
    original = original_query(3)
    original["collection"] = "orders"
    original["pipeline"][1]["$lookup"]["from"] = "customers"
    app = fresh_pipeline(monkeypatch, original)
    app.query_generator.generate.return_value = template_from_concrete(original, True)
    app.validator = Mock(wraps=ValidatorAgent())
    before_cost = {"status": "heavy", "query_signature": query_signature(original)[0],
                   "optimization_evidence": [evidence()]}
    after_cost = {"status": "safe", "query_signature": "optimized", "estimated_rows": 3}
    cost = Mock(side_effect=[before_cost, after_cost])
    monkeypatch.setattr(module.QueryCostGuard, "analyze", cost)
    record = Mock()
    monkeypatch.setattr(module.PerformanceLogger, "record_optimization", record)
    result = app.run("Show top 3 related records")
    assert result["execution_success"] and result["learned_optimization"]["automatic"]
    assert app.validator.validate.call_count == 2 and cost.call_count == 2
    updated = OptimizationAdvisor().apply(original, "optimize_aggregation")
    app.executor.execute.assert_called_once_with(updated)
    assert result["query_parameters"]["limit"] == 3
    assert bind_query_parameters(result["generated_query"]["template"], result["query_parameters"]) == updated
    assert record.call_args.args[0] == original and record.call_args.args[1] == updated
