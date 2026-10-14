"""Deterministic pre-execution guard. Estimation never executes an aggregation."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.query_cost_guard"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "query_cost_guard", _sys.modules[__spec__.name])
import logging
from datetime import datetime
from agent.tracing import traceable
from agent.config import (QUERY_COST_GUARD_ENABLED, QUERY_HEAVY_RESULT_THRESHOLD,
    QUERY_HEAVY_EXECUTION_MS_THRESHOLD, QUERY_HEAVY_RISK_THRESHOLD, MONGO_QUERY_TIMEOUT_MS)
from database.mongo_client import get_database
from database.value_codec import decode_special_values
from services.query_signature import query_signature
from services.query_performance_memory import QueryPerformanceMemory, optimization_evidence
logger = logging.getLogger("services.query_cost_guard")

class QueryCostGuard:
    @traceable(name="QueryCostGuard", run_type="chain")
    def analyze(self, query):
        signature, structure = query_signature(query)
        logger.info("query_signature_created signature=%s", signature)
        profile, similarity = {}, 0.
        try: profile, similarity = QueryPerformanceMemory().match(signature, structure)
        except Exception as exc: logger.warning("query_performance_lookup_failed error_type=%s", type(exc).__name__)
        signals, risk, estimated = [], 0., None
        plan_evidence = {"available": False, "collection_scan": False, "index_names": []}
        def signal(name, weight):
            nonlocal risk
            signals.append(name); risk += weight
        if not structure["has_limit"] and query["operation"] in {"find", "aggregate", "distinct"}: signal("no_limit", .25)
        limit = query.get("limit")
        if limit and limit > QUERY_HEAVY_RESULT_THRESHOLD: signal("large_limit", .7)
        if QUERY_COST_GUARD_ENABLED and query["operation"] == "find":
            logger.info("query_cost_estimation_started signature=%s", signature)
            try:
                estimated = get_database()[query["collection"]].count_documents(
                    decode_special_values(query.get("filter", {})), limit=QUERY_HEAVY_RESULT_THRESHOLD+1,
                    maxTimeMS=min(1000, MONGO_QUERY_TIMEOUT_MS))
                output = max(0, estimated-int(query.get("skip", 0)))
                if limit: output = min(output, limit)
                if output > QUERY_HEAVY_RESULT_THRESHOLD: signal("large_estimated_result", .75)
            except Exception:
                signal("estimation_unavailable", .45)
            logger.info("query_cost_estimation_completed signature=%s estimated_rows=%s", signature, estimated)
        # Planner-only explain inspects index use without executing the query.
        if QUERY_COST_GUARD_ENABLED and query["operation"] in {"find", "aggregate"}:
            try:
                if query["operation"] == "find":
                    command = {"find": query["collection"], "filter": decode_special_values(query.get("filter", {}))}
                    if query.get("sort"): command["sort"] = dict(query["sort"])
                else:
                    command = {"aggregate": query["collection"], "pipeline": decode_special_values(query.get("pipeline", [])),
                               "cursor": {}, "allowDiskUse": False}
                plan = get_database().command({"explain": command, "verbosity": "queryPlanner", "maxTimeMS": min(1000, MONGO_QUERY_TIMEOUT_MS)})
                stages_found, indexes = set(), set()
                def winning_plans(v):
                    if isinstance(v, dict):
                        if "winningPlan" in v:
                            inspect_plan(v["winningPlan"])
                        for key, child in v.items():
                            if key not in {"winningPlan", "rejectedPlans", "command"}:
                                winning_plans(child)
                    elif isinstance(v, list):
                        for child in v: winning_plans(child)
                def inspect_plan(v):
                    if isinstance(v, dict):
                        if isinstance(v.get("stage"), str): stages_found.add(v["stage"])
                        if isinstance(v.get("indexName"), str): indexes.add(v["indexName"])
                        for child in v.values(): inspect_plan(child)
                    elif isinstance(v, list):
                        for child in v: inspect_plan(child)
                winning_plans(plan)
                plan_evidence = {"available": bool(stages_found), "collection_scan": "COLLSCAN" in stages_found,
                                 "index_names": sorted(indexes)[:10]}
                if plan_evidence["collection_scan"]:
                    if estimated is None or estimated > QUERY_HEAVY_RESULT_THRESHOLD:
                        signal("collection_scan", .45)
            except Exception:
                pass  # Missing explain permission is not a query failure.
        def inspect(value, depth=0):
            if depth > 8: return True
            if isinstance(value, dict):
                if "$gte" in value and ("$lte" in value or "$lt" in value):
                    try:
                        def date(v):
                            if isinstance(v, dict): v = v.get("$date")
                            return datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else v
                        if (date(value.get("$lte", value.get("$lt")))-date(value["$gte"])).days > 90:
                            signal("wide_date_range", .15)
                    except (TypeError, ValueError, AttributeError): pass
                return any(inspect(child, depth+1) for child in value.values())
            if isinstance(value, list): return any(inspect(child, depth+1) for child in value)
            return False
        if inspect(query): signal("deep_query", .2)
        stages = structure["pipeline_stages"]
        if "$lookup" in stages: signal("expensive_lookup", .35*stages.count("$lookup"))
        if "$unwind" in stages: signal("expensive_unwind", .25*stages.count("$unwind"))
        if "$group" in stages or "$sortByCount" in stages: signal("aggregation_group", .25)
        if "$sort" in stages or query.get("sort"):
            # Output limits do not bound upstream sort work.
            if not structure["filter_fields"] and (query["operation"] != "find" or estimated is None or estimated > QUERY_HEAVY_RESULT_THRESHOLD): signal("unrestricted_sort", .45)
        if "$facet" in stages: signal("aggregation_facet", .45)
        if len(stages) > 10: signal("deep_pipeline", .25)
        if query["operation"] == "aggregate" and not structure["filter_fields"]: signal("unrestricted_aggregation", .25)
        if query["operation"] in {"count_documents", "distinct"} and not structure["filter_fields"]: signal("unrestricted_scan", .45)
        if profile.get("execution_count", 0) >= 2:
            if (profile.get("heavy_rate", 0) >= .5 or (profile.get("avg_execution_ms") or 0) >= QUERY_HEAVY_EXECUTION_MS_THRESHOLD or (profile.get("avg_returned_rows") or 0) > QUERY_HEAVY_RESULT_THRESHOLD):
                # Current bounded evidence takes precedence over broad historical profiles.
                fresh_small = estimated is not None and estimated <= QUERY_HEAVY_RESULT_THRESHOLD
                bounded_find = query["operation"] == "find" and limit and limit <= QUERY_HEAVY_RESULT_THRESHOLD
                weight = .10 if fresh_small or bounded_find else .35
                signal("historically_heavy_pattern", weight*similarity)
        risk = round(min(1., risk), 3)
        status = "heavy" if QUERY_COST_GUARD_ENABLED and risk >= QUERY_HEAVY_RISK_THRESHOLD else "safe"
        logger.info("query_cost_guard_%s signature=%s risk_score=%s", status, signature, risk)
        return {"status": status, "query_signature": signature, "risk_score": risk, "signals": signals,
                "reason": signals[-1] if signals else None, "estimated_rows": estimated,
                "estimate_is_lower_bound": estimated == QUERY_HEAVY_RESULT_THRESHOLD+1,
                "historical_avg_rows": profile.get("avg_returned_rows"), "historical_heavy_rate": profile.get("heavy_rate"),
                "historical_avg_execution_ms": profile.get("avg_execution_ms"),
                "plan_evidence": plan_evidence, "optimization_evidence": optimization_evidence(profile),
                "matched_historical_signature": profile.get("signature"), "similarity_score": similarity,
                "common_optimization": max(profile.get("choices", {}), key=profile.get("choices", {}).get) if profile.get("choices") else None}
