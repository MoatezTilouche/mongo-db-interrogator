"""Bounded matching and atomic aggregated performance profiles; no result documents."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.query_performance_memory"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "query_performance_memory", _sys.modules[__spec__.name])
import logging
from datetime import datetime, timezone
from agent.config import QUERY_PERFORMANCE_COLLECTION, QUERY_PERFORMANCE_SIMILARITY_THRESHOLD
from database.mongo_client import get_database
from services.query_signature import structural_similarity
logger = logging.getLogger("services.query_performance_memory")
OPTIMIZATION_CHOICES = {"limit", "latest_100", "paginate_100", "summary", "narrow_date", "refine", "optimize_aggregation"}


def optimization_evidence(profile):
    """Expose bounded counters, never stored queries, parameter values or documents."""
    result = []
    for choice, counters in (profile.get("optimization_outcomes") or {}).items():
        if choice not in OPTIMIZATION_CHOICES or not isinstance(counters, dict):
            continue
        attempts = counters.get("attempts", 0)
        if not attempts:
            continue
        successes = counters.get("successes", 0)
        comparisons = counters.get("time_comparisons", 0)
        time_samples = counters.get("execution_time_samples", 0)
        result.append({"id": choice, "attempts": attempts, "successes": successes,
            "failures": counters.get("failures", 0), "success_rate": successes / attempts,
            "safe_successes": counters.get("safe_successes", 0),
            "time_comparisons": comparisons,
            "avg_time_reduction": counters.get("total_time_reduction", 0) / comparisons if comparisons else None,
            "avg_execution_ms": counters.get("total_execution_ms", 0) / time_samples if time_samples else None,
            "last_optimized_signature": counters.get("last_optimized_signature")})
    return sorted(result, key=lambda item: (item["success_rate"],
                  item["avg_time_reduction"] if item["avg_time_reduction"] is not None else -1,
                  item["successes"]), reverse=True)

class QueryPerformanceMemory:
    def collection(self):
        c = get_database()[QUERY_PERFORMANCE_COLLECTION]
        c.create_index("signature", unique=True)
        c.create_index([("collections", 1), ("operation", 1)])
        c.create_index("updated_at")
        return c

    def match(self, signature, structure):
        c = self.collection()
        exact = c.find_one({"signature": signature}, {"_id": 0})
        if exact:
            logger.info("query_performance_exact_match signature=%s", signature)
            return exact, 1.
        candidates = c.find({"collections": {"$in": structure["collections"]}, "operation": structure["operation"]}, {"_id": 0}).sort("updated_at", -1).limit(200)
        ranked = sorted(((structural_similarity(structure, p), p) for p in candidates), key=lambda x: x[0], reverse=True)
        if ranked and ranked[0][0] >= QUERY_PERFORMANCE_SIMILARITY_THRESHOLD:
            logger.info("query_performance_similar_match signature=%s score=%s", signature, ranked[0][0])
            return ranked[0][1], ranked[0][0]
        logger.info("query_performance_no_match signature=%s", signature)
        return {}, 0.

    def update(self, signature, structure, metrics):
        # Update pipeline calculates averages atomically, including concurrent executions.
        selected = metrics.get("optimization_selected")
        succeeded = metrics.get("execution_success", True)
        increments = {"execution_count": 1, "heavy_count": int(metrics["was_heavy"]),
                      "successful_execution_count": int(succeeded), "failed_execution_count": int(not succeeded),
                      "optimization_suggested_count": int(metrics["optimization_suggested"]),
                      "optimization_accepted_count": int(bool(selected) and selected != "run_anyway")}
        for source, target in [("returned_rows", "returned_rows"), ("execution_time_ms", "execution_ms"),
                               ("estimated_rows", "estimated_rows"), ("documents_examined", "documents_examined")]:
            if succeeded and metrics.get(source) is not None:
                increments["total_"+target] = metrics[source]
                increments[target+"_samples"] = 1
        if selected in OPTIMIZATION_CHOICES and succeeded: increments["choices."+selected] = 1
        sets = {key: {"$add": [{"$ifNull": ["$"+key, 0]}, value]} for key, value in increments.items()}
        sets.update({k: {"$literal": v} for k,v in structure.items()})
        sets.update({"signature": signature, "updated_at": datetime.now(timezone.utc),
                     "max_execution_ms": {"$max": [{"$ifNull": ["$max_execution_ms", 0]}, metrics["execution_time_ms"]]}})
        averages = {"heavy_rate": {"$divide": ["$heavy_count", "$execution_count"]},
                    "optimization_acceptance_rate": {"$cond": [{"$gt": ["$optimization_suggested_count", 0]}, {"$divide": ["$optimization_accepted_count", "$optimization_suggested_count"]}, 0]}}
        for target in ("returned_rows", "execution_ms", "estimated_rows", "documents_examined"):
            averages["avg_"+target] = {"$cond": [{"$gt": [{"$ifNull": ["$"+target+"_samples", 0]}, 0]}, {"$divide": ["$total_"+target, "$"+target+"_samples"]}, None]}
        c = self.collection()
        common = {"$reduce": {"input": {"$objectToArray": {"$ifNull": ["$choices", {}]}}, "initialValue": {"k": None, "v": 0},
                  "in": {"$cond": [{"$gt": ["$$this.v", "$$value.v"]}, "$$this", "$$value"]}}}
        c.update_one({"signature": signature}, [{"$set": sets}, {"$set": averages},
            {"$set": {"common_optimization": {"$let": {"vars": {"winner": common}, "in": "$$winner.k"}},
                       "common_optimization_value": metrics.get("optimization_value")}}], upsert=True)
        logger.info("query_performance_updated signature=%s", signature)

    def record_optimization(self, original, optimized, before_cost, execution, optimization):
        from services.query_signature import query_signature
        from services.optimization_advisor import OptimizationAdvisor
        choice = optimization.get("selected")
        if choice not in OPTIMIZATION_CHOICES or original == optimized:
            return
        signature, structure = query_signature(original)
        optimized_signature, _ = query_signature(optimized)
        succeeded = bool(execution.get("ok"))
        safe = False
        if choice == "optimize_aggregation":
            try:
                safe = OptimizationAdvisor().apply(original, choice) == optimized
            except ValueError:
                pass
        prefix = f"optimization_outcomes.{choice}."
        increments = {prefix + "attempts": 1, prefix + "successes": int(succeeded),
                      prefix + "failures": int(not succeeded), prefix + "safe_successes": int(safe and succeeded)}
        elapsed = execution.get("execution_time_ms")
        if succeeded and isinstance(elapsed, (int, float)) and elapsed >= 0:
            increments[prefix + "total_execution_ms"] = elapsed
            increments[prefix + "execution_time_samples"] = 1
            baseline = before_cost.get("historical_avg_execution_ms")
            if (before_cost.get("matched_historical_signature") == signature
                    and isinstance(baseline, (int, float)) and baseline > 0):
                increments[prefix + "time_comparisons"] = 1
                increments[prefix + "total_time_reduction"] = (baseline - elapsed) / baseline
        self.collection().update_one({"signature": signature}, {
            "$setOnInsert": {"signature": signature, **structure},
            "$inc": increments,
            "$set": {"updated_at": datetime.now(timezone.utc),
                     prefix + "last_optimized_signature": optimized_signature}}, upsert=True)
        logger.info("optimization_outcome_recorded original_signature=%s optimized_signature=%s choice=%s success=%s scope_preserved=%s",
                    signature, optimized_signature, choice, succeeded, safe)

class PerformanceLogger:
    def record(self, query, cost, execution, optimization=None):
        from services.query_signature import query_signature
        from agent.config import QUERY_HEAVY_RESULT_THRESHOLD, QUERY_HEAVY_EXECUTION_MS_THRESHOLD
        signature, structure = query_signature(query)
        try:
            QueryPerformanceMemory().update(signature, structure, {
                "execution_success": bool(execution.get("ok")),
                "returned_rows": execution.get("row_count", 0), "execution_time_ms": execution.get("execution_time_ms", 0),
                "estimated_rows": cost.get("estimated_rows"), "documents_examined": execution.get("documents_examined"),
                "was_heavy": execution.get("row_count", 0) > QUERY_HEAVY_RESULT_THRESHOLD or execution.get("execution_time_ms", 0) >= QUERY_HEAVY_EXECUTION_MS_THRESHOLD or cost.get("status") == "heavy",
                "optimization_value": (optimization or {}).get("value"),
                "optimization_suggested": bool(optimization), "optimization_selected": (optimization or {}).get("selected")})
        except Exception as exc:
            logger.warning("query_performance_update_failed error_type=%s", type(exc).__name__)

    def record_optimization(self, original, optimized, before_cost, after_cost, execution, optimization):
        try:
            QueryPerformanceMemory().record_optimization(original, optimized, before_cost, execution, optimization)
        except Exception as exc:
            logger.warning("optimization_outcome_record_failed error_type=%s", type(exc).__name__)
