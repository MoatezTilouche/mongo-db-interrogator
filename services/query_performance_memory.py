"""Bounded matching and atomic aggregated performance profiles; no result documents."""
import logging
from datetime import datetime, timezone
from agent.config import QUERY_PERFORMANCE_COLLECTION, QUERY_PERFORMANCE_SIMILARITY_THRESHOLD
from database.mongo_client import get_database
from services.query_signature import structural_similarity
logger = logging.getLogger(__name__)

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
        increments = {"execution_count": 1, "heavy_count": int(metrics["was_heavy"]),
                      "optimization_suggested_count": int(metrics["optimization_suggested"]),
                      "optimization_accepted_count": int(bool(selected) and selected != "run_anyway")}
        for source, target in [("returned_rows", "returned_rows"), ("execution_time_ms", "execution_ms"),
                               ("estimated_rows", "estimated_rows"), ("documents_examined", "documents_examined")]:
            if metrics.get(source) is not None:
                increments["total_"+target] = metrics[source]
                increments[target+"_samples"] = 1
        if selected: increments["choices."+selected] = 1
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

class PerformanceLogger:
    def record(self, query, cost, execution, optimization=None):
        from services.query_signature import query_signature
        from agent.config import QUERY_HEAVY_RESULT_THRESHOLD, QUERY_HEAVY_EXECUTION_MS_THRESHOLD
        signature, structure = query_signature(query)
        try:
            QueryPerformanceMemory().update(signature, structure, {
                "returned_rows": execution.get("row_count", 0), "execution_time_ms": execution.get("execution_time_ms", 0),
                "estimated_rows": cost.get("estimated_rows"), "documents_examined": execution.get("documents_examined"),
                "was_heavy": execution.get("row_count", 0) > QUERY_HEAVY_RESULT_THRESHOLD or execution.get("execution_time_ms", 0) >= QUERY_HEAVY_EXECUTION_MS_THRESHOLD or cost.get("status") == "heavy",
                "optimization_value": (optimization or {}).get("value"),
                "optimization_suggested": bool(optimization), "optimization_selected": (optimization or {}).get("selected")})
        except Exception as exc:
            logger.warning("query_performance_update_failed error_type=%s", type(exc).__name__)
