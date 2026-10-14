"""Recompute only successful, scope-preserving transformations for current data."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.learned_optimization_service"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "learned_optimization_service", _sys.modules[__spec__.name])
import logging
from agent.config import QUERY_LEARNED_OPTIMIZATION_ENABLED, QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES
from agent.progress import stage
from agent.telemetry import current_request_id
from agents.validator_agent import ValidatorAgent, QueryValidationError
from services.optimization_advisor import OptimizationAdvisor
from services.aggregation_validation import normalize_lookups, normalization_context

logger = logging.getLogger("services.learned_optimization_service")


class LearnedOptimizationService:
    def apply(self, query, cost, schema_spec, validator=None):
        if not QUERY_LEARNED_OPTIMIZATION_ENABLED:
            return query, None
        for evidence in cost.get("optimization_evidence", []):
            if evidence.get("id") != "optimize_aggregation":
                continue
            successes = evidence.get("successes", 0)
            reduction = evidence.get("avg_time_reduction")
            if (successes < QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES
                    or evidence.get("failures", 0) or evidence.get("safe_successes", 0) != successes
                    or (reduction is not None and reduction <= 0)):
                continue
            try:
                updated = OptimizationAdvisor().apply(query, "optimize_aggregation")
                updated = stage("AggregationNormalizer", normalize_lookups, updated)
                with normalization_context():
                    updated = stage("ValidatorAgent", (validator or ValidatorAgent()).validate, updated, schema_spec)
            except (QueryValidationError, ValueError):
                logger.info("learned_optimization_skipped request_id=%s reason=transformation_not_applicable",
                            current_request_id())
                continue
            logger.info("learned_optimization_applied request_id=%s choice=optimize_aggregation prior_successes=%s",
                        current_request_id(), successes)
            return updated, {"selected": "optimize_aggregation", "automatic": True,
                             "scope_preserved": True, "prior_successes": successes,
                             "matched_signature": cost.get("matched_historical_signature"),
                             "avg_time_reduction": reduction}
        return query, None
