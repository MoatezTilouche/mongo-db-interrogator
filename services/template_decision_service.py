"""Durable, request-bound user decisions with atomic execution claims."""
import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4
from fastapi import HTTPException
from agent.config import DEFAULT_USER_ID
from agent.progress import capture_workflow, emit, stage
from agent.telemetry import request_context, capture_usage, totals
from database.mongo_client import get_database
from database.query_template_repository import QueryTemplateRepository, template_validation_schema
from services.semantic_template_cache import SemanticTemplateCache
from services.template_parameter_extractor import extract_parameters
from services.query_parameter_binder import bind_query_parameters
from agents.validator_agent import ValidatorAgent, QueryValidationError
from agents.executor_agent import ExecutorAgent
from agents.results_summary_agent import ResultsSummaryAgent
from services.query_cost_guard import QueryCostGuard
from services.query_performance_memory import PerformanceLogger
from services.optimization_advisor import OptimizationAdvisor

logger = logging.getLogger(__name__)


class TemplateDecisionService:
    def collection(self):
        collection = get_database()["_system_template_decisions"]
        collection.create_index("expires_at", expireAfterSeconds=0)
        return collection

    def pause(self, request_id, question, result):
        self.collection().insert_one({"_id": request_id, "user_id": DEFAULT_USER_ID, "state": "pending",
            "question": question, "template_id": result["similarity"]["template_id"],
            "cache": result["semantic_template_cache"], "expires_at": datetime.now(timezone.utc) + timedelta(hours=1)})

    def decide(self, request_id, payload):
        collection = self.collection()
        claim = uuid4().hex
        pending = collection.find_one_and_update({"_id": request_id, "user_id": DEFAULT_USER_ID,
            "template_id": payload.template_id, "state": "pending", "expires_at": {"$gt": datetime.now(timezone.utc)}},
            {"$set": {"state": "executing", "claim": claim}})
        if not pending: raise HTTPException(409, "Template decision is expired, unavailable, or already handled")
        started = perf_counter()
        attempted_execution = False
        cache = {**pending["cache"], "user_decision": payload.action}
        from services.query_history_service import QueryHistoryService
        try:
            if payload.action == "regenerate":
                logger.info("semantic_template_cache_user_regenerate request_id=%s", request_id)
                attempted_execution = True
                with request_context(request_id):
                    emit("SemanticTemplateCache", "completed", message="User chose regeneration")
                    result = QueryHistoryService().run(pending["question"], request_id, skip_cache=True, cache_metadata=cache)
            else:
                logger.info("semantic_template_cache_user_reuse request_id=%s", request_id)
                repository = QueryTemplateRepository()
                document = repository.get(payload.template_id)
                from services.schema_bootstrap_service import schema_bootstrap_service
                if not document or not SemanticTemplateCache.compatible(document, pending["question"],
                        set(get_database().list_collection_names()), schema_bootstrap_service.status().get("schema_version")):
                    raise HTTPException(409, "Template is no longer compatible. Choose Regenerate.")
                with request_context(request_id), capture_usage() as usage, capture_workflow() as trace:
                    emit("TemplateDecision", "completed", message="Template selected")
                    extraction_started = perf_counter()
                    cache["template_parameter_extraction_started"] = True
                    logger.info("template_parameter_extraction_started request_id=%s", request_id)
                    try:
                        parameters, method = stage("TemplateParameterExtractor", extract_parameters, pending["question"], document)
                    except Exception:
                        cache["template_parameter_extraction_failed"] = True
                        cache["parameter_extraction_ms"] = round((perf_counter() - extraction_started) * 1000, 2)
                        logger.warning("template_parameter_extraction_failed request_id=%s", request_id)
                        raise HTTPException(422, "Could not safely extract the new parameters. Choose Regenerate.") from None
                    extraction_ms = round((perf_counter() - extraction_started) * 1000, 2)
                    cache["parameter_extraction_ms"] = extraction_ms
                    cache["template_parameter_extraction_completed"] = True
                    logger.info("template_parameter_extraction_completed request_id=%s parameter_extraction_ms=%.2f method=%s", request_id, extraction_ms, method)
                    generated = {"template": document["template"], "parameters": parameters, "semantic_intent": document["semantic_text"]}
                    query = stage("QueryParameterBinder", bind_query_parameters, document["template"], parameters)
                    try:
                        schema_spec = template_validation_schema(document["schema_spec"])
                        from services.user_preference_service import UserPreferenceService, apply_preferences_to_query
                        preference_service = UserPreferenceService(user_id=DEFAULT_USER_ID)
                        preference_catalog = {"collections": schema_spec.get("collections", {})}
                        preferences = preference_service.retrieve_for_query(
                            pending["question"], preference_catalog, request_id,
                            selected_collections={query.get("collection")} if query.get("collection") else None,
                        )
                        applied_preferences = apply_preferences_to_query(
                            query, preferences, DEFAULT_USER_ID, pending["question"],
                        )
                        query = stage("ValidatorAgent", ValidatorAgent().validate, query, schema_spec)
                    except QueryValidationError:
                        cache["template_reuse_validation_failed"] = True
                        logger.warning("template_reuse_validation_failed request_id=%s", request_id)
                        repository.record_failure(payload.template_id)
                        raise HTTPException(422, "Template failed validation. Choose Regenerate.") from None
                    cost = stage("QueryCostGuard", QueryCostGuard().analyze, query)
                    result = {"request_id": request_id, "question": pending["question"],
                        "semantic_template_cache": cache, "generated_query": generated, "validated_query": query,
                        "schema_spec": schema_spec, "query_parameters": parameters,
                        "query_template_id": payload.template_id, "cost_analysis": cost, "query_signature": cost["query_signature"],
                        "applied_preferences": applied_preferences, "applied_knowledge": [],
                        "execution_path": {"route": "template_reuse", "agents_called": [], "summary_skipped": True, "repair_used": False}}
                    if cost["status"] == "heavy":
                        result.update({"status": "optimization_required", **stage("OptimizationAdvisor", OptimizationAdvisor().suggest, query, cost)})
                        from services.query_optimization_service import QueryOptimizationService
                        QueryOptimizationService().pause(result, pending["question"])
                    else:
                        attempted_execution = True
                        execution = stage("ExecutorAgent", ExecutorAgent().execute, query)
                        PerformanceLogger().record(query, cost, execution)
                        result.update({"status": "completed", "execution_success": execution["ok"],
                            "execution_result": execution.get("all_rows", []), "execution_error": execution.get("error"),
                            "execution_error_code": execution.get("error_code"), "execution_metadata": {
                                "row_count": execution.get("row_count", 0), "execution_time_ms": execution.get("execution_time_ms", 0)}})
                        if execution["ok"]:
                            logger.info("template_reuse_execution_success request_id=%s", request_id)
                            if query["operation"] == "aggregate" and execution.get("row_count"):
                                summary = stage("ResultsSummaryAgent", ResultsSummaryAgent().summarize, pending["question"], execution)
                                result["execution_path"]["agents_called"] = ["summary"]
                                result["execution_path"]["summary_skipped"] = False
                            else:
                                rows = execution.get("all_rows", [])
                                summary = {"summary_text": f"Count: {rows[0]['count']}." if query["operation"] == "count_documents" and rows else f"Found {len(rows)} results."}
                                emit("ResultsSummaryAgent", "skipped", message="Deterministic summary used")
                            result["results_summary_agent_output"] = summary
                            result["final_response"] = summary.get("summary_text")
                        else:
                            repository.record_failure(payload.template_id)
                    result["telemetry"] = {**totals(usage), "parameter_extraction_ms": extraction_ms,
                        "semantic_cache_lookup_ms": cache.get("semantic_cache_lookup_ms", 0), "parameter_extraction_method": method,
                        "semantic_template_cache_user_reuse": True,
                        "template_parameter_extraction_started": True, "template_parameter_extraction_completed": True,
                        "template_reuse_execution_success": bool(result.get("execution_success"))}
                    result["execution_trace"] = trace
                QueryHistoryService().save_result(request_id, pending["question"], result,
                    (perf_counter() - started) * 1000, update_existing=True)
            collection.update_one({"_id": request_id, "claim": claim}, {"$set": {"state": "completed"}})
            return result
        except Exception:
            collection.update_one({"_id": request_id, "claim": claim}, {"$set": {"state": "completed" if attempted_execution else "pending"}})
            # Preserve the decision even if extraction/validation fails.
            try:
                from agent.config import QUERY_HISTORY_COLLECTION
                get_database()[QUERY_HISTORY_COLLECTION].update_one({"_id": request_id}, {"$set": {"semantic_template_cache": cache}})
            except Exception: pass
            raise
