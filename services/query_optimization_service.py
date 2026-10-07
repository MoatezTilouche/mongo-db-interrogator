"""Durable authoritative continuation state with atomic execution claims."""
import logging
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from copy import deepcopy
from fastapi import HTTPException
from agent.config import DEFAULT_USER_ID, QUERY_HISTORY_COLLECTION, QUERY_MAX_RETURN_ROWS
from agent.telemetry import request_context
from agent.progress import stage, workflow_listener, emit
import re
from agent.config import QUERY_MAX_OPTIMIZATION_ITERATIONS
from agents.query_generator_agent import QueryGeneratorAgent
from services.query_parameter_binder import bind_query_parameters, template_from_concrete, reconcile_template, parameterize_filter_values
from services.query_cost_guard import QueryCostGuard
from database.mongo_client import get_database
from database.storage_codec import encode_record, decode_record
from database.value_codec import to_json_safe
from agents.executor_agent import ExecutorAgent
from agents.validator_agent import ValidatorAgent
from agents.results_summary_agent import ResultsSummaryAgent
from services.optimization_advisor import OptimizationAdvisor
from services.query_performance_memory import PerformanceLogger
logger = logging.getLogger(__name__)

def is_new_intent(refinement):
    return bool(re.search(r"\b(forget (?:this|the) request|actually (?:show|calculate|find)|instead (?:show|calculate|find)|instead of (?:orders|customers|products))\b", refinement, re.I))


class QueryOptimizationService:
    def save_history(self, request_id, result):
        try:
            fields = {key: result.get(key) for key in ("status", "query_signature", "cost_analysis", "optimization",
                "optimization_iteration", "optimization_history", "validated_query", "generated_query", "query_template_id", "query_parameters", "execution_success", "execution_error",
                "execution_error_code", "execution_metadata", "final_response", "execution_trace")}
            fields["cost_guard"] = result.get("cost_analysis")
            fields["execution_metrics"] = result.get("execution_metadata")
            if result.get("execution_metadata") is not None:
                from services.query_history_service import QueryHistoryService
                fields["result_metadata"] = result["execution_metadata"]
                fields["result_preview"] = QueryHistoryService()._preview(result.get("execution_result") or [])
            get_database()[QUERY_HISTORY_COLLECTION].update_one({"_id": request_id}, {"$set": encode_record(to_json_safe(fields))})
        except Exception as exc:
            logger.warning("query_optimization_history_failed request_id=%s error_type=%s error_code=%s",
                           request_id, type(exc).__name__, getattr(exc, "code", None))

    def collection(self):
        c = get_database()["_system_pending_queries"]
        c.create_index("expires_at", expireAfterSeconds=0)
        return c

    def pause(self, result, question):
        self.collection().insert_one(encode_record({"_id": result["request_id"], "user_id": DEFAULT_USER_ID,
            "state": "pending", "question": question, "original_question": question, "result": result,
            "optimization_iteration": 0, "optimization_history": [],
            "expires_at": datetime.now(timezone.utc)+timedelta(hours=1)}))
        logger.info("query_optimization_requested request_id=%s", result["request_id"])

    def continue_query(self, request_id, payload):
        c = self.collection()
        claim = uuid4().hex
        pending = decode_record(c.find_one_and_update({"_id": request_id, "user_id": DEFAULT_USER_ID,
            "state": {"$in": ["pending", "paged"]}, "expires_at": {"$gt": datetime.now(timezone.utc)}},
            {"$set": {"state": "executing", "claim": claim}}))
        if not pending: raise HTTPException(409, "Request is expired, unavailable, or already executing")
        executed = False
        try:
            result = deepcopy(pending["result"])
            choice = payload.optimization
            allowed = {s["id"] for s in result.get("suggestions", [])} | {"pagination"}
            if choice == "pagination" and not any(s["id"] == "paginate_100" for s in result.get("suggestions", [])):
                raise HTTPException(422, "Pagination is not supported for this query")
            if choice not in allowed: raise HTTPException(422, "Unsupported optimization")
            if pending["state"] == "paged" and choice not in {"pagination", "paginate_100"}:
                raise HTTPException(409, "This request is already completed")
            logger.info("query_optimization_selected request_id=%s choice=%s", request_id, choice)
            if choice == "run_anyway":
                logger.info("query_optimization_declined request_id=%s", request_id)
                logger.info("query_run_anyway request_id=%s", request_id)
            logger.info("query_optimization_context_loaded request_id=%s", request_id)
            previous = deepcopy(result["validated_query"])
            previous_cost = result["cost_analysis"]
            iteration = pending.get("optimization_iteration", 0)
            history = pending.get("optimization_history", [])
            natural = choice in {"refine", "narrow_date"} or (choice == "summary" and previous["operation"] != "find")
            intent = payload.refinement if choice != "summary" else "Summarize instead of returning raw rows using a compact aggregation."
            if natural and not (intent or "").strip(): raise HTTPException(422, "Refinement text is required")
            if natural and is_new_intent(intent):
                logger.info("query_optimization_full_pipeline_restart request_id=%s reason=explicit_new_intent", request_id)
                from services.query_history_service import QueryHistoryService
                regenerated = QueryHistoryService().run(intent)
                regenerated["question"] = intent
                c.update_one({"_id": request_id, "claim": claim}, {"$set": {"state": "completed"}})
                return regenerated
            if natural and iteration >= QUERY_MAX_OPTIMIZATION_ITERATIONS:
                raise HTTPException(422, "Refinement limit reached. Choose pagination, a limit, run anyway, or submit a new question.")
            logger.info("query_optimization_context_reused request_id=%s", request_id)
            if pending["state"] == "paged":
                stored_size = pending.get("page_size")
                if payload.value is not None and payload.value != stored_size: raise HTTPException(422, "Page size cannot change")
                payload.value = stored_size
                previous = deepcopy(pending.get("pagination_base_query", previous))
            trace = []
            with request_context(request_id), workflow_listener(trace.append):
                if natural:
                    logger.info("query_optimization_generator_started request_id=%s", request_id)
                    emit("QueryGeneratorAgent", "running", message="Refining existing query; context reused")
                    generated = stage("QueryGeneratorAgent", QueryGeneratorAgent().optimize,
                        pending["question"], result.get("query_context"), result["schema_spec"], previous,
                        intent, result.get("applied_preferences"), result.get("applied_knowledge"))
                    query = stage("QueryParameterBinder", bind_query_parameters, generated["template"], generated["parameters"])
                    saved_preferences = result.get("applied_preferences") or []
                    if saved_preferences:
                        from services.user_preference_service import apply_preferences_to_query
                        apply_preferences_to_query(
                            query, saved_preferences, DEFAULT_USER_ID,
                            question=intent or pending["question"],
                        )
                    logger.info("query_optimization_generator_completed request_id=%s", request_id)
                    if query.get("collection") != previous["collection"] or (choice != "summary" and query.get("operation") != previous["operation"]):
                        raise HTTPException(422, "Refinement changed the primary intent. Submit a new question for that change.")
                    # Keep existing find constraints even if the model accidentally drops them.
                    replacing = bool(re.search(r"\b(replace|remove|instead of|no longer)\b", intent, re.I))
                    if previous["operation"] == query.get("operation") == "find" and not replacing:
                        old_filter = previous.get("filter", {})
                        if old_filter and query.get("filter") != old_filter:
                            query["filter"] = {"$and": [old_filter, query.get("filter", {})]}
                        if previous.get("limit"):
                            query["limit"] = min(previous["limit"], query.get("limit") or previous["limit"])
                else:
                    query = OptimizationAdvisor().apply(previous, choice, payload.value, payload.page)
                    existing = result.get("generated_query") or {}
                    if choice == "run_anyway" and "template" in existing:
                        generated = reconcile_template(existing, query)
                    else:
                        generated = template_from_concrete(query,
                            choice in {"latest_100", "paginate_100", "pagination", "limit"}
                            or "limit" in existing.get("parameters", {}))
                generated = parameterize_filter_values(reconcile_template(generated, query))
                query = stage("QueryParameterBinder", bind_query_parameters, generated["template"], generated["parameters"])
                logger.info("query_optimization_query_changed request_id=%s changed=%s", request_id, query != previous)
                query = stage("ValidatorAgent", ValidatorAgent().validate, query, result["schema_spec"])
                logger.info("query_optimization_validation_completed request_id=%s", request_id)
                generated = parameterize_filter_values(reconcile_template(generated, query))
                cost = stage("QueryCostGuard", QueryCostGuard().analyze, query)
                cost["previous_estimated_rows"] = previous_cost.get("estimated_rows")
                emit("QueryCostGuard", "completed", message=f"{cost['status']}; estimated rows: {cost.get('estimated_rows', 'unknown')}")
                logger.info("query_optimization_cost_reestimated request_id=%s estimated_rows=%s", request_id, cost.get("estimated_rows"))
                iteration += int(pending["state"] != "paged" and choice != "run_anyway")
                optimization = {"suggested": True, "selected": choice, "value": payload.value or query.get("limit"),
                    "iteration": iteration, "refinement": intent if natural else None,
                    "previous_estimated_rows": previous_cost.get("estimated_rows"), "new_estimated_rows": cost.get("estimated_rows")}
                if pending["state"] != "paged": history = history + [deepcopy(optimization)]
                result.update({"query_signature": cost["query_signature"], "cost_analysis": cost,
                    "generated_query": generated, "query_parameters": generated["parameters"], "query_template_id": None, "validated_query": query, "optimization": optimization,
                    "optimization_iteration": iteration, "optimization_history": history,
                    "execution_trace": trace})
                if cost["status"] == "heavy" and choice != "run_anyway":
                    logger.info("query_optimization_still_heavy request_id=%s", request_id)
                    result.update(stage("OptimizationAdvisor", OptimizationAdvisor().suggest, query, cost))
                    if iteration >= QUERY_MAX_OPTIMIZATION_ITERATIONS:
                        result["suggestions"] = [item for item in result["suggestions"] if item["id"] not in {"refine", "narrow_date", "summary"}]
                        result["message"] = "This request is still large. Choose pagination, a result limit, run anyway, or submit a more specific new question."
                    result.update({"status": "optimization_required", "execution_success": False})
                    for key in ("execution_result", "execution_error", "execution_error_code", "pagination"):
                        result.pop(key, None)
                    c.update_one({"_id": request_id, "claim": claim}, {"$set": encode_record({
                        "state": "pending", "result": result, "optimization_iteration": iteration, "optimization_history": history})})
                    self.save_history(request_id, result)
                    return result
                logger.info("query_optimization_became_safe request_id=%s run_anyway=%s", request_id, choice == "run_anyway")
                execution = stage("ExecutorAgent", ExecutorAgent().execute, query)
                executed = True
                PerformanceLogger().record(query, cost, execution, optimization)
                if execution.get("ok") and query["operation"] == "aggregate" and execution.get("row_count"):
                    summary = stage("ResultsSummaryAgent", ResultsSummaryAgent().summarize, pending["question"], execution)
                else:
                    rows = execution.get("all_rows", [])
                    summary = {"summary_text": f"Count: {rows[0]['count']}." if query["operation"] == "count_documents" and rows else f"Found {len(rows)} results."}
                    emit("ResultsSummaryAgent", "skipped", message="Deterministic summary used")
            if not execution.get("ok"):
                summary = {"summary_text": execution.get("error") if execution.get("error_code") == "query_timeout" else "The query could not be executed. Refine the request and try again."}
            result.update({"status": "completed", "execution_success": execution["ok"],
                "execution_result": execution["all_rows"], "execution_error": execution.get("error"),
                "execution_error_code": execution.get("error_code"),
                "validated_query": query, "final_response": summary.get("summary_text"),
                "results_summary_agent_output": summary, "execution_trace": trace, "optimization": optimization,
                "execution_metadata": {"row_count": execution["row_count"], "execution_time_ms": execution["execution_time_ms"]}})
            paged = choice in {"paginate_100", "pagination"}
            if paged:
                # Each page is bounded at the database; full pages permit a next-page probe.
                result["pagination"] = {"page": payload.page, "page_size": query["limit"],
                    "has_next": execution["row_count"] == query["limit"], "has_previous": payload.page > 1}
            c.update_one({"_id": request_id, "claim": claim}, {"$set": encode_record({"state": "paged" if paged else "completed", "page_size": query.get("limit") if paged else None, "result": {key: value for key, value in result.items() if key not in {"execution_result", "results_summary_agent_output", "final_response"}},
                "pagination_base_query": previous if paged else None, "optimization_iteration": iteration, "optimization_history": history})})
            self.save_history(request_id, result)
            return result
        except Exception:
            c.update_one({"_id": request_id, "claim": claim}, {"$set": {"state": "completed" if executed else pending["state"]}})
            raise
