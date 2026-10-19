"""Direct multi-agent orchestrator. No LangGraph."""
from __future__ import annotations

from typing import Any
from copy import deepcopy
from agent.progress import stage, emit
import logging
from agent.telemetry import capture_usage, totals
from agent.llm import LLMRouterError
from agents.query_context_agent import QueryContextAgent
from services.schema_bootstrap_service import schema_bootstrap_service
from services.context_builder import ContextBuilder
from services.knowledge_retriever import KnowledgeRetriever
from services.knowledge_collection_suggestions import apply_rule_suggestions
from services.filter_value_resolution import apply_value_resolution, parameterize_resolved_query
from services.query.dates.normalization import apply_date_normalization
from services.query_entity_guard import rule_entity_mismatch
from database.record_safety import safe_record
from services.query_parameter_binder import bind_query_parameters, template_from_concrete
from services.query_cost_guard import QueryCostGuard
from services.aggregation_validation import normalize_lookups, normalization_context
from services.optimization_advisor import OptimizationAdvisor
from services.query_performance_memory import PerformanceLogger
from services.learned_optimization_service import LearnedOptimizationService

_knowledge_retriever = KnowledgeRetriever()

from agent.tracing import traceable
from agents.executor_agent import ExecutorAgent
from agents.query_generator_agent import QueryGeneratorAgent
from agents.results_summary_agent import ResultsSummaryAgent
from agents.validator_agent import QueryValidationError, ValidatorAgent
from agent.config import DEFAULT_USER_ID
from agent.telemetry import current_request_id
from services.user_preference_service import (
    UserPreferenceService,
    applied_preferences_for_query,
    apply_preferences_to_query,
)


class OrchestratorAgent:
    def __init__(self) -> None:
        self.query_context_agent = QueryContextAgent()
        self.query_generator = QueryGeneratorAgent()
        self.validator = ValidatorAgent()
        self.executor = ExecutorAgent()
        self.summary_agent = ResultsSummaryAgent()
        self.preference_service = UserPreferenceService(user_id=DEFAULT_USER_ID)
        self.context_builder = ContextBuilder()
        self.knowledge_retriever = _knowledge_retriever

    @traceable(name="orchestrator.run")
    def run(self, question: str) -> dict[str, Any]:
        with capture_usage() as records:
            result = self._run(question)
            usage = totals(records)
            cache = result.pop("_schema_cache", {})
            usage.update({
                "schema_cache_hit": bool(cache.get("hit")),
                "schema_cache_miss": bool(cache.get("miss")),
                "schema_cache_refresh": bool(cache.get("refresh")),
                "schema_cache_stale_fallback": bool(cache.get("stale_fallback")),
            })
            usage.update(result.pop("_schema_retrieval", {}))
            usage.update(result.pop("_knowledge_retrieval", {}))
            usage["query_context_input_tokens"] = sum(
                attempt.get("input_tokens", 0) for attempt in usage["attempts"]
                if attempt.get("agent") == "query_context"
            )
            result["telemetry"] = usage
            path = result.get("execution_path")
            if path is not None:
                path["llm_calls"] = usage["total_llm_calls"]
                logging.getLogger(__name__).debug(
                    "execution_path route=%s agents_called=%s llm_calls=%d summary_skipped=%s repair_used=%s",
                    path["route"], ",".join(path["agents_called"]), path["llm_calls"],
                    path["summary_skipped"], path["repair_used"],
                )
            return result

    def _run(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"execution_success": False, "execution_error": "question is empty"}

        route = query_context = schema_spec = generated_query = validated_query = None
        cache_status: dict[str, Any] = {}
        retrieval_status: dict[str, Any] = {}
        knowledge_status: dict[str, Any] = {}
        pending_knowledge_diagnostics = None
        applied_knowledge: list[dict[str, Any]] = []
        execution: dict[str, Any] = {}
        execution_time_ms = 0.0
        validation_errors: list[dict[str, Any]] = []
        agents_called: list[str] = []
        learned_optimization = None
        learning_original = learning_cost = None
        try:
            catalog = stage("Precheck", schema_bootstrap_service.get_catalog)
            state = schema_bootstrap_service.status()
            cache_status = {
                "hit": state.get("schema_source") == "cache",
                "miss": state.get("schema_source") == "rebuilt",
                "refresh": state.get("schema_source") == "refreshed",
                "stale_fallback": state.get("schema_source") == "stale_fallback",
            }
            if not catalog.get("collections"):
                return {"execution_success": False, "execution_error": "The configured MongoDB database has no collections.",
                        "_schema_cache": cache_status}
            agents_called.append("query_context")
            preferences = self.preference_service.retrieve_for_query(
                question, catalog, current_request_id(),
            ) if catalog.get("database") else []
            retrieval = stage("SchemaRetriever", schema_bootstrap_service.retrieve, question, preferences)
            candidate_catalog = retrieval["catalog"]
            retrieval_status = retrieval["telemetry"]
            # Natural wording may not name the collection. Schema selection gives
            # deterministic collection scope for a second relevance check.
            preferences = stage("PreferenceRetriever", self.preference_service.retrieve_for_query,
                question, catalog, current_request_id(),
                selected_collections=set(candidate_catalog.get("collections", {})))
            question_embedding = retrieval.pop("_query_embedding", None)
            knowledge_result = self.knowledge_retriever.rules_for_context(
                question_embedding, candidate_catalog, catalog,
                question=question, question_embedding_reused=bool(question_embedding), defer_diagnostics=True,
            )
            knowledge_candidates = knowledge_result["knowledge"]
            pending_knowledge_diagnostics = knowledge_result.get("_context_diagnostics")
            knowledge_status = knowledge_result["telemetry"]
            context = self.context_builder.build(
                question, candidate_catalog, knowledge_candidates, preferences, full_catalog=catalog,
            )
            context_result = stage("QueryContextAgent", self.query_context_agent.build,
                context["question"], context["schema"], context["user_preferences"],
                context["database_knowledge"],
            )
            retrieval_status.update(context_result.pop("_prompt_compaction", {}))
            if context_result.get("error"):
                if "_context_diagnostics" in knowledge_result:
                    self.knowledge_retriever.log_context_diagnostics(knowledge_result["_context_diagnostics"])
                    pending_knowledge_diagnostics = None
                return {
                    "execution_success": False,
                    "execution_error": context_result.get("detail"),
                    "_schema_cache": cache_status,
                    "_schema_retrieval": retrieval_status,
                    "_knowledge_retrieval": knowledge_status,
                }
            query_context = context_result["query_context"]
            schema_spec = context_result["schema_spec"]
            query_context, schema_spec, suggested_collections = apply_rule_suggestions(
                query_context, schema_spec, knowledge_candidates, catalog, question=question,
            )
            if "_context_diagnostics" in knowledge_result:
                self.knowledge_retriever.log_context_diagnostics(
                    knowledge_result["_context_diagnostics"], suggested_collections,
                )
                pending_knowledge_diagnostics = None
            preferences = context_result.get("user_preferences", preferences)
            selected_collections = set(query_context.get("collections", []))
            applicable_records = []
            for record in knowledge_result["_active_rules"]:
                if rule_entity_mismatch(record, question):
                    logging.getLogger(__name__).info(
                        "knowledge_rule_dropped request_id=%s stage=generator knowledge_id=%s reason=entity_mismatch",
                        safe_record(current_request_id()), safe_record(str(record.get("_id", ""))),
                    )
                else:
                    applicable_records.append(record)
            generator_rules = self.knowledge_retriever.rules_for_generator(
                question, selected_collections, query_context.get("fields", {}),
                full_schema=catalog, records=applicable_records, user_preferences=preferences,
                context_rules=knowledge_candidates,
                **({"question_embedding": question_embedding} if question_embedding is not None else {}),
            )
            applied_knowledge = generator_rules["knowledge"]
            preferences = generator_rules["user_preferences"]
            # Restore catalog-owned rule fields within finalized collections.
            schema_spec = self.context_builder.build(
                question, schema_spec, [], preferences, full_catalog=catalog,
            )["schema"]
            for item in applied_knowledge:
                for name in set(item.get("collections") or selected_collections) & selected_collections:
                    available_fields = catalog["collections"][name].get("fields", {})
                    rule_fields = list(item.get("fields", []))
                    from services.query.plans.config import PlanConfig
                    if PlanConfig.from_env().enabled:
                        from services.query.plans.directives import directive_field_references
                        rule_fields.extend(sorted(directive_field_references(item, name) - set(rule_fields)))
                    for field in rule_fields:
                        if field in available_fields:
                            schema_spec["collections"][name]["fields"][field] = deepcopy(available_fields[field])
            query_context["fields"] = {
                name: list(entry["fields"]) for name, entry in schema_spec["collections"].items()
            }
            if applied_knowledge:
                logging.getLogger(__name__).info(
                    "knowledge_applied_to_query request_id=%s count=%d",
                    current_request_id(), len(applied_knowledge),
                )
            route = {
                "complexity": query_context["complexity"],
                "operation": query_context["operation"],
            }
            agents_called.append("query_generator")
            generated_query = stage("QueryGeneratorAgent", self.query_generator.generate,
                question, schema_spec, operation_hint=query_context["operation"],
                user_preferences=preferences, database_knowledge=applied_knowledge,
            )
            bound_query = stage("QueryParameterBinder", bind_query_parameters, generated_query["template"], generated_query["parameters"])
            apply_preferences_to_query(bound_query, preferences, self.preference_service.user_id, question=question)
            repairs = 0

            validation_errors: list[dict[str, Any]] = []
            validated_query: dict[str, Any] | None = None
            for attempt in range(2):
                try:
                    bound_query = stage("AggregationNormalizer", normalize_lookups, bound_query)
                    with normalization_context():
                        validated_query = stage("ValidatorAgent", self.validator.validate, bound_query, schema_spec)
                    validated_query = apply_date_normalization(validated_query, question=question)
                    break
                except QueryValidationError as exc:
                    validation_errors.append(exc.to_dict())
                    if attempt == 0 and exc.repairable and "Banned" not in str(exc) and "Unsupported operation" not in str(exc):
                        repairs += 1
                        agents_called.append("repair")
                        generated_query = stage("QueryGeneratorAgent", self.query_generator.repair,
                            question, schema_spec, bound_query, str(exc), preferences,
                            database_knowledge=applied_knowledge,
                        )
                        bound_query = stage("QueryParameterBinder", bind_query_parameters, generated_query["template"], generated_query["parameters"])
                        apply_preferences_to_query(bound_query, preferences, self.preference_service.user_id, question=question)
                    else:
                        raise

            assert validated_query is not None
            validated_query = apply_value_resolution(validated_query, schema_spec, knowledge_result["_active_rules"])
            generated_query = parameterize_resolved_query(generated_query, validated_query)
            cost = stage("QueryCostGuard", QueryCostGuard().analyze, validated_query)
            learning_original, learning_cost = validated_query, cost
            validated_query, learned_optimization = stage("LearnedOptimization", LearnedOptimizationService().apply,
                validated_query, cost, schema_spec, self.validator)
            if learned_optimization:
                try:
                    validated_query = apply_date_normalization(validated_query, question=question)
                except QueryValidationError:
                    logging.getLogger(__name__).info("learned_optimization_skipped reason=date_validation")
                    validated_query, learned_optimization = learning_original, None
            if learned_optimization:
                validated_query = apply_value_resolution(validated_query, schema_spec, knowledge_result["_active_rules"])
                intent = generated_query.get("semantic_intent")
                generated_query = template_from_concrete(validated_query, "limit" in generated_query["parameters"])
                if intent:
                    generated_query["semantic_intent"] = intent
                cost = stage("QueryCostGuard", QueryCostGuard().analyze, validated_query)
            if cost["status"] == "heavy":
                return {"status": "optimization_required", "question": question, "cost_analysis": cost,
                        "learned_optimization": learned_optimization,
                        "query_signature": cost["query_signature"], **stage("OptimizationAdvisor", OptimizationAdvisor().suggest, validated_query, cost, schema_spec),
                        "schema_spec": schema_spec, "query_context": query_context, "generated_query": generated_query,
                        "validated_query": validated_query, "applied_preferences": preferences, "applied_knowledge": applied_knowledge}
            execution = stage("ExecutorAgent", self.executor.execute, validated_query)
            PerformanceLogger().record(validated_query, cost, execution)
            execution_time_ms += execution.get("execution_time_ms", 0)

            # One repair attempt for a database execution error.
            # Only structural MongoDB command failures are repairable, not connectivity/auth failures.
            if not execution.get("ok") and repairs == 0 and execution.get("repairable", False):
                repairs += 1
                agents_called.append("repair")
                repaired = stage("QueryGeneratorAgent", self.query_generator.repair,
                    question,
                    schema_spec,
                    validated_query,
                    str(execution.get("error") or "MongoDB execution failed"),
                    preferences,
                    database_knowledge=applied_knowledge,
                )
                bound_query = stage("QueryParameterBinder", bind_query_parameters, repaired["template"], repaired["parameters"])
                apply_preferences_to_query(bound_query, preferences, self.preference_service.user_id, question=question)
                try:
                    bound_query = stage("AggregationNormalizer", normalize_lookups, bound_query)
                    with normalization_context():
                        validated_query = stage("ValidatorAgent", self.validator.validate, bound_query, schema_spec)
                    validated_query = apply_date_normalization(validated_query, question=question)
                    validated_query = apply_value_resolution(validated_query, schema_spec, knowledge_result["_active_rules"])
                    generated_query = parameterize_resolved_query(repaired, validated_query)
                    cost = stage("QueryCostGuard", QueryCostGuard().analyze, validated_query)
                    if cost["status"] == "heavy":
                        return {"status": "optimization_required", "question": question, "cost_analysis": cost,
                                "query_signature": cost["query_signature"], **stage("OptimizationAdvisor", OptimizationAdvisor().suggest, validated_query, cost, schema_spec),
                                "validated_query": validated_query, "generated_query": generated_query, "schema_spec": schema_spec,
                                "query_context": query_context, "applied_preferences": preferences, "applied_knowledge": applied_knowledge}
                    execution = stage("ExecutorAgent", self.executor.execute, validated_query)
                    PerformanceLogger().record(validated_query, cost, execution)
                    execution_time_ms += execution.get("execution_time_ms", 0)
                except QueryValidationError as exc:
                    validation_errors.append(exc.to_dict())
                    raise

            if learned_optimization:
                PerformanceLogger().record_optimization(learning_original, validated_query, learning_cost,
                                                        cost, execution, learned_optimization)
            template_id = None  # Reusable templates require explicit positive feedback.
            needs_summary = bool(execution.get("ok") and execution.get("row_count") and (
                query_context["complexity"] == "complex" or validated_query["operation"] == "aggregate"
            ))
            if needs_summary:
                agents_called.append("summary")
                summary = stage("ResultsSummaryAgent", self.summary_agent.summarize, question, execution)
            else:
                emit("ResultsSummaryAgent", "skipped", message="Deterministic summary used")
                rows = execution.get("all_rows", [])
                if not execution.get("ok"):
                    text = execution.get("error") if execution.get("error_code") == "query_timeout" else "I could not execute the MongoDB query."
                elif validated_query["operation"] == "count_documents" and rows:
                    text = f"Count: {rows[0]['count']}."
                else:
                    text = f"Found {len(rows)} results." if rows else "No matching data was found."
                summary = {"summary_text": text, "row_count": len(rows)}
            path = {"route": route["complexity"], "agents_called": agents_called, "llm_calls": len(agents_called),
                    "repair_used": bool(repairs), "summary_used": needs_summary, "summary_skipped": not needs_summary}
            logging.getLogger(__name__).debug("execution_path=%s", path)
            return {
                "status": "completed", "cost_analysis": cost, "query_signature": cost["query_signature"],
                "learned_optimization": learned_optimization,
                "route": route,
                "query_context": query_context,
                "execution_path": path,
                "final_response": summary.get("summary_text"),
                "execution_success": bool(execution.get("ok")),
                "execution_result": execution.get("all_rows", []),
                "execution_error": execution.get("error"),
                "execution_error_code": execution.get("error_code"),
                "execution_metadata": {"row_count": execution.get("row_count", 0), "execution_time_ms": execution_time_ms},
                "schema_spec": schema_spec,
                "query_plan": None,
                "generated_query": generated_query,
                "validated_query": validated_query,
                "query_template_id": template_id,
                "query_parameters": generated_query["parameters"],
                "applied_preferences": applied_preferences_for_query(validated_query, preferences),
                "applied_knowledge": applied_knowledge,
                "validation_errors": validation_errors,
                "python_artifact": None,
                "results_summary_agent_output": summary,
                "_schema_cache": cache_status,
                "_schema_retrieval": retrieval_status,
                "_knowledge_retrieval": knowledge_status,
            }
        except Exception as exc:
            if pending_knowledge_diagnostics is not None:
                self.knowledge_retriever.log_context_diagnostics(pending_knowledge_diagnostics)
            # Keep completed stages available for history even when a later stage fails.
            if isinstance(exc, QueryValidationError):
                detail = exc.to_dict()
                if not validation_errors or validation_errors[-1] != detail:
                    validation_errors.append(detail)
            error = str(exc) if isinstance(exc, QueryValidationError) else "Query pipeline failed."
            code = exc.details["code"] if isinstance(exc, QueryValidationError) else "query_pipeline_failed"
            if isinstance(exc, LLMRouterError):
                code = exc.code
                error = ("The query generator returned an invalid JSON query template."
                         if code == "invalid_query_output" else
                         "The language model returned an invalid structured response."
                         if code == "invalid_llm_output" else "The language model is unavailable. Please retry.")
                if exc.detail:
                    error += " " + exc.detail
                validation_errors.append({"code": code, "detail": error, "repairable": False})
            return {
                "execution_success": False,
                "execution_error": error,
                "execution_error_code": code,
                "final_response": "I could not complete this query. See execution_error for details.",
                "route": route, "query_context": query_context, "schema_spec": schema_spec, "query_plan": None,
                "execution_path": {"route": route.get("complexity") if route else None,
                                   "agents_called": agents_called, "llm_calls": len(agents_called),
                                   "repair_used": "repair" in agents_called,
                                   "summary_used": "summary" in agents_called,
                                   "summary_skipped": "summary" not in agents_called},
                "generated_query": generated_query, "validated_query": validated_query,
                "validation_errors": validation_errors,
                "applied_knowledge": applied_knowledge,
                "execution_result": execution.get("all_rows", []),
                "execution_metadata": {"row_count": execution.get("row_count", 0), "execution_time_ms": execution_time_ms},
                "_schema_cache": cache_status,
                "_schema_retrieval": retrieval_status,
                "_knowledge_retrieval": knowledge_status,
            }


def run_request(question: str) -> dict[str, Any]:
    return OrchestratorAgent().run(question)
