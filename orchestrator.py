"""Direct multi-agent orchestrator. No LangGraph."""
from __future__ import annotations

from typing import Any
import logging
from agent.telemetry import capture_usage, totals
from agents.query_context_agent import QueryContextAgent
from services.schema_bootstrap_service import schema_bootstrap_service
from services.context_builder import ContextBuilder
from services.knowledge_retriever import KnowledgeRetriever

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
        applied_knowledge: list[dict[str, Any]] = []
        execution: dict[str, Any] = {}
        execution_time_ms = 0.0
        validation_errors: list[str] = []
        agents_called: list[str] = []
        try:
            catalog = schema_bootstrap_service.get_catalog()
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
            retrieval = schema_bootstrap_service.retrieve(question, preferences)
            candidate_catalog = retrieval["catalog"]
            retrieval_status = retrieval["telemetry"]
            knowledge_result = self.knowledge_retriever.retrieve(
                question, candidate_catalog, retrieval.pop("_query_embedding", None), catalog,
            )
            knowledge_candidates = knowledge_result["knowledge"]
            knowledge_status = knowledge_result["telemetry"]
            context = self.context_builder.build(
                question, candidate_catalog, knowledge_candidates, preferences,
            )
            context_result = self.query_context_agent.build(
                context["question"], context["schema"], context["user_preferences"],
                context["database_knowledge"],
            )
            if context_result.get("error"):
                return {
                    "execution_success": False,
                    "execution_error": context_result.get("detail"),
                    "_schema_cache": cache_status,
                    "_schema_retrieval": retrieval_status,
                    "_knowledge_retrieval": knowledge_status,
                }
            query_context = context_result["query_context"]
            schema_spec = context_result["schema_spec"]
            preferences = context_result.get("user_preferences", preferences)
            selected_collections = set(query_context.get("collections", []))
            applied_knowledge = [
                item for item in knowledge_candidates
                if not item.get("collections") or selected_collections.intersection(item["collections"])
            ]
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
            generated_query = self.query_generator.generate(
                question, schema_spec, operation_hint=query_context["operation"],
                user_preferences=preferences, database_knowledge=applied_knowledge,
            )
            apply_preferences_to_query(generated_query, preferences, self.preference_service.user_id)
            repairs = 0

            validation_errors: list[str] = []
            validated_query: dict[str, Any] | None = None
            for attempt in range(2):
                try:
                    validated_query = self.validator.validate(generated_query, schema_spec)
                    break
                except QueryValidationError as exc:
                    validation_errors.append(str(exc))
                    if attempt == 0 and "Banned" not in str(exc) and "Unsupported operation" not in str(exc):
                        repairs += 1
                        agents_called.append("repair")
                        generated_query = self.query_generator.repair(
                            question, schema_spec, generated_query, str(exc), preferences,
                            database_knowledge=applied_knowledge,
                        )
                        apply_preferences_to_query(generated_query, preferences, self.preference_service.user_id)
                    else:
                        raise

            assert validated_query is not None
            execution = self.executor.execute(validated_query)
            execution_time_ms += execution.get("execution_time_ms", 0)

            # One repair attempt for a database execution error.
            # Only structural MongoDB command failures are repairable, not connectivity/auth failures.
            if not execution.get("ok") and repairs == 0 and execution.get("repairable", False):
                repairs += 1
                agents_called.append("repair")
                repaired = self.query_generator.repair(
                    question,
                    schema_spec,
                    validated_query,
                    str(execution.get("error") or "MongoDB execution failed"),
                    preferences,
                    database_knowledge=applied_knowledge,
                )
                apply_preferences_to_query(repaired, preferences, self.preference_service.user_id)
                try:
                    validated_query = self.validator.validate(repaired, schema_spec)
                    generated_query = repaired
                    execution = self.executor.execute(validated_query)
                    execution_time_ms += execution.get("execution_time_ms", 0)
                except QueryValidationError as exc:
                    validation_errors.append(str(exc))

            needs_summary = bool(execution.get("ok") and execution.get("row_count") and (
                query_context["complexity"] == "complex" or validated_query["operation"] == "aggregate"
            ))
            if needs_summary:
                agents_called.append("summary")
                summary = self.summary_agent.summarize(question, execution)
            else:
                rows = execution.get("all_rows", [])
                if not execution.get("ok"):
                    text = "I could not execute the MongoDB query."
                elif validated_query["operation"] == "count_documents" and rows:
                    text = f"Count: {rows[0]['count']}."
                else:
                    text = f"Found {len(rows)} results." if rows else "No matching data was found."
                summary = {"summary_text": text, "row_count": len(rows)}
            path = {"route": route["complexity"], "agents_called": agents_called, "llm_calls": len(agents_called),
                    "repair_used": bool(repairs), "summary_used": needs_summary, "summary_skipped": not needs_summary}
            logging.getLogger(__name__).debug("execution_path=%s", path)
            return {
                "route": route,
                "query_context": query_context,
                "execution_path": path,
                "final_response": summary.get("summary_text"),
                "execution_success": bool(execution.get("ok")),
                "execution_result": execution.get("all_rows", []),
                "execution_error": execution.get("error"),
                "execution_metadata": {"row_count": execution.get("row_count", 0), "execution_time_ms": execution_time_ms},
                "schema_spec": schema_spec,
                "query_plan": None,
                "generated_query": generated_query,
                "validated_query": validated_query,
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
            # Keep completed stages available for history even when a later stage fails.
            return {
                "execution_success": False, "execution_error": str(exc),
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
