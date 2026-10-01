"""Direct multi-agent orchestrator. No LangGraph."""
from __future__ import annotations

from typing import Any
import logging
from agent.telemetry import capture_usage, totals
from agents.query_router import QueryRouter
from database.schema_inspector import inspect_database_schema

from agent.tracing import traceable
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.python_artifact_agent import PythonArtifactAgent
from agents.query_generator_agent import QueryGeneratorAgent
from agents.results_summary_agent import ResultsSummaryAgent
from agents.schema_agent import SchemaAgent
from agents.validator_agent import QueryValidationError, ValidatorAgent


class OrchestratorAgent:
    def __init__(self) -> None:
        self.schema_agent = SchemaAgent()
        self.query_router = QueryRouter()
        self.planner_agent = PlannerAgent()
        self.query_generator = QueryGeneratorAgent()
        self.validator = ValidatorAgent()
        self.executor = ExecutorAgent()
        self.python_artifact = PythonArtifactAgent()
        self.summary_agent = ResultsSummaryAgent()

    @traceable(name="orchestrator.run")
    def run(self, question: str) -> dict[str, Any]:
        with capture_usage() as records:
            result = self._run(question)
            result["telemetry"] = totals(records)
            return result

    def _run(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"execution_success": False, "execution_error": "question is empty"}

        try:
            catalog = inspect_database_schema()
            if not catalog.get("collections"):
                return {"execution_success": False, "execution_error": "The configured MongoDB database has no collections."}
            route = self.query_router.route(question)
            schema_spec = self.schema_agent.identify_collections(question, catalog)
            if schema_spec.get("error"):
                return {
                    "execution_success": False,
                    "execution_error": schema_spec.get("detail"),
                    "schema_spec": schema_spec,
                }

            if len(schema_spec["collections"]) > 1:
                route.update(complexity="complex", needs_planner=True, needs_summary=True)
            query_plan = self.planner_agent.plan(question, schema_spec) if route["needs_planner"] else None
            generated_query = self.query_generator.generate(question, schema_spec, query_plan, operation_hint=route["operation"])
            # Guard against a classification miss using selected/generated structure.
            if query_plan is None and (len(schema_spec["collections"]) > 1 or generated_query.get("operation") == "aggregate"):
                route.update(complexity="complex", needs_planner=True, needs_summary=True)
                query_plan = self.planner_agent.plan(question, schema_spec)
                generated_query = self.query_generator.generate(question, schema_spec, query_plan, operation_hint="aggregate")
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
                        generated_query = self.query_generator.repair(
                            question, schema_spec, generated_query, str(exc)
                        )
                    else:
                        raise

            assert validated_query is not None
            execution = self.executor.execute(validated_query)

            # One repair attempt for a database execution error.
            # Only structural MongoDB command failures are repairable, not connectivity/auth failures.
            if not execution.get("ok") and repairs == 0 and execution.get("repairable", False):
                repairs += 1
                repaired = self.query_generator.repair(
                    question,
                    schema_spec,
                    validated_query,
                    str(execution.get("error") or "MongoDB execution failed"),
                )
                try:
                    validated_query = self.validator.validate(repaired, schema_spec)
                    generated_query = repaired
                    execution = self.executor.execute(validated_query)
                except QueryValidationError as exc:
                    validation_errors.append(str(exc))

            artifact = {"enabled": False, "generated": False}
            if execution.get("ok") and query_plan and query_plan.get("python_postprocessing_required"):
                artifact = self.python_artifact.generate(
                    question=question,
                    goal=str(query_plan.get("python_goal") or "Post-process the query result"),
                    rows=execution.get("all_rows", []),
                )

            needs_summary = bool(execution.get("ok") and execution.get("row_count") and (route["needs_summary"] or validated_query["operation"] == "aggregate"))
            if needs_summary:
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
            called = ["query_router", "schema_selection"] + (["planner"] if query_plan else []) + ["query_generator"] + (["repair"] if repairs else []) + (["summary"] if needs_summary else [])
            path = {"route": route["complexity"], "agents_called": called, "planner_skipped": query_plan is None, "summary_skipped": not needs_summary}
            logging.getLogger(__name__).debug("execution_path=%s", path)
            return {
                "route": route,
                "execution_path": path,
                "final_response": summary.get("summary_text"),
                "execution_success": bool(execution.get("ok")),
                "execution_result": execution.get("all_rows", []),
                "execution_error": execution.get("error"),
                "schema_spec": schema_spec,
                "query_plan": query_plan,
                "generated_query": generated_query,
                "validated_query": validated_query,
                "validation_errors": validation_errors,
                "python_artifact": artifact,
                "results_summary_agent_output": summary,
            }
        except Exception as exc:
            return {"execution_success": False, "execution_error": str(exc)}


def run_request(question: str) -> dict[str, Any]:
    return OrchestratorAgent().run(question)
