"""Direct multi-agent orchestrator. No LangGraph."""
from __future__ import annotations

from typing import Any

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
        self.planner_agent = PlannerAgent()
        self.query_generator = QueryGeneratorAgent()
        self.validator = ValidatorAgent()
        self.executor = ExecutorAgent()
        self.python_artifact = PythonArtifactAgent()
        self.summary_agent = ResultsSummaryAgent()

    @traceable(name="orchestrator.run")
    def run(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"execution_success": False, "execution_error": "question is empty"}

        try:
            schema_spec = self.schema_agent.identify_collections(question)
            if schema_spec.get("error"):
                return {
                    "execution_success": False,
                    "execution_error": schema_spec.get("detail"),
                    "schema_spec": schema_spec,
                }

            query_plan = self.planner_agent.plan(question, schema_spec)
            generated_query = self.query_generator.generate(question, schema_spec, query_plan)

            validation_errors: list[str] = []
            validated_query: dict[str, Any] | None = None
            for attempt in range(2):
                try:
                    validated_query = self.validator.validate(generated_query, schema_spec)
                    break
                except QueryValidationError as exc:
                    validation_errors.append(str(exc))
                    if attempt == 0:
                        generated_query = self.query_generator.repair(
                            question, schema_spec, query_plan, generated_query, str(exc)
                        )
                    else:
                        raise

            assert validated_query is not None
            execution = self.executor.execute(validated_query)

            # One repair attempt for a database execution error.
            if not execution.get("ok"):
                repaired = self.query_generator.repair(
                    question,
                    schema_spec,
                    query_plan,
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
            if execution.get("ok") and query_plan.get("python_postprocessing_required"):
                artifact = self.python_artifact.generate(
                    question=question,
                    goal=str(query_plan.get("python_goal") or "Post-process the query result"),
                    rows=execution.get("all_rows", []),
                )

            summary = self.summary_agent.summarize(question, execution)
            return {
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
