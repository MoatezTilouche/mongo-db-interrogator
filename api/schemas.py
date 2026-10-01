from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    route: dict[str, Any] | None = None
    execution_path: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    final_response: str | None = None
    execution_success: bool = False
    execution_result: list[dict[str, Any]] | None = None
    execution_error: str | None = None
    schema_spec: dict[str, Any] | None = None
    query_plan: dict[str, Any] | None = None
    generated_query: dict[str, Any] | None = None
    validated_query: dict[str, Any] | None = None
    validation_errors: list[str] = []
    python_artifact: dict[str, Any] | None = None
    results_summary_agent_output: dict[str, Any] | None = None
