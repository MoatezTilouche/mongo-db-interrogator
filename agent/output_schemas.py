from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
class Route(StrictOutput):
    complexity: Literal["simple", "complex"]
    operation: Literal["find", "count", "distinct", "aggregate", "unknown"]
    needs_planner: bool
    needs_schema_selection: bool
    needs_summary: bool
    @model_validator(mode="after")
    def consistent(self):
        if self.needs_planner != (self.complexity == "complex") or (self.operation == "aggregate" and not self.needs_planner):
            raise ValueError("inconsistent route")
        return self
class Selection(StrictOutput):
    collections: list[str] = Field(min_length=1)
    fields: dict[str, list[str]]
class Plan(StrictOutput):
    operation: Literal["find", "count_documents", "distinct", "aggregate"]
    collection: str
    steps: list[str]
    python_postprocessing_required: bool = False
    python_goal: str = ""
class Query(StrictOutput):
    operation: Literal["find", "count_documents", "distinct", "aggregate"]
    collection: str
    filter: dict[str, Any] = Field(default_factory=dict)
    projection: dict[str, Any] | None = None
    sort: list[list[Any]] = Field(default_factory=list)
    limit: int | None = None
    pipeline: list[dict[str, Any]] | None = None
    field: str | None = None
    @model_validator(mode="after")
    def arguments(self):
        if self.operation == "aggregate" and self.pipeline is None:
            raise ValueError("pipeline required")
        if self.operation == "distinct" and not self.field:
            raise ValueError("field required")
        return self
