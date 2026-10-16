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
class QueryContext(StrictOutput):
    complexity: Literal["simple", "complex"]
    operation: Literal["find", "count", "distinct", "aggregate"]
    collections: list[str] = Field(min_length=1)
    fields: dict[str, list[str]]

    @model_validator(mode="before")
    @classmethod
    def normalize_operation_alias(cls, value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if result.get("operation") == "count_documents":
            result["operation"] = "count"
        if result.get("complexity") == "aggregate" and result.get("operation") == "aggregate":
            result["complexity"] = "complex"
        return result


class CollectionEnrichment(StrictOutput):
    collection: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=400)
    aliases: list[str] = Field(max_length=12)


class SchemaEnrichmentBatch(StrictOutput):
    collections: list[CollectionEnrichment] = Field(min_length=1, max_length=50)
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
    skip: int = 0
    pipeline: list[dict[str, Any]] | None = Field(default=None, description=(
        "Read-only one-key aggregation stages. $lookup: from/as nonempty strings; "
        "either paired localField/foreignField, or pipeline array with optional let object. "
        "Mix forms only when the deployment capability in the prompt permits it. "
        "Nested lookup and facet pipelines obey the same rules. "
        "Deterministic structural validation runs after parameter binding."
    ))
    field: str | None = None
    @model_validator(mode="after")
    def arguments(self):
        if self.operation == "aggregate" and self.pipeline is None:
            raise ValueError("pipeline required")
        if self.operation == "distinct" and not self.field:
            raise ValueError("field required")
        return self


class LimitParameter(StrictOutput):
    name: Literal["limit"] = Field(alias="$param")


class QueryTemplate(Query):
    limit: int | LimitParameter | None = None


class GeneratedQuery(StrictOutput):
    semantic_intent: str = Field(default="", max_length=600)
    template: QueryTemplate
    parameters: dict[str, Any]

    @classmethod
    def generation_json_schema(cls):
        """Simpler outbound contract; parsing still accepts valid legacy bindings."""
        document = cls.model_json_schema()
        document["properties"]["parameters"] = {
            "title": "Parameters", "type": "object", "properties": {}, "additionalProperties": False,
        }
        document["$defs"]["QueryTemplate"]["properties"]["limit"] = {
            "title": "Limit", "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None,
        }
        document["$defs"].pop("LimitParameter", None)
        return document

    @model_validator(mode="before")
    @classmethod
    def normalize_placeholder_syntax(cls, value):
        from services.query_parameter_binder import normalize_parameter_placeholders
        return normalize_parameter_placeholders(value)

    @model_validator(mode="after")
    def valid_binding(self):
        from services.query_parameter_binder import bind_query_parameters
        bind_query_parameters(self.template.model_dump(by_alias=True, exclude_unset=True), self.parameters)
        return self
