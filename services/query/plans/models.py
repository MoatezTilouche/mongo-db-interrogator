"""Small intent contract; source identifiers are restricted per request."""
from copy import deepcopy
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

Task = Literal["list", "count", "distinct", "group_count", "top_n", "latest_per_key"]
Operator = Literal["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains", "exists"]


class PlanFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    op: Operator
    value: Any


class PlanOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    dir: Literal[1, -1]


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str
    task: Task
    filters: list[PlanFilter] = Field(default_factory=list)
    select: list[str] | None = None
    order: PlanOrder | None = None
    limit: int | None = Field(default=None, gt=0, strict=True)
    group_by: list[str] = Field(default_factory=list)
    metric: Literal["count"] | None = None
    distinct_field: str | None = None
    uses_time: bool = Field(default=False, strict=True)
    time_field: str | None = None
    # A join selects a profiled relationship; the model never supplies endpoints.
    join: str | None = None


def plan_json_schema(selected_schema):
    """Provider-compatible root object; resolution enforces field ownership."""
    document = QueryPlan.model_json_schema()
    names = list(selected_schema.get("collections", {}))
    fields = sorted({path for entry in selected_schema.get("collections", {}).values() for path in entry.get("fields", {})})
    props = deepcopy(document["properties"])
    props["collection"] = {"type": "string", "enum": names}
    scalars = [{"type": kind} for kind in ("string", "number", "boolean", "null")]
    props["filters"] = {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["field", "op", "value"],
        "properties": {"field": {"type": "string", "enum": fields},
                       "op": document["$defs"]["PlanFilter"]["properties"]["op"],
                       "value": {"anyOf": [*scalars, {"type": "array", "items": {"anyOf": scalars}}]}}}}
    for key in ("select", "group_by"):
        array = {"type": "array", "items": {"type": "string", "enum": fields}}
        props[key] = {"anyOf": [array, {"type": "null"}]} if key == "select" else array
    for key in ("distinct_field", "time_field"):
        props[key] = {"anyOf": [{"type": "string", "enum": fields}, {"type": "null"}]}
    props["order"] = {"anyOf": [{"type": "object", "additionalProperties": False,
        "required": ["field", "dir"], "properties": {"field": {"type": "string", "enum": fields},
                                                      "dir": {"type": "integer", "enum": [1, -1]}}}, {"type": "null"}]}
    props["join"] = {"anyOf": [{"type": "string", "enum": names}, {"type": "null"}]} if len(names) > 1 else {"type": "null"}
    props["metric"] = {"anyOf": [{"type": "string", "enum": ["count"]}, {"type": "null"}]}
    props["limit"] = {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}
    # Strict providers require all properties, with optional values nullable.
    for spec in props.values():
        spec.pop("default", None)
    return {"title": "QueryPlan", "type": "object", "additionalProperties": False,
            "properties": props, "required": list(props)}


class PlanOutputContract:
    """Router sends the schema but leaves tolerant parsing/retries to this path."""
    __name__ = "QueryPlan"
    plan_output = True

    def __init__(self, selected_schema):
        self.schema = plan_json_schema(selected_schema)

    def generation_json_schema(self):
        return self.schema

    def __call__(self, parsed):
        return parsed
