"""Deprecated planner retained for compatibility; the active pipeline skips it."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import llm_router, StructuredOutputError
from agent.output_schemas import Plan
from database.schema_inspector import compact_schema
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class PlannerAgent:
    @staticmethod
    def _validate_plan(parsed: Any, schema_spec: dict[str, Any]) -> Plan:
        plan = Plan.model_validate(parsed)
        if plan.collection not in schema_spec.get("collections", {}):
            raise StructuredOutputError("Plan collection must be one of the selected collections, not a field name.")
        return plan

    @traceable(name="planner.plan")
    def plan(self, question: str, schema_spec: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt("planner", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
        })
        def validate_output(parsed):
            self._validate_plan(parsed, schema_spec)

        response = llm_router.invoke("planner", [HumanMessage(content=prompt)], validate_output)
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("Planner returned a non-object JSON response")
        return self._validate_plan(parsed, schema_spec).model_dump()
