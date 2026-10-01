"""PlannerAgent: create a database-neutral plan for a MongoDB query."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.output_schemas import Plan
from database.schema_inspector import compact_schema
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class PlannerAgent:
    @traceable(name="planner.plan")
    def plan(self, question: str, schema_spec: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt("planner", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
        })
        response = llm_router.invoke("planner", [HumanMessage(content=prompt)], Plan)
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("Planner returned a non-object JSON response")
        return Plan.model_validate(parsed).model_dump()
