"""PlannerAgent: create a database-neutral plan for a MongoDB query."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class PlannerAgent:
    @traceable(name="planner.plan")
    def plan(self, question: str, schema_spec: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt("planner", {
            "question": question,
            "schema_json": json.dumps(schema_spec, ensure_ascii=False, default=str),
        })
        response = get_llm().invoke([HumanMessage(content=prompt)])
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("Planner returned a non-object JSON response")
        parsed.setdefault("task", question)
        parsed.setdefault("query_type", "find")
        parsed.setdefault("related_collections", [])
        parsed.setdefault("steps", [])
        parsed.setdefault("output_fields", [])
        parsed.setdefault("python_postprocessing_required", False)
        parsed.setdefault("python_goal", "")
        return parsed
