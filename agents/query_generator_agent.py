"""Generate or repair a structured, read-only MongoDB query specification."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class QueryGeneratorAgent:
    def _invoke_json(self, prompt: str) -> dict[str, Any]:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("Query generator returned a non-object JSON response")
        return parsed

    @traceable(name="query_generator.generate")
    def generate(self, question: str, schema_spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        return self._invoke_json(render_prompt("query_generator", {
            "question": question,
            "schema_json": json.dumps(schema_spec, ensure_ascii=False, default=str),
            "plan_json": json.dumps(plan, ensure_ascii=False, default=str),
        }))

    @traceable(name="query_generator.repair")
    def repair(
        self,
        question: str,
        schema_spec: dict[str, Any],
        plan: dict[str, Any],
        previous_query: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        return self._invoke_json(render_prompt("query_repair", {
            "question": question,
            "schema_json": json.dumps(schema_spec, ensure_ascii=False, default=str),
            "plan_json": json.dumps(plan, ensure_ascii=False, default=str),
            "query_json": json.dumps(previous_query, ensure_ascii=False, default=str),
            "error": error,
        }))
