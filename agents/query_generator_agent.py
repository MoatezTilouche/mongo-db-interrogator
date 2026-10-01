"""Generate or repair a structured, read-only MongoDB query specification."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.output_schemas import Query
from agent.config import DEFAULT_LIMIT, MAX_LIMIT
from database.schema_inspector import compact_schema
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class QueryGeneratorAgent:
    def _invoke_json(self, prompt: str, role: str) -> dict[str, Any]:
        response = llm_router.invoke(role, [HumanMessage(content=prompt)], Query)
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("Query generator returned a non-object JSON response")
        return parsed

    @traceable(name="query_generator.generate")
    def generate(self, question: str, schema_spec: dict[str, Any], plan: dict[str, Any] | None = None, operation_hint: str = "unknown") -> dict[str, Any]:
        return self._invoke_json(render_prompt("query_generator", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
            "optional_plan": "Plan: " + json.dumps(plan) if plan else "",
            "operation_hint": operation_hint,
            "constraints_json": json.dumps({"read_only": True, "default_limit": DEFAULT_LIMIT, "max_limit": MAX_LIMIT}),
        }), "query_generator")

    @traceable(name="query_generator.repair")
    def repair(
        self,
        question: str,
        schema_spec: dict[str, Any],
        previous_query: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        return self._invoke_json(render_prompt("query_repair", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
            "query_json": json.dumps(previous_query, ensure_ascii=False, default=str),
            "error": error,
        }), "repair")
