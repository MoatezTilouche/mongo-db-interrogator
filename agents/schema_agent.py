"""SchemaAgent: inspect MongoDB and select collections relevant to the question."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from database.schema_inspector import inspect_database_schema, select_schema


class SchemaAgent:
    @traceable(name="schema.identify_collections")
    def identify_collections(self, question: str) -> dict[str, Any]:
        full_schema = inspect_database_schema()
        collections = full_schema.get("collections", {})
        if not collections:
            return {"error": "no_collections", "detail": "The configured MongoDB database has no collections."}

        prompt = render_prompt("schema_selector", {
            "question": question,
            "schema_json": json.dumps(full_schema, ensure_ascii=False, default=str),
        })
        response = get_llm().invoke([HumanMessage(content=prompt)])
        parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
        requested = parsed.get("collections", []) if isinstance(parsed, dict) else []
        requested = [str(name) for name in requested if str(name) in collections]

        if not requested:
            # Safe fallback: keep all collections rather than hallucinating a name.
            requested = list(collections.keys())
        result = select_schema(full_schema, requested)
        result["selection_reason"] = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        return result
