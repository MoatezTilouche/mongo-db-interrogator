"""Choose relevant schema metadata and query hints in one concise LLM call."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import QueryContext
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from database.schema_inspector import compact_schema, inspect_database_schema


class QueryContextAgent:
    @traceable(name="query_context.build")
    def build(self, question: str, catalog: dict[str, Any] | None = None,
              user_preferences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        catalog = inspect_database_schema() if catalog is None else catalog
        user_preferences = user_preferences or []
        available = compact_schema(catalog)
        if not available:
            return {"error": "no_collections", "detail": "The configured MongoDB database has no collections."}

        prompt = render_prompt("query_context", {
            "question": question,
            "schema_catalog": json.dumps(available, ensure_ascii=False, separators=(",", ":")),
            "user_preferences_json": json.dumps(user_preferences, ensure_ascii=False, separators=(",", ":")),
        })
        response = llm_router.invoke("query_context", [HumanMessage(content=prompt)], QueryContext)
        context = QueryContext.model_validate(parse_json_from_text(response.content))
        if not set(context.collections) <= set(available):
            raise ValueError("Query context contains unknown collections")

        selected = {}
        normalized_fields = {}
        selected_preferences = [item for item in user_preferences if item.get("collection") in context.collections]
        for name in context.collections:
            catalog_fields = available[name]["fields"]
            requested = context.fields[name]
            # Drop hallucinated fields. If all are absent, retain this collection's
            # compact field catalog so generation can choose a real field safely.
            valid = list(dict.fromkeys(field for field in requested if field in catalog_fields))
            for preference in selected_preferences:
                if preference.get("collection") == name and preference.get("field") in catalog_fields:
                    valid.append(preference["field"])
            valid = list(dict.fromkeys(valid))
            if not valid:
                valid = list(catalog_fields)
            normalized_fields[name] = valid
            selected[name] = {"fields": {field: catalog_fields[field] for field in valid}}

        changes = {"fields": normalized_fields}
        if len(context.collections) > 1 and context.complexity == "simple":
            # A single find/count cannot use multiple collections; generation must aggregate.
            changes.update(complexity="complex", operation="aggregate")
        normalized_context = context.model_copy(update=changes).model_dump()
        return {
            "query_context": normalized_context,
            "schema_spec": {"collections": selected},
            "user_preferences": selected_preferences,
        }
