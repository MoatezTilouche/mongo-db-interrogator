"""Choose relevant schema metadata and query hints in one concise LLM call."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import QueryContext
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from database.schema_inspector import compact_schema
from services.query_context_compactor import QueryContextCompactor, compact_json, count_tokens

logger = logging.getLogger(__name__)


class QueryContextAgent:
    @traceable(name="query_context.build")
    def build(self, question: str, catalog: dict[str, Any] | None = None,
              user_preferences: list[dict[str, Any]] | None = None,
              database_knowledge: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if catalog is None:
            return {"error": "schema_not_ready", "detail": "The schema subsystem is not ready."}
        user_preferences = user_preferences or []
        database_knowledge = database_knowledge or []
        available = compact_schema(catalog)
        if not available:
            return {"error": "no_collections", "detail": "The configured MongoDB database has no collections."}

        compacted = QueryContextCompactor().build(question, catalog, database_knowledge, user_preferences)
        available = compacted["schema"]
        prompt = render_prompt("query_context", {
            "question": question,
            "schema_catalog": compact_json(available),
            "database_knowledge_json": compact_json(compacted["knowledge"]),
            "user_preferences_json": compact_json(compacted["preferences"]),
            "relationship_candidates_json": compact_json(compacted["joins"]),
        })
        metrics = compacted["telemetry"]
        metrics["query_context_prompt_tokens_estimate"] = count_tokens(prompt)
        logger.info("query_context_compacted before_tokens=%d after_tokens=%d prompt_tokens=%d budget_exceeded=%s",
                    metrics["query_context_context_tokens_before"], metrics["query_context_context_tokens_after"],
                    metrics["query_context_prompt_tokens_estimate"], metrics["query_context_budget_exceeded"])
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
            for item in database_knowledge:
                if name in item.get("collections", []):
                    valid.extend(field for field in item.get("fields", []) if field in catalog_fields)
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
            "_prompt_compaction": metrics,
        }
