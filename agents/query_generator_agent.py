"""Generate or repair a structured, read-only MongoDB query specification."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import llm_router, LLMRouterError
from agent.output_schemas import GeneratedQuery
from services.query_parameter_binder import parameterize_filter_values
from database.schema_inspector import compact_schema
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from services.mongo_capabilities import get_mongo_capabilities
from agent.telemetry import current_request_id
import logging

logger = logging.getLogger(__name__)


class QueryGeneratorAgent:
    def __init__(self, capabilities=None):
        self.capabilities = capabilities or get_mongo_capabilities()

    def _invoke_json(self, prompt: str, role: str) -> dict[str, Any]:
        prompt += "\n" + self.capabilities.lookup_guidance()
        try:
            response = llm_router.invoke(role, [HumanMessage(content=prompt)], GeneratedQuery)
            parsed = parse_json_from_text((getattr(response, "content", "") or "").strip())
            if not isinstance(parsed, dict):
                raise ValueError("Query generator returned a non-object JSON response")
            generated = GeneratedQuery.model_validate(parsed).model_dump(by_alias=True, exclude_unset=True)
            return parameterize_filter_values(generated)
        except (ValueError, LLMRouterError) as exc:
            if role != "repair":
                raise
            from services.aggregation_validation import QueryValidationError
            code = "repair_failed" if isinstance(exc, LLMRouterError) else "invalid_repair_output"
            raise QueryValidationError("Repair did not return a usable parameterized template",
                                       code=code, repairable=False) from None

    @traceable(name="query_generator.generate")
    def generate(self, question: str, schema_spec: dict[str, Any], plan: dict[str, Any] | None = None,
                 operation_hint: str = "unknown", user_preferences: list[dict[str, Any]] | None = None,
                 database_knowledge: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # `plan` is accepted for compatibility but intentionally never enters the prompt.
        return self._invoke_json(render_prompt("query_generator", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
            "operation_hint": operation_hint,
            "constraints_json": json.dumps({"read_only": True, "default_limit": None, "preserve_requested_limit": True, "server_enforces_output_cap": True}),
            "user_preferences_json": json.dumps(user_preferences or [], ensure_ascii=False, separators=(",", ":")),
            "database_knowledge_json": json.dumps(database_knowledge or [], ensure_ascii=False, separators=(",", ":")),
        }), "query_generator")

    @traceable(name="query_generator.optimize")
    def optimize(self, original_question, query_context, schema_spec, previous_query,
                 refinement, user_preferences=None, database_knowledge=None):
        prompt = (
            "You are refining an existing validated MongoDB query. Do not reinterpret intent from scratch. "
            "Preserve the primary collection, original operation (except an explicitly requested summary), "
            "all valid filters, limits and business rules. Apply only the supplied refinement. "
            "Do not broaden or remove constraints unless the user explicitly replaces them. "
            "Use only the selected schema; do not retrieve new context. Prefer more selective filters. "
            "For summaries compute compact counts/groups in MongoDB, never fetch raw rows to summarize. "
            'Return only {"semantic_intent": <normalized intent>, "template": <revised query>, "parameters": <current values>} JSON. Parameterize filter literal values (including command IDs, statuses, numbers, dates, and membership lists) using named $param objects; keep field names and operators fixed. Parameterize explicit requested limit using {"$param":"limit"}, including top-level pipeline $limit. Keep sort, skip, projection, joins, aggregation structure, and filter operators fixed. Never parameterize $expr field references or structural $exists/$type/$options flags. Keep unrequested defaults literal. Also include semantic_intent: a compact normalized intent without current parameter values. No explanations.\n'
        )
        context = {"mode": "optimize", "original_question": original_question,
                   "query_context": query_context, "selected_schema": compact_schema(schema_spec),
                   "previous_query": previous_query, "refinement": refinement,
                   "user_preferences": user_preferences or [], "database_knowledge": database_knowledge or []}
        return self._invoke_json(prompt + json.dumps(context, ensure_ascii=False, default=str), "query_generator")

    @traceable(name="query_generator.repair")
    def repair(
        self,
        question: str,
        schema_spec: dict[str, Any],
        previous_query: dict[str, Any],
        error: str,
        user_preferences: list[dict[str, Any]] | None = None,
        database_knowledge: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from services.aggregation_validation import lookup_form
        def log_lookups(pipeline, path="pipeline"):
            if not isinstance(pipeline, list):
                return
            for index, stage in enumerate(pipeline):
                if not isinstance(stage, dict):
                    continue
                if "$lookup" in stage:
                    lookup = stage["$lookup"]
                    logger.info("lookup_repair request_id=%s stage_index=%s path=%s form=%s "
                                "reason=structural_failure normalization_attempted=true llm_repair_invoked=true",
                                current_request_id(), index, path, lookup_form(lookup))
                    if isinstance(lookup, dict):
                        log_lookups(lookup.get("pipeline"), f"{path}[{index}].$lookup.pipeline")
                if isinstance(stage.get("$facet"), dict):
                    for name, child in stage["$facet"].items():
                        log_lookups(child, f"{path}[{index}].$facet.{name}")
        log_lookups(previous_query.get("pipeline"))
        logger.info("query_repair request_id=%s llm_repair_invoked=true", current_request_id())
        return self._invoke_json(render_prompt("query_repair", {
            "question": question,
            "schema_json": json.dumps(compact_schema(schema_spec), ensure_ascii=False, default=str),
            "query_json": json.dumps(previous_query, ensure_ascii=False, default=str),
            "error": error,
            "user_preferences_json": json.dumps(user_preferences or [], ensure_ascii=False, separators=(",", ":")),
            "database_knowledge_json": json.dumps(database_knowledge or [], ensure_ascii=False, separators=(",", ":")),
        }), "repair")
