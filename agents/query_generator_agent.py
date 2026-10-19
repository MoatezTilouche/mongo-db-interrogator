"""Generate or repair a structured, read-only MongoDB query specification."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
from langchain_core.messages import HumanMessage

from agent.llm import llm_router, LLMRouterError
from agent.output_schemas import GeneratedQuery
from services.query_parameter_binder import parameterize_filter_values, bind_query_parameters, template_from_concrete
from services.query.generation_examples import generation_examples_text
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from services.mongo_capabilities import get_mongo_capabilities
from agent.telemetry import current_request_id
from services.prompt_context import knowledge_texts, compact_preferences
from services.query_prompt_instructions import normalize_operation, operation_instructions, output_contract
from services.query.prompt_selection.builder import build_prompt_inputs, prompt_query_context
import logging

logger = logging.getLogger(__name__)


class QueryGeneratorAgent:
    def __init__(self, capabilities=None):
        self.capabilities = capabilities or get_mongo_capabilities()

    def _invoke_json(self, prompt: str, role: str, *, include_lookup_guidance: bool = True, question: str = "") -> dict[str, Any]:
        if include_lookup_guidance:
            prompt += "\n" + self.capabilities.lookup_guidance()
        try:
            response = llm_router.invoke(role, [HumanMessage(content=prompt)], GeneratedQuery)
            parsed = parse_json_from_text(getattr(response, "content", "") or "")
            if not isinstance(parsed, dict):
                raise ValueError("Query generator returned a non-object JSON response")
            generated = GeneratedQuery.model_validate(parsed).model_dump(by_alias=True, exclude_unset=True)
            from services.preferences.question_intent import explicit_result_limit
            if explicit_result_limit(question) and not generated["parameters"]:
                concrete = bind_query_parameters(generated["template"], generated["parameters"])
                result = template_from_concrete(concrete, parameterize_limit=True)
                if generated.get("semantic_intent"):
                    result["semantic_intent"] = generated["semantic_intent"]
                return result
            return parameterize_filter_values(generated)
        except (ValueError, LLMRouterError) as exc:
            if role != "repair":
                if isinstance(exc, LLMRouterError):
                    raise
                from services.aggregation_validation import QueryValidationError
                raise QueryValidationError("Query generator did not return a usable JSON query template.",
                                           code="invalid_query_output", repairable=False) from None
            from services.aggregation_validation import QueryValidationError
            code = "repair_failed" if isinstance(exc, LLMRouterError) else "invalid_repair_output"
            raise QueryValidationError("Repair did not return a usable parameterized template",
                                       code=code, repairable=False) from None

    @traceable(name="query_generator.generate")
    def generate(self, question: str, schema_spec: dict[str, Any], plan: dict[str, Any] | None = None,
                 operation_hint: str = "unknown", user_preferences: list[dict[str, Any]] | None = None,
                 database_knowledge: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        from services.query.plans.config import PlanConfig
        config = PlanConfig.from_env()
        if config.enabled:
            from services.query.plans.generation import generate_with_plan
            return generate_with_plan(question, schema_spec, user_preferences or [], database_knowledge or [], config,
                llm_router.invoke, lambda: self._generate_freeform(question, schema_spec, plan, operation_hint,
                                                                  user_preferences, database_knowledge))
        return self._generate_freeform(question, schema_spec, plan, operation_hint, user_preferences, database_knowledge)

    def _generate_freeform(self, question, schema_spec, plan=None, operation_hint="unknown", user_preferences=None,
                           database_knowledge=None):
        # `plan` is accepted for compatibility but intentionally never enters the prompt.
        operation = normalize_operation(operation_hint)
        inputs = build_prompt_inputs(schema_spec, question, database_knowledge, user_preferences)
        return self._invoke_json(render_prompt("query_generator", {
            "today": datetime.now(ZoneInfo("Africa/Tunis")).date().isoformat(),
            "question": question,
            "schema_json": json.dumps(inputs["schema"], ensure_ascii=False, separators=(",", ":"), default=str),
            "operation_hint": operation,
            "constraints_json": json.dumps({"read_only": True, "default_limit": None, "preserve_requested_limit": True, "server_enforces_output_cap": True}),
            "user_preferences_json": json.dumps(compact_preferences(inputs["preferences"]), ensure_ascii=False, separators=(",", ":")),
            "database_knowledge_json": json.dumps(knowledge_texts(inputs["rules"]), ensure_ascii=False, separators=(",", ":")),
            "operation_instructions": operation_instructions(operation),
            "output_contract": output_contract(operation),
            "generation_examples": generation_examples_text(question, inputs["selected_schema"], operation),
        }) + inputs["facts"], "query_generator", include_lookup_guidance=operation in {"aggregate", "unknown"}, question=question)

    @traceable(name="query_generator.optimize")
    def optimize(self, original_question, query_context, schema_spec, previous_query,
                 refinement, user_preferences=None, database_knowledge=None):
        inputs = build_prompt_inputs(schema_spec, refinement or original_question, database_knowledge, user_preferences,
                                     fallback_question=original_question, previous_query=previous_query)
        prompt = (
            "You are refining an existing validated MongoDB query. Do not reinterpret intent from scratch. "
            "Preserve the primary collection, original operation (except an explicitly requested summary), "
            "all valid filters, limits and business rules. Apply only the supplied refinement. "
            "Do not broaden or remove constraints unless the user explicitly replaces them. "
            "Use only the selected schema; do not retrieve new context. Prefer more selective filters. "
            "For summaries compute compact counts/groups in MongoDB, never fetch raw rows to summarize. "
            'Return only {"semantic_intent": <normalized intent>, "template": <revised query>, "parameters": {}} JSON. Use concrete filter literals and a numeric requested limit; the backend parameterizes them. Keep field names, operators, sort, skip, projection, joins and aggregation structure fixed unless the refinement changes them. Never invent IDs or extra filters. No explanations.\n'
        )
        context = {"mode": "optimize", "original_question": original_question,
                   "query_context": prompt_query_context(query_context, inputs["selected_schema"]) if inputs["active"] else query_context, "selected_schema": inputs["schema"],
                   "previous_query": previous_query, "refinement": refinement,
                   "user_preferences": compact_preferences(inputs["preferences"]),
                   "database_knowledge": knowledge_texts(inputs["rules"])}
        operation = normalize_operation(previous_query.get("operation"))
        return self._invoke_json(prompt + json.dumps(context, ensure_ascii=False, default=str)
                                 + inputs["facts"], "query_generator",
                                 include_lookup_guidance=operation in {"aggregate", "unknown"}, question=original_question + "\n" + (refinement or ""))

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
        operation = normalize_operation(previous_query.get("operation"))
        inputs = build_prompt_inputs(schema_spec, question, database_knowledge, user_preferences, previous_query=previous_query)
        return self._invoke_json(render_prompt("query_repair", {
            "question": question,
            "schema_json": json.dumps(inputs["schema"], ensure_ascii=False, separators=(",", ":"), default=str),
            "query_json": json.dumps(previous_query, ensure_ascii=False, default=str),
            "error": error,
            "user_preferences_json": json.dumps(compact_preferences(inputs["preferences"]), ensure_ascii=False, separators=(",", ":")),
            "database_knowledge_json": json.dumps(knowledge_texts(inputs["rules"]), ensure_ascii=False, separators=(",", ":")),
            "operation_instructions": operation_instructions(operation),
            "repair_instructions": (render_prompt("query_aggregate_repair", {"operation": operation})
                                    if operation in {"aggregate", "unknown"} else ""),
            "output_contract": output_contract(operation),
            "generation_examples": generation_examples_text(question, inputs["selected_schema"], operation),
        }) + inputs["facts"], "repair", include_lookup_guidance=operation in {"aggregate", "unknown"}, question=question)
