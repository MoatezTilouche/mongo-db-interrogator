"""Retry repairable plans; stop when required stored metadata is unavailable."""
from datetime import datetime, timezone
from copy import deepcopy
import json
import logging
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

from agent.prompt_loader import render_prompt
from agent.telemetry import capture_usage, current_request_id
from agent.llm import LLMRouterError
from services.query.prompt_selection.builder import build_prompt_inputs
from services.prompt_context import knowledge_texts, compact_preferences
from services.query_parameter_binder import template_from_concrete
from services.query.dates.normalization import apply_date_normalization
from services.query.dates.resolver import localized
from services.query.profile_config import RuntimeProfileConfig
from services.query.generation_examples import _request_examples
from services.aggregation_validation import QueryValidationError
from .models import PlanOutputContract
from .normalization import normalize_plan
from .directives import approved_directives, directive_field_references
from .resolution import PlanError, resolve_plan
from .compiler import compile_plan

logger = logging.getLogger(__name__)


def plan_examples(question, schema):
    """Reuse already-verified examples without teaching query syntax to a planner."""
    examples = []
    tasks = {"find": "list", "count_documents": "count", "distinct": "distinct"}
    for item in _request_examples(question, schema, "unknown"):
        query = item["output"]["template"]
        if query.get("operation") in tasks:
            examples.append({"question": item["question"], "plan": {
                "collection": query["collection"], "task": tasks[query["operation"]],
                "select": list(query.get("projection") or {}) or None,
                "distinct_field": query.get("field")}})
    return examples


def _errors(exc):
    if isinstance(exc, PlanError):
        return exc.errors
    if isinstance(exc, ValidationError):
        # No literal inputs or arbitrary model-produced names in diagnostics.
        return [".".join(str(part) if isinstance(part, int) or part in {
            "collection", "task", "filters", "field", "op", "value", "select", "order", "dir", "limit",
            "group_by", "metric", "distinct_field", "uses_time", "time_field", "join"} else "output"
            for part in error["loc"]) + ": " + error["type"]
            for error in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
    if isinstance(exc, QueryValidationError):
        return ["compiled_query: " + exc.to_dict()["code"]]
    return ["plan_output_invalid"]


def _tokens(records):
    return sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) for row in records)


def _metadata_error(exc, plan=None):
    """A model cannot repair a missing or unsupported stored date format."""
    if isinstance(exc, QueryValidationError):
        if exc.to_dict()["code"] in {"date_format_unknown", "date_format_unsupported"}:
            exc.repairable = False
            exc.details["repairable"] = False
            return exc
        return None
    if isinstance(exc, PlanError) and len(exc.errors) == 1:
        code = {"order_date_format_unknown": "date_format_unknown",
                "order_date_format_unsupported": "date_format_unsupported"}.get(exc.errors[0])
        if code:
            field = f"{plan.collection}.{plan.order.field}" if plan is not None and plan.order else None
            requirement = ("an exact stored date format is required" if code == "date_format_unknown"
                           else "the stored date format is unsupported for MongoDB date parsing")
            return QueryValidationError(
                f"Cannot order date field {field or 'in this plan'}: {requirement}. "
                "Verify the field's date_format in the stored profile before retrying.",
                code=code, path=field, repairable=False)
    return None


def generate_with_plan(question, schema_spec, preferences, rules, config, invoke, fallback, validator=None, profile_lookup=None, now=None):
    from agents.validator_agent import ValidatorAgent
    rules = deepcopy(list(rules))
    for rule in rules:
        fields = set(rule.get("fields") or [])
        for name in schema_spec.get("collections", {}):
            fields.update(directive_field_references(rule, name))
        rule["fields"] = sorted(fields)
    reason, retries, plan_tokens, free_tokens = "", 0, 0, 0
    try:
        inputs = build_prompt_inputs(schema_spec, question, rules, preferences, profile_lookup=profile_lookup, now=now)
        visible = inputs["selected_schema"]
        contract = PlanOutputContract(visible)
        from services.query.profile_facts import safe_profile
        if not visible.get("collections") or not any(safe_profile(name, profile_lookup) for name in visible["collections"]):
            raise PlanError("stored_profile_required")
        directives = approved_directives(inputs["rules"], question, config.vocabulary(), profile_lookup)
        prompt = render_prompt("query_plan", {
            "today": localized(now or datetime.now(timezone.utc), RuntimeProfileConfig.from_env().zone()).date().isoformat(),
            "question": question,
            "schema_json": json.dumps(inputs["schema"], ensure_ascii=False, default=str),
            "profile_facts": inputs["facts"], "rules_json": json.dumps(knowledge_texts(inputs["rules"]), ensure_ascii=False),
            "preferences_json": json.dumps(compact_preferences(inputs["preferences"]), ensure_ascii=False),
            "examples_json": json.dumps(plan_examples(question, visible), ensure_ascii=False),
            "plan_schema": json.dumps(contract.schema, ensure_ascii=False),
        })
        messages = [HumanMessage(content=prompt)]
        max_retries = min(1, max(0, config.max_retries))
        for attempt in range(max_retries + 1):
            plan = None
            try:
                with capture_usage(propagate=True) as records:
                    try:
                        response = invoke("query_generator", messages, contract)
                    finally:
                        plan_tokens += _tokens(records)
                plan = normalize_plan(getattr(response, "content", response))
                plan, profile, local = resolve_plan(plan, visible, question, config, directives,
                                                    inputs["preferences"], profile_lookup)
                query, intent = compile_plan(plan, profile, visible, question, inputs["rules"], local, profile_lookup, now)
                query = apply_date_normalization(query, question=question, profile_lookup=profile_lookup, now=now)
                query = (validator or ValidatorAgent()).validate(query, schema_spec, profile_lookup=profile_lookup)
                generated = template_from_concrete(query, parameterize_limit=True)
                generated["semantic_intent"] = intent
                logger.info("plan_path request_id=%s path=plan retries=%d plan_tokens=%d free_form_tokens=0",
                            current_request_id(), retries, plan_tokens)
                return generated
            except (ValueError, LLMRouterError, TypeError, KeyError) as exc:
                terminal = _metadata_error(exc, plan)
                if terminal is not None:
                    raise terminal from None
                errors = _errors(exc)
                reason = errors[0]
                if attempt == max_retries or isinstance(exc, LLMRouterError):
                    break
                retries += 1
                logger.info("plan_retry request_id=%s count=%d errors=%d", current_request_id(), retries, len(errors))
                messages = [HumanMessage(content=prompt), HumanMessage(content="Correct these plan errors: " + "; ".join(errors))]
    except (ValueError, TypeError, KeyError, OSError) as exc:
        terminal = _metadata_error(exc)
        if terminal is not None:
            logger.info("plan_path request_id=%s path=blocked reason=%s retries=%d plan_tokens=%d free_form_tokens=0",
                        current_request_id(), terminal.to_dict()["code"], retries, plan_tokens)
            raise terminal from None
        reason = _errors(exc)[0]
    logger.info("plan_fallback request_id=%s reason=%s retries=%d", current_request_id(), reason, retries)
    try:
        with capture_usage(propagate=True) as records:
            try:
                return fallback()
            finally:
                free_tokens = _tokens(records)
    finally:
        logger.info("plan_path request_id=%s path=fallback retries=%d plan_tokens=%d free_form_tokens=%d",
                    current_request_id(), retries, plan_tokens, free_tokens)
