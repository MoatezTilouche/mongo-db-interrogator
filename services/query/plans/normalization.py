"""Tolerant transport extraction followed by a small, closed plan contract."""
from collections import Counter
import json
import logging

from agent.llm_structured import extract_text_content, parse_json_from_text
from agent.telemetry import current_request_id
from .models import QueryPlan, PlanFilter, PlanOrder

logger = logging.getLogger(__name__)


def normalize_plan(content):
    actions = Counter()

    def extract(value):
        if isinstance(value, list):
            actions["content_blocks"] += 1
            value = extract_text_content(value)
        if isinstance(value, str):
            actions["fences"] += int(value.strip().startswith("```"))
            value = parse_json_from_text(value)
        for _ in range(6):
            if not isinstance(value, dict) or "collection" in value:
                break
            key = next((key for key in ("plan", "query_plan", "output", "result", "response", "data", "content", "text", "query", "payload", "arguments")
                        if key in value), None)
            if key is None:
                break
            actions["wrapper"] += 1
            value = extract(value[key])
        return value

    try:
        raw = extract(content)
        if not isinstance(raw, dict):
            raise ValueError("plan_object_required")

        def closed(value, model):
            if not isinstance(value, dict):
                raise ValueError("plan_object_required")
            actions["unknown_keys"] += len(set(value) - set(model.model_fields))
            return {key: child for key, child in value.items() if key in model.model_fields}

        plan = closed(raw, QueryPlan)
        plan["filters"] = [closed(item, PlanFilter) for item in plan.get("filters") or []]
        for key in ("select", "group_by"):
            if isinstance(plan.get(key), str):
                plan[key] = [plan[key]]
                actions["coercions"] += 1
        if plan.get("group_by") is None:
            plan["group_by"] = []
        if isinstance(plan.get("order"), dict):
            plan["order"] = closed(plan["order"], PlanOrder)
            direction = plan["order"].get("dir")
            if type(direction) is bool:
                raise ValueError("order_direction_invalid")
            mapped = {"asc": 1, "desc": -1, "1": 1, "-1": -1}.get(str(direction).lower(), direction)
            actions["coercions"] += int(mapped != direction)
            plan["order"]["dir"] = mapped
        if isinstance(plan.get("limit"), str):
            try:
                plan["limit"] = int(plan["limit"])
                actions["coercions"] += 1
            except ValueError:
                raise ValueError("limit_integer_required") from None
        if isinstance(plan.get("uses_time"), str) and plan["uses_time"].lower() in {"true", "false"}:
            plan["uses_time"] = plan["uses_time"].lower() == "true"
            actions["coercions"] += 1
        return QueryPlan.model_validate(plan)
    finally:
        logger.info("plan_normalization request_id=%s actions=%s", current_request_id(), json.dumps(dict(actions), sort_keys=True))
