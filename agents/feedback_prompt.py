"""Compact context and API-only output contract for feedback classification."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import re

from database.record_safety import safe_record

_PREFIX = "Original question and any earlier corrections:\n"
_LATEST = "\n\nLatest correction submitted by the user for this question:\n"
_INSTRUCTIONS = "\n\nApply the latest correction to this generation."


def _question_and_corrections(history):
    request = history.get("request_context")
    if isinstance(request, dict) and isinstance(request.get("original_question"), str):
        question = request["original_question"].strip()
        adjustments = request.get("adjustments")
        if isinstance(adjustments, list) and all(
            isinstance(item, dict)
            and item.get("kind") in {"correction", "refinement"}
            and isinstance(item.get("text"), str)
            for item in adjustments
        ):
            earlier = [item["text"] for item in adjustments]
        else:
            # Older structured records did not retain a combined timeline.
            earlier = [text for key in ("corrections", "refinements")
                       for text in (request.get(key) if isinstance(request.get(key), list) else [])
                       if isinstance(text, str)]
        return question, [" ".join(text.split()) for text in earlier if text.strip()]

    question = str(history.get("original_question") or history.get("question") or "").strip()
    effective = str(history.get("effective_question") or history.get("question") or question).strip()
    earlier = []
    while effective.startswith(_PREFIX):
        base, marker, remainder = effective[len(_PREFIX):].rpartition(_LATEST)
        correction, instructions, _ = remainder.partition(_INSTRUCTIONS)
        if not marker or not instructions:
            break
        earlier.append(" ".join(correction.split()))
        effective = base.strip()
    earlier.reverse()
    if not question:
        question = effective
    if effective and effective != question:
        earlier.insert(0, " ".join(effective.split()))
    return question, list(dict.fromkeys(line for line in earlier if line and line != question))


def _mentioned(text, name):
    return bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I))


def _query_usage(query):
    """Include only executed filter/sort/limit, including aggregation matches."""
    used = {key: deepcopy(query[key]) for key in ("filter", "sort", "limit")
            if key in query and query[key] not in (None, {}, [])}
    matches, sorts, limits = [], [], []

    def stages(pipeline):
        if not isinstance(pipeline, list):
            return
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            if "$match" in stage:
                matches.append(deepcopy(stage["$match"]))
            if "$sort" in stage:
                sorts.append(deepcopy(stage["$sort"]))
            if "$limit" in stage:
                limits.append(stage["$limit"])
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict):
                stages(lookup.get("pipeline"))
            facet = stage.get("$facet")
            if isinstance(facet, dict):
                for nested in facet.values():
                    stages(nested)
    stages(query.get("pipeline"))
    if matches:
        used["match"] = matches
    if sorts:
        used["sort_stages"] = sorts
    if limits:
        used["limit_stages"] = limits
    return used


def build_feedback_context(history, comment):
    question, earlier = _question_and_corrections(history)
    query = history.get("validated_query") or {}
    query = query if isinstance(query, dict) else {}
    collections = []
    if isinstance(query.get("collection"), str):
        collections.append(query["collection"])

    def collect(pipeline):
        if not isinstance(pipeline, list):
            return
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict):
                if isinstance(lookup.get("from"), str):
                    collections.append(lookup["from"])
                collect(lookup.get("pipeline"))
            graph = stage.get("$graphLookup")
            if isinstance(graph, dict) and isinstance(graph.get("from"), str):
                collections.append(graph["from"])
            facet = stage.get("$facet")
            if isinstance(facet, dict):
                for nested in facet.values():
                    collect(nested)
    collect(query.get("pipeline"))
    if not collections:
        context = history.get("query_context") or {}
        if isinstance(context, dict):
            collections.extend(name for name in context.get("collections", []) if isinstance(name, str))
    if not collections:
        collections.extend((history.get("selected_schema") or {}).keys())
    output = {"q": question, "cols": list(dict.fromkeys(collections)), "comment": comment}
    if earlier:
        output["prior"] = earlier
    used = _query_usage(query)
    if used:
        output["used"] = used
    mentioned_fields = {}
    for name, metadata in (history.get("selected_schema") or {}).items():
        if not isinstance(metadata, dict):
            continue
        fields = metadata.get("fields", {})
        if not isinstance(fields, (dict, list)):
            continue
        names = [field for field in fields if isinstance(field, str)
                 and (_mentioned(comment, name) or _mentioned(comment, field))]
        if names:
            mentioned_fields[name] = names
    if mentioned_fields:
        output["fields"] = mentioned_fields
    return safe_record(output)


def derive_scope_fields(payload):
    """Enforce only the deterministic relationships stated in the old prompt."""
    if not isinstance(payload, dict):
        return payload
    output = dict(payload)
    scope = output.get("scope")
    if scope == "database":
        output.update(knowledge_candidate=True, reusable_preference=False)
    elif scope == "user":
        output["knowledge_candidate"] = False
    elif scope == "one_off":
        output.update(knowledge_candidate=False, reusable_preference=False)
    if output.get("knowledge_candidate") is False:
        output["knowledge_type"] = None
    return output


@lru_cache(maxsize=1)
def feedback_output_contract():
    from agents.feedback_analysis_agent import FeedbackAnalysis

    class FeedbackOutputContract:
        model_fields = FeedbackAnalysis.model_fields

        @staticmethod
        def model_validate(payload):
            return FeedbackAnalysis.model_validate(derive_scope_fields(payload))

        @staticmethod
        @lru_cache(maxsize=1)
        def feedback_response_format():
            schema = deepcopy(FeedbackAnalysis.model_json_schema())
            schema["properties"].pop("knowledge_candidate")

            def strict(node):
                if isinstance(node, dict):
                    # Some providers do not support length keywords. Preserve
                    # those bounds as instructions when cleaning the API schema;
                    # the original Pydantic contract still enforces them locally.
                    if "maxLength" in node:
                        bound = f"Maximum {node['maxLength']} characters, including spaces."
                        node["description"] = " ".join(filter(None, (node.get("description"), bound)))
                    for key in ("title", "default", "maxLength", "minLength", "maximum", "minimum", "maxItems", "minItems"):
                        node.pop(key, None)
                    if node.get("type") == "object":
                        node["required"] = list(node.get("properties", {}))
                        node["additionalProperties"] = False
                    for child in node.values():
                        strict(child)
                elif isinstance(node, list):
                    for child in node:
                        strict(child)
            strict(schema)
            return {"type": "json_schema", "json_schema": {"name": "feedback_analysis", "strict": True, "schema": schema}}
    return FeedbackOutputContract
