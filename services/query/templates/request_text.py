"""Separate user intent from the application's generation instructions.

Structured metadata is authoritative for new requests. The narrow legacy parser
also makes existing feedback/refinement histories usable without embedding the
application's correction wrapper.
"""
from __future__ import annotations

from copy import deepcopy


CORRECTION_INSTRUCTION = (
    "Apply the latest correction to this generation. It overrides conflicting earlier details; "
    "preserve all unrelated requested filters, dates, limits, and output requirements. "
    "Use valid schema references and retain read-only query constraints."
)
_ORIGINAL_PREFIX = "Original question and any earlier corrections:\n"
_CORRECTION_MARKER = "\n\nLatest correction submitted by the user for this question:\n"
_INSTRUCTION_SUFFIX = "\n\n" + CORRECTION_INSTRUCTION
_REFINEMENT_MARKER = "\nRefinement: "


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _strings(value):
    return [_text(item) for item in value if _text(item)] if isinstance(value, (list, tuple)) else []


def _base(question):
    return {"original_question": question, "corrections": [],
            "generation_instructions": [], "refinements": [],
            "adjustments": [],
            "resolved_question": question}


def _append(context, kind, text):
    context[kind].append(text)
    context["adjustments"].append({
        "kind": "correction" if kind == "corrections" else "refinement", "text": text})
    label = "Correction" if kind == "corrections" else "Refinement"
    context["resolved_question"] += f"\n{label}: {text}"


def _parse_legacy(question):
    if question.startswith(_ORIGINAL_PREFIX):
        end = question.rfind(_INSTRUCTION_SUFFIX)
        suffix = question[end + len(_INSTRUCTION_SUFFIX):] if end >= 0 else ""
        if end >= 0 and (not suffix or suffix.startswith(_REFINEMENT_MARKER)):
            body = question[len(_ORIGINAL_PREFIX):end]
            previous, marker, correction = body.rpartition(_CORRECTION_MARKER)
            if marker and previous and correction:
                result = _parse_legacy(previous)
                _append(result, "corrections", correction.strip())
                if CORRECTION_INSTRUCTION not in result["generation_instructions"]:
                    result["generation_instructions"].append(CORRECTION_INSTRUCTION)
                for refinement in suffix.split(_REFINEMENT_MARKER)[1:]:
                    if refinement.strip():
                        _append(result, "refinements", refinement.strip())
                return result
        # A lookalike user message is not a known application wrapper.
        return _base(question)
    original, *refinements = question.split(_REFINEMENT_MARKER)
    result = _base(original.strip())
    for refinement in refinements:
        if refinement.strip():
            _append(result, "refinements", refinement.strip())
    return result


def parse_request_context(question, context=None):
    """Return original question, changes, instructions, and clean resolved text.

    Keep the resolved text's chronology when structured metadata is available;
    corrections can follow refinements, and later changes can override earlier
    ones. No schema names, values, or business meaning are inferred here.
    """
    if not isinstance(context, dict):
        return _parse_legacy(_text(question))
    original = _text(context.get("original_question"))
    if not original:
        original = _parse_legacy(_text(question))["original_question"]
    result = _base(original)
    for key in ("corrections", "generation_instructions", "refinements"):
        result[key] = _strings(context.get(key))
    adjustments = context.get("adjustments")
    if isinstance(adjustments, list):
        result["adjustments"] = [
            {"kind": item["kind"], "text": _text(item.get("text"))}
            for item in adjustments if isinstance(item, dict)
            and item.get("kind") in ("correction", "refinement") and _text(item.get("text"))]
        # Ordered events also recover partial structured records missing arrays.
        for key, kind in (("corrections", "correction"), ("refinements", "refinement")):
            remaining = [item["text"] for item in result["adjustments"] if item["kind"] == kind]
            for change in result[key]:
                if change in remaining:
                    remaining.remove(change)
                else:
                    result["adjustments"].append({"kind": kind, "text": change})
            result[key] = [item["text"] for item in result["adjustments"] if item["kind"] == kind]
    else:
        result["adjustments"] = [
            {"kind": kind, "text": change}
            for key, kind in (("corrections", "correction"), ("refinements", "refinement"))
            for change in result[key]]
    resolved = _text(context.get("resolved_question"))
    if resolved and (not isinstance(adjustments, list) or not result["adjustments"]):
        result["resolved_question"] = resolved
    else:
        for change in result["adjustments"]:
            label = "Correction" if change["kind"] == "correction" else "Refinement"
            result["resolved_question"] += f"\n{label}: {change['text']}"
    return result


def extend_request_context(question, context=None, *, correction=None,
                           refinement=None, generation_instruction=None):
    """Append an actual user change without re-parsing generation prose."""
    result = deepcopy(parse_request_context(question, context))
    for key, value in (("corrections", correction), ("refinements", refinement)):
        change = _text(value)
        if change:
            _append(result, key, change)
    instruction = _text(generation_instruction)
    if instruction and instruction not in result["generation_instructions"]:
        result["generation_instructions"].append(instruction)
    return result


def resolved_request_text(question, context=None):
    return parse_request_context(question, context)["resolved_question"]
