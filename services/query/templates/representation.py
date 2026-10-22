"""Shared template search text, derived from request data and typed value slots.

No model calls or schema-specific names. Search normalization never authorizes
query reuse: the complete resolved wording remains the extraction contract.
"""
from collections import defaultdict
import re

from .request_text import parse_request_context


REPRESENTATION_INPUTS = (
    "source_question", "request_context", "parameter_schema", "original_question",
    "resolved_question", "corrections", "refinements", "generation_instructions", "adjustments",
)


def template_metadata(document):
    context = document.get("request_context")
    if not isinstance(context, dict) and document.get("original_question"):
        context = {key: document[key] for key in (
            "original_question", "resolved_question", "corrections", "refinements", "generation_instructions", "adjustments"
        ) if key in document}
    context = parse_request_context(document.get("source_question") or "", context)
    return {"request_context": context, **context}


def matching_question(document):
    """Retain every meaningful correction/refinement for full-match checks."""
    return template_metadata(document)["resolved_question"]


def _typed_spans(text, specifications):
    from services.template_parameter_extractor import _source_matches

    spans = []
    for spec in specifications.values():
        kind = spec.get("type")
        if (kind not in {"string", "integer", "number", "boolean", "null", "date", "objectId"}
                or spec.get("user_variable") is False or spec.get("source_visible") is False):
            continue
        spans.extend((match.start(), match.end(), kind.lower()) for match in _source_matches(text, spec))
    return spans


def _normalize(text, spans):
    from services.template_parameter_extractor import normalized_question

    # Conflicting/overlapping interpretations stay literal. Never repeatedly
    # substitute into placeholder text or let equal examples hide two slots.
    kinds = defaultdict(set)
    for start, end, kind in spans:
        if start < end:
            kinds[start, end].add(kind)
    candidates = [(start, end, next(iter(types))) for (start, end), types in kinds.items() if len(types) == 1]
    accepted = [(start, end, kind) for start, end, kind in candidates
                if not any(start < other_end and other_start < end and (start, end) != (other_start, other_end)
                           for other_start, other_end, _ in candidates)]
    for start, end, kind in sorted(accepted, reverse=True):
        text = text[:start] + "<" + kind + ">" + text[end:]
    return normalized_question(text)


def template_search_text(document):
    text = matching_question(document)
    return _normalize(text, _typed_spans(text, document.get("parameter_schema") or {}))


def _bounded_free_text_spans(source, question, specifications):
    """Ranking-only support for unquoted multiword values with fixed word count.

    Keep every surrounding word and the number of value tokens unchanged. An
    extra filter or refinement cannot be swallowed by an unbounded text slot.
    Typed extraction deliberately retains its stricter ambiguity policy.
    """
    from services.template_parameter_extractor import _source_matches, _example_text

    variable = [spec for spec in specifications.values()
                if spec.get("user_variable") is not False and spec.get("source_visible") is not False]
    if len(variable) != 1 or variable[0].get("type") != "string":
        return []
    example = _example_text(variable[0])
    if re.fullmatch(r"[\w./:@+-]+", example):
        return []
    matches = _source_matches(source, variable[0])
    if not matches:
        return []
    words = len(example.split())
    if not words:
        return []
    parts, end = [], 0
    for match in matches:
        parts.extend((re.escape(source[end:match.start()]), r"(\S+(?:\s+\S+){" + str(words - 1) + "})"))
        end = match.end()
    parts.append(re.escape(source[end:]))
    matched = re.fullmatch("".join(parts), question, re.I)
    return [(*matched.span(index), "string") for index in range(1, len(matches) + 1)] if matched else []


def query_search_text(question, documents=()):
    """Mask incoming values only when a saved typed wording proves their slots.

    Patterns from the original request can establish value positions even when
    the saved resolved request also contains corrections. These patterns affect
    ranking only; matching and extraction still require the entire resolved text.
    Ambiguous paraphrases remain literal instead of guessing values with a model.
    """
    from services.template_parameter_extractor import _parameter_pattern

    # An incoming user's words are authoritative. Legacy wrapper recovery is
    # restricted to stored documents; lookalike user input must stay intact.
    text = str(question or "").strip()
    spans = []
    for document in documents:
        metadata = template_metadata(document)
        sources = {metadata["original_question"], metadata["resolved_question"]}
        for source in sources:
            try:
                specs, pattern, names, ambiguous = _parameter_pattern({
                    "source_question": source, "parameter_schema": document.get("parameter_schema") or {},
                })
                match = None if ambiguous else re.fullmatch(pattern, text, re.I)
                if match is not None:
                    spans.extend((*match.span(index), specs[name]["type"].lower())
                                 for index, name in enumerate(names, 1))
                elif not ambiguous:
                    spans.extend(_bounded_free_text_spans(source, text, specs))
            except (KeyError, ValueError, TypeError):
                continue
    return _normalize(text, spans)
