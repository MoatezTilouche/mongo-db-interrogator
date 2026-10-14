"""Reject explicit command/request mismatches without approximating intent."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.query_entity_guard"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "query_entity_guard", _sys.modules[__spec__.name])

import re
from services.knowledge_keywords import normalize


def entity_families(text):
    text = normalize(text or "")
    # Identifier namespaces and named ID fields are stronger than incidental
    # nouns such as the verb 'request' in a correction's surrounding prose.
    strong = set()
    if re.search(r"(?<!\w)cmd[-_]\d[\w-]*\b|(?<!\w)command_?id(?!\w)", text):
        strong.add("command")
    if re.search(r"(?<!\w)req[-_]\d[\w-]*\b|(?<!\w)request_?id(?!\w)", text):
        strong.add("request")
    if strong:
        return strong
    families = set()
    if re.search(r"\b(?:commands?|commandes?|orders?)\b", text):
        families.add("command")
    if re.search(r"\b(?:requests?|requetes?|demandes?)\b", text):
        families.add("request")
    return families


def entity_mismatch(question, supported):
    requested = entity_families(question)
    return bool(requested and supported and not requested <= supported)


def template_entity_mismatch(document, question):
    supported = set()
    specs = document.get("parameter_schema")
    for spec in specs.values() if isinstance(specs, dict) else ():
        if not isinstance(spec, dict) or spec.get("user_variable") is False:
            continue
        paths = spec.get("source_paths") or [spec.get("source_path", "")]
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            supported.update(entity_families(path))
        example = spec.get("example")
        if isinstance(example, str):
            # Only identifier evidence from examples, not arbitrary status text.
            for identifier in re.findall(r"(?<!\w)(?:CMD|REQ)[-_]\d[\w-]*\b", example, re.I):
                supported.update(entity_families(identifier))
    if not supported:
        supported = entity_families(document.get("source_question") or document.get("semantic_text"))
    return entity_mismatch(question, supported)


def rule_entity_mismatch(record, question):
    # Text is authoritative for rule applicability. Do not infer entity from a
    # collection name or a source question that might concern a different join.
    supported = entity_families(record.get("corrected_meaning") or record.get("text"))
    requested = entity_families(question)
    # A join can need separate rules for each entity. Reject disjoint families,
    # while retaining a command-only rule in a question naming both entities.
    return bool(requested and supported and not requested & supported)
