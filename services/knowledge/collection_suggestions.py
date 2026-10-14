"""Carry selected rule targets past collection selection without changing prompts."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.knowledge_collection_suggestions"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "knowledge_collection_suggestions", _sys.modules[__spec__.name])

from copy import deepcopy
import logging
import re

from agent.config import KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED, KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX
from agent.telemetry import current_request_id
from database.record_safety import safe_record
from services.knowledge_keywords import normalize
from services.knowledge_rule_policy import rule_type

logger = logging.getLogger("services.knowledge_collection_suggestions")
logger.setLevel(logging.INFO)
_STORAGE_WORDS = {"proc", "hist", "history", "historique", "processing", "collection", "log", "logs"}


def _entity_terms(name, entry):
    # Catalogs have no canonical entity key. Shared name/alias terms provide a
    # conservative, inspectable conflict signal; they never remove a collection.
    values = [name, *(entry.get("aliases") or [])]
    terms = {word for value in values for word in re.findall(r"[a-z0-9]+", normalize(value))
             if len(word) >= 3 and word not in _STORAGE_WORDS}
    return {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in terms}


def apply_rule_suggestions(query_context, schema_spec, knowledge, catalog):
    """Return copied context/schema and actual additions; flags off are a no-op."""
    if not KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED:
        return query_context, schema_spec, []
    context, schema = deepcopy(query_context), deepcopy(schema_spec)
    available = catalog.get("collections", {})
    chosen = list(context.get("collections", []))
    additions = []
    for rule in knowledge:
        if rule_type(rule) not in {"business_rule", "relationship"}:
            continue
        for name in rule.get("collections", []):
            if name not in available or name in context.get("collections", []):
                continue
            if len(additions) >= KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX:
                break
            conflicting = [other for other in chosen if other in available and
                           _entity_terms(name, available[name]) & _entity_terms(other, available[other])]
            if conflicting:
                logger.info("rule_suggested_conflict request_id=%s collection=%s agent_collections=%s",
                            safe_record(current_request_id()), safe_record(name), [safe_record(other) for other in conflicting])
            context.setdefault("collections", []).append(name)
            # Restore authoritative fields so the existing generator scope and
            # rule-field restoration can use this collection immediately.
            schema.setdefault("collections", {})[name] = {"fields": deepcopy(available[name].get("fields", {}))}
            context.setdefault("fields", {})[name] = list(available[name].get("fields", {}))
            context.setdefault("collection_selection_reasons", {})[name] = "rule_suggested"
            additions.append(name)
            logger.info("rule_suggested_collection request_id=%s collection=%s knowledge_id=%s reason=rule_suggested",
                        safe_record(current_request_id()), safe_record(name), safe_record(str(rule.get("knowledge_id", ""))))
    return context, schema, additions
