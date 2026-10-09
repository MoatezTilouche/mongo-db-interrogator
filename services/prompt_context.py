"""Project learned context to its business meaning at LLM prompt boundaries."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_RULE_KEYS = ("type", "text", "collections", "fields", "mandatory", "operation", "enforcement")
_PREFERENCE_KEYS = ("type", "collection", "source_collection", "field", "direction", "limit",
                    "latest", "apply_mode", "condition", "preference", "value", "values", "operator")
_BOOKKEEPING_KEYS = {"_id", "knowledge_id", "source_request_id", "source_request_ids", "confidence",
                     "semantic_similarity", "ranking_score", "created_at", "updated_at", "embedding",
                     "embedding_model", "created_from_feedback", "confirmation_count", "normalized_key"}


def compact_knowledge(rules: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Retain complete ordered rules, including enforcement and business metadata.

    IDs used as real business filter values (including an enforced _id filter)
    remain intact. Only record provenance and ranking metadata are omitted.
    """
    compacted = []
    for rule in rules or []:
        entry = {key: deepcopy(rule[key]) for key in _RULE_KEYS if key in rule}
        metadata = rule.get("metadata")
        if isinstance(metadata, dict):
            business_metadata = {key: deepcopy(value) for key, value in metadata.items()
                                 if key not in _BOOKKEEPING_KEYS}
            if business_metadata:
                entry["metadata"] = business_metadata
        compacted.append(entry)
    return compacted


def compact_preferences(preferences: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [{key: deepcopy(item[key]) for key in _PREFERENCE_KEYS if key in item}
            for item in preferences or []]
