"""Cosine similarity and reference-list validation."""
from __future__ import annotations
from typing import Any

def _cosine(left: list[float], right: list[float]) -> float | None:
    """Keep the raw cosine; unavailable/invalid embeddings cannot prove relevance."""
    from services.knowledge.retrieval.retriever import (
        math,
    )
    if (not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple))
            or not left or not right or len(left) != len(right)):
        return None
    try:
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in [*left, *right]):
            return None
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if not norm_left or not norm_right:
            return None
        similarity = dot / (norm_left * norm_right)
        return similarity if math.isfinite(similarity) else None
    except (TypeError, ValueError, OverflowError):
        return None

_cosine.__module__ = "services.knowledge_retriever"
_cosine.__qualname__ = "_cosine"

def _references(value: Any) -> set[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return None
    return set(value)

_references.__module__ = "services.knowledge_retriever"
_references.__qualname__ = "_references"

