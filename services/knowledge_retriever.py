"""Retrieve a bounded set of active business-knowledge records only."""
from __future__ import annotations

import logging
import math
import re
import time
from typing import Any

from agent.config import KNOWLEDGE_RETRIEVAL_TOP_K
from database.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)
_STOP = {"a", "an", "the", "show", "give", "me", "find", "get", "list", "of", "for", "in", "on", "to", "by", "with", "and", "or"}


def _tokens(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", str(value).lower()) if word not in _STOP}


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class KnowledgeRetriever:
    def __init__(self, repository: KnowledgeRepository | None = None, top_k: int = KNOWLEDGE_RETRIEVAL_TOP_K):
        self.repository = repository or KnowledgeRepository()
        self.top_k = max(1, top_k)

    def retrieve(self, question: str, schema_candidates: dict[str, Any],
                 query_embedding: list[float] | None,
                 full_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        candidate_collections = schema_candidates.get("collections", {})
        candidate_names = set(candidate_collections)
        candidate_fields = {
            name: set((entry.get("fields") or {}).keys())
            for name, entry in candidate_collections.items()
        }
        question_tokens = _tokens(question)
        scored = []
        try:
            # Repository query enforces BOTH status=active and active=true.
            records = self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            records = []
        for record in records:
            collections = set(record.get("collections") or [])
            if full_schema is not None and collections:
                full_collections = full_schema.get("collections", {})
                if not collections <= set(full_collections):
                    continue
                verified_fields = {
                    field for name in collections
                    for field in full_collections.get(name, {}).get("fields", {})
                }
                record_fields = set(record.get("fields") or [])
                if record_fields - verified_fields:
                    continue
            overlap = collections & candidate_names
            if collections and not overlap:
                continue
            fields = set(record.get("fields") or [])
            field_overlap = set().union(*(
                fields & candidate_fields.get(name, set()) for name in overlap
            )) if overlap else set()
            text = str(record.get("text") or "")
            lexical_tokens = _tokens(text)
            lexical = len(question_tokens & lexical_tokens) / max(1, len(question_tokens))
            semantic = max(0.0, _cosine(query_embedding or [], record.get("embedding") or []))
            collection_score = len(overlap) / max(1, len(collections)) if collections else 0.0
            field_score = min(1.0, len(field_overlap) / max(1, len(fields))) if fields else 0.0
            kind = str(record.get("type") or "")
            type_terms = {
                "relationship": {"related", "relationship", "link", "join", "reference"},
                "field_definition": {"meaning", "means", "definition", "represent", "represents"},
                "coded_value": {"code", "coded", "status", "value", "means"},
                "business_rule": {"rule", "revenue", "include", "exclude", "policy"},
            }.get(kind, set())
            type_score = 0.05 if question_tokens & type_terms else 0.0
            if not query_embedding:
                score = 0.50 * lexical + 0.30 * collection_score + 0.15 * field_score + type_score
            else:
                score = 0.42 * semantic + 0.23 * lexical + 0.20 * collection_score + 0.10 * field_score + type_score
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("_id", ""))))
        selected = []
        for score, record in scored[:self.top_k]:
            selected.append({
                "knowledge_id": str(record.get("_id", "")),
                "type": record.get("type"), "text": record.get("text"),
                "collections": record.get("collections", []), "fields": record.get("fields", []),
                "confidence": record.get("confidence", 0),
            })
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = {"knowledge_retrieval_latency_ms": elapsed,
                     "knowledge_retrieval_top_k": self.top_k,
                     "knowledge_candidates_returned": len(selected)}
        logger.info("knowledge_retrieval_completed candidates=%d top_k=%d latency_ms=%.1f",
                    len(selected), self.top_k, elapsed)
        return {"knowledge": selected, "telemetry": telemetry}
