"""Rank relevant, schema-valid active knowledge before applying the prompt budget."""
from __future__ import annotations

import logging
import math
import time
from typing import Any

from agent.config import (
    KNOWLEDGE_COLLECTION_MATCH_BONUS,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_FEEDBACK_PRIORITY_BONUS,
    KNOWLEDGE_FIELD_MATCH_BONUS,
    KNOWLEDGE_MIN_SEMANTIC_SIMILARITY,
    KNOWLEDGE_RETRIEVAL_TOP_K,
    KNOWLEDGE_SCOPE_MATCH_BONUS,
)
from database.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)


def _cosine(left: list[float], right: list[float]) -> float | None:
    """Keep the raw cosine; unavailable/invalid embeddings cannot prove relevance."""
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


def _references(value: Any) -> set[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return None
    return set(value)


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
        authoritative = (full_schema if full_schema is not None else schema_candidates).get("collections", {})
        scored = []
        try:
            records = self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            records = []
        for record in records:
            knowledge_id = str(record.get("_id", ""))
            source = record.get("source") or "manual"
            semantic = _cosine(query_embedding or [], record.get("embedding") or [])
            collections = _references(record.get("collections", []))
            fields = _references(record.get("fields", []))
            rejection = None
            overlap = (collections or set()) & candidate_names
            # Defense in depth: never rely solely on the repository's filter.
            if record.get("status") != "active" or record.get("active") is not True:
                rejection = "inactive"
            elif not knowledge_id or not isinstance(record.get("text"), str) or not record["text"].strip():
                rejection = "invalid_record"
            elif collections is None or fields is None or record.get("scope", "database") != "database":
                rejection = "invalid_scope"
            elif not collections <= set(authoritative):
                rejection = "unknown_collection"
            else:
                # Fields are flat in storage; each scoped collection may own
                # different fields. ContextBuilder restores them to their owner.
                scoped_names = collections or candidate_names
                allowed_fields = {field for name in scoped_names
                                  for field in authoritative.get(name, {}).get("fields", {})}
                compatible_names = overlap if collections else candidate_names
                if fields - allowed_fields:
                    rejection = "unknown_field"
                elif collections and not overlap:
                    rejection = "collection_mismatch"
                elif record.get("embedding_model") not in (None, KNOWLEDGE_EMBEDDING_MODEL):
                    rejection = "embedding_model_mismatch"
                elif semantic is None:
                    rejection = "embedding_unavailable"
                elif semantic < KNOWLEDGE_MIN_SEMANTIC_SIMILARITY:
                    rejection = "below_semantic_threshold"
            if rejection:
                logger.info("knowledge_candidate knowledge_id=%s source=%s semantic_similarity=%s "
                            "feedback_bonus=0 collection_bonus=0 field_bonus=0 scope_bonus=0 "
                            "ranking_score=None eligible=false reason=%s",
                            knowledge_id, source, semantic, rejection)
                continue

            # Legacy user_feedback records also originated solely in the negative
            # feedback path. Missing source remains ordinary knowledge.
            from_feedback = source in {"negative_feedback", "user_feedback"}
            feedback_bonus = KNOWLEDGE_FEEDBACK_PRIORITY_BONUS if from_feedback else 0.0
            scope_bonus = KNOWLEDGE_SCOPE_MATCH_BONUS if collections <= candidate_names else 0.0
            collection_bonus = KNOWLEDGE_COLLECTION_MATCH_BONUS * len(overlap) / max(1, len(collections))
            retrieved_fields = {field for name in compatible_names
                                for field in candidate_collections.get(name, {}).get("fields", {})}
            field_bonus = KNOWLEDGE_FIELD_MATCH_BONUS * len(fields & retrieved_fields) / max(1, len(fields))
            ranking_score = semantic + feedback_bonus + scope_bonus + collection_bonus + field_bonus
            logger.info("knowledge_candidate knowledge_id=%s source=%s semantic_similarity=%.6f "
                        "feedback_bonus=%.6f collection_bonus=%.6f field_bonus=%.6f scope_bonus=%.6f "
                        "ranking_score=%.6f eligible=true",
                        knowledge_id, source, semantic, feedback_bonus, collection_bonus,
                        field_bonus, scope_bonus, ranking_score)
            scored.append({
                "knowledge_id": knowledge_id, "source": source,
                "created_from_feedback": from_feedback,
                "source_request_id": record.get("source_request_id"),
                "semantic_similarity": semantic, "ranking_score": ranking_score,
                "type": record.get("type"), "text": record["text"],
                "collections": record.get("collections", []), "fields": record.get("fields", []),
                "confidence": record.get("confidence", 0),
            })
        scored.sort(key=lambda item: (-item["ranking_score"], -item["semantic_similarity"], item["knowledge_id"]))
        selected = []
        seen_ids = set()
        seen_rules = set()
        for item in scored:
            identity = (str(item["type"] or ""), " ".join(item["text"].split()).casefold(),
                        tuple(sorted(set(item["collections"]))), tuple(sorted(set(item["fields"]))))
            if item["knowledge_id"] in seen_ids or identity in seen_rules:
                continue
            seen_ids.add(item["knowledge_id"])
            seen_rules.add(identity)
            selected.append(item)
            if len(selected) == self.top_k:
                break
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = {"knowledge_retrieval_latency_ms": elapsed,
                     "knowledge_retrieval_top_k": self.top_k,
                     "knowledge_candidates_returned": len(selected)}
        logger.info("knowledge_retrieval_result ordered_knowledge_ids=%s top_k=%d returned_count=%d latency_ms=%.1f",
                    [item["knowledge_id"] for item in selected], self.top_k, len(selected), elapsed)
        return {"knowledge": selected, "telemetry": telemetry}
